//! Where a node's lateral inflow comes from.
//!
//! Two sources, behind one interface:
//!
//! - [`FlowSource::Csv`] reads the per-catchment CSV files ngen or bmi-driver wrote to disk.
//! - [`FlowSource::Bmi`] runs the catchment models itself, in process, and never writes a file.
//!
//! [`FlowSource`] is the configuration and is shared between threads; [`FlowProvider`] is the
//! per-thread handle that actually produces flows. They are separate types because a BMI provider
//! owns loaded model libraries, which cannot be shared across threads.

use std::collections::VecDeque;
use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::io::csv::load_external_flows;

/// Seconds in an hour, the interval the router's external forcing is assumed to use.
#[cfg(feature = "bmi")]
const SECONDS_PER_HOUR: f32 = 3600.0;
/// Square metres in a square kilometre.
#[cfg(feature = "bmi")]
const SQM_PER_SQKM: f32 = 1_000_000.0;

/// How to obtain lateral inflows, and enough information to answer questions about the
/// simulation period before any node is routed.
#[derive(Debug, Clone)]
pub enum FlowSource {
    /// Per-catchment CSV files, one column per variable.
    Csv {
        dir: PathBuf,
        /// Column to read. `None` falls back to the third column, as before.
        variable: Option<String>,
    },
    /// Run the catchment models through bmi-driver.
    #[cfg(feature = "bmi")]
    Bmi {
        /// Directory holding `config/`, `forcings/` and the model libraries.
        data_dir: PathBuf,
        /// Realization config. `None` means `<data_dir>/config/realization.json`.
        config: Option<PathBuf>,
        /// Model output variable to route.
        variable: String,
    },
}

impl FlowSource {
    /// A per-thread handle. Call this inside each worker thread, never once and then share it:
    /// a BMI provider owns dynamically loaded model libraries and is deliberately not `Send`.
    pub fn provider(&self) -> Result<FlowProvider> {
        match self {
            FlowSource::Csv { dir, variable } => Ok(FlowProvider::Csv {
                dir: dir.clone(),
                variable: variable.clone(),
            }),
            #[cfg(feature = "bmi")]
            FlowSource::Bmi {
                data_dir,
                config,
                variable,
            } => {
                let sim = match config {
                    Some(cfg) => bmi_driver::Simulation::with_config(data_dir, cfg),
                    None => bmi_driver::Simulation::open(data_dir),
                }
                .with_context(|| {
                    format!("Failed to open bmi-driver data directory: {:?}", data_dir)
                })?;
                Ok(FlowProvider::Bmi {
                    sim,
                    variable: variable.clone(),
                    to_depth_per_hour: None,
                })
            }
        }
    }

    /// Number of external (hourly) timesteps, and the simulation start time.
    pub fn simulation_span(&self, sample_id: u32) -> Result<(usize, chrono::NaiveDateTime)> {
        match self {
            FlowSource::Csv { dir, .. } => csv_simulation_span(dir, sample_id),
            #[cfg(feature = "bmi")]
            FlowSource::Bmi {
                data_dir, config, ..
            } => {
                // The realization config knows the period; nothing has to be run to find it.
                let sim = match config {
                    Some(cfg) => bmi_driver::Simulation::with_config(data_dir, cfg),
                    None => bmi_driver::Simulation::open(data_dir),
                }
                .with_context(|| {
                    format!("Failed to open bmi-driver data directory: {:?}", data_dir)
                })?;
                let start = chrono::DateTime::from_timestamp(sim.start_epoch(), 0)
                    .ok_or_else(|| anyhow::anyhow!("Invalid start time: {}", sim.start_epoch()))?
                    .naive_utc();
                Ok((sim.total_steps(), start))
            }
        }
    }
}

/// A thread-local handle that turns a node id into its lateral inflow series.
pub enum FlowProvider {
    Csv {
        dir: PathBuf,
        variable: Option<String>,
    },
    #[cfg(feature = "bmi")]
    Bmi {
        sim: bmi_driver::Simulation,
        variable: String,
        /// Conversion from the model's output units to metres per hour. Resolved on first use,
        /// once the models have been loaded and can report their units.
        to_depth_per_hour: Option<bmi_driver::units::UnitConversion>,
    },
}

impl FlowProvider {
    /// Lateral inflow for one node, in cubic metres per second.
    ///
    /// Catchment models report a depth per unit time over the catchment; the router wants a
    /// volumetric rate, hence the multiplication by area.
    pub fn load(&mut self, id: u32, area_sqkm: f32) -> Result<VecDeque<f32>> {
        match self {
            FlowProvider::Csv { dir, variable } => {
                let path = dir.join(format!("cat-{}.csv", id));
                load_external_flows(path, &id, variable.as_deref(), area_sqkm)
            }
            #[cfg(feature = "bmi")]
            FlowProvider::Bmi {
                sim,
                variable,
                to_depth_per_hour,
            } => {
                let location = format!("cat-{}", id);

                if to_depth_per_hour.is_none() {
                    *to_depth_per_hour = Some(resolve_depth_conversion(sim, &location, variable)?);
                }
                let conversion = to_depth_per_hour.as_ref().expect("just resolved");

                let values = sim
                    .run_variable(&location, variable)
                    .with_context(|| format!("Failed to run models for {}", location))?;

                let area_sqm = area_sqkm * SQM_PER_SQKM;
                Ok(values
                    .into_iter()
                    .map(|v| (conversion.convert(v) as f32 * area_sqm) / SECONDS_PER_HOUR)
                    .collect())
            }
        }
    }
}

/// Work out how to get from the model's reported units to metres per hour.
///
/// The CSV path hardcodes the assumption that the routed variable is a depth in metres per hour.
/// Running the models directly means the units are available, so use them: a model reporting
/// mm h-1 or m s-1 converts correctly instead of being off by a factor of a thousand.
#[cfg(feature = "bmi")]
fn resolve_depth_conversion(
    sim: &mut bmi_driver::Simulation,
    location: &str,
    variable: &str,
) -> Result<bmi_driver::units::UnitConversion> {
    let units = sim.var_units(location, variable).unwrap_or_default();
    if units.is_empty() {
        eprintln!(
            "Warning: model does not report units for '{}'; assuming m h-1",
            variable
        );
        return Ok(bmi_driver::units::UnitConversion::identity("m h-1"));
    }

    let timestep = Some(sim.output_interval() as f64);
    let (conversion, warning) =
        bmi_driver::units::find_conversion_or_identity(&units, "m h-1", timestep);
    if let Some(w) = warning {
        return Err(anyhow::anyhow!(
            "Cannot convert '{}' from {} to m h-1: {}",
            variable,
            units,
            w
        ));
    }
    Ok(conversion)
}

/// Read the simulation period out of a sample CSV file.
fn csv_simulation_span(
    dir: &std::path::Path,
    sample_id: u32,
) -> Result<(usize, chrono::NaiveDateTime)> {
    let file_name = dir.join(format!("cat-{}.csv", sample_id));
    let content = std::fs::read_to_string(&file_name)
        .with_context(|| format!("Failed to read file: {:?}", file_name))?;

    match content.lines().count() {
        0 => {
            return Err(anyhow::anyhow!("CSV file is empty: {:?}", file_name))
                .with_context(|| format!("Failed to read CSV file: {:?}", file_name))
        }
        1 => {
            return Err(anyhow::anyhow!(
                "CSV file only contains header: {:?}",
                file_name
            ))
            .with_context(|| format!("Failed to read CSV file: {:?}", file_name))
        }
        _ => {}
    }

    let max_external_steps = content.lines().count().saturating_sub(2);

    let line = content
        .lines()
        .nth(1)
        .with_context(|| format!("Failed to read second line of CSV file: {:?}", file_name))?;
    let time = line
        .split(',')
        .nth(1)
        .with_context(|| format!("Failed to parse time from CSV line: {:?}", line))?;

    let reference_time = chrono::NaiveDateTime::parse_from_str(time, "%Y-%m-%d %H:%M:%S")
        .context("Failed to parse reference time")?;

    Ok((max_external_steps, reference_time))
}
