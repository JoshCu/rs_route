use crate::config::EXTERNAL_TIMESTEP_SECONDS;
use crate::kernel::muskingum::{MuskingumCungeKernel, SecantBracket};
use anyhow::{Context, Result};
use clap::Parser;
use colored::Colorize;
use num_cpus;
use std::path::PathBuf;
/// Network routing simulation tool
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Route directory path
    route_dir: PathBuf,

    /// Path to the GeoPackage (.gpkg) hydrofabric file
    #[arg(long)]
    hf: Option<PathBuf>,

    /// Path to the input directory containing CSV files
    #[arg(short = 'i', long)]
    input_dir: Option<PathBuf>,

    /// Path to the output directory
    #[arg(short, long)]
    output_dir: Option<PathBuf>,

    /// Internal timestep in seconds. Must divide 3600 exactly. Larger values
    /// are cheaper and cost little accuracy: 900 is ~3x faster than 300 for a
    /// ~0.06% volume bias.
    #[arg(short = 't', long, default_value_t = 300)]
    internal_timestep_seconds: usize,
    #[arg(short, long, default_value_t = MuskingumCungeKernel::TRouteModernized)]
    kernel: MuskingumCungeKernel,
    #[arg(short, long, default_value_t = num_cpus::get())]
    num_threads: usize,

    /// Start the depth solver from a tighter bracket. Converges in fewer
    /// iterations, so depths are slightly less converged: ~1.45x faster with
    /// the SIMD kernel, total flow shifted by ~0.07%. Rust kernels only.
    #[arg(long, default_value_t = false)]
    fast_converge: bool,
}
pub fn print_banner(config: &Config) {
    eprintln!("   {}", "🌊 Route RS".cyan().bold());
    eprintln!("  Kernel:   {}", format!("{}", config.kernel).green());
    eprintln!("  Timestep: {}s", config.internal_timestep_seconds);
    eprintln!("  Threads:  {}", config.num_threads);
    if config.secant_bracket.high < SecantBracket::WIDE.high {
        eprintln!("  Solver:   {}", "fast-converge".yellow());
    }
    eprintln!(
        "  GeoPackage: {}",
        config.gpkg_file.display().to_string().dimmed()
    );
    eprintln!();
}
pub struct Config {
    pub csv_dir: PathBuf,
    pub gpkg_file: PathBuf,
    pub internal_timestep_seconds: usize,
    pub output_dir: PathBuf,
    pub kernel: MuskingumCungeKernel,
    pub num_threads: usize,
    pub secant_bracket: SecantBracket,
}

/// The routing clock advances `3600 / dt` steps per forcing step, so a `dt`
/// that does not divide 3600 silently runs the model slow: `-t 700` would take
/// 5 steps of 700 s per forcing hour, losing 100 s every hour.
fn validate_timestep(dt: usize) -> Result<usize> {
    if dt == 0 || dt > EXTERNAL_TIMESTEP_SECONDS || EXTERNAL_TIMESTEP_SECONDS % dt != 0 {
        let valid: Vec<String> = (1..=EXTERNAL_TIMESTEP_SECONDS)
            .filter(|d| EXTERNAL_TIMESTEP_SECONDS % d == 0 && *d >= 60)
            .map(|d| d.to_string())
            .collect();
        return Err(anyhow::anyhow!(
            "Internal timestep of {}s does not divide the {}s forcing step evenly, \
             so the routing clock would drift from the forcing clock. \
             Valid values of 60s or more: {}",
            dt,
            EXTERNAL_TIMESTEP_SECONDS,
            valid.join(", ")
        ));
    }
    Ok(dt)
}

pub fn get_args() -> Result<Config> {
    let args = Args::parse();

    let root_dir = args.route_dir;
    let csv_dir = args.input_dir.unwrap_or_else(|| root_dir.join("outputs").join("ngen"));
    let config_dir = root_dir.join("config");
    let output_dir = args.output_dir.unwrap_or_else(|| root_dir.join("outputs").join("troute"));

    // Check directories valid
    if !root_dir.exists() || !root_dir.is_dir() {
        return Err(anyhow::anyhow!(
            "Given root directory does not exist or is not a directory: {:?}",
            root_dir
        ))
        .with_context(|| format!("Failed to access root directory: {:?}", root_dir));
    }

    let dirs_to_check: Vec<&PathBuf> = if args.hf.is_some() {
        vec![&csv_dir, &output_dir]
    } else {
        vec![&csv_dir, &config_dir, &output_dir]
    };
    let mut missing_dirs = Vec::new();
    for dir in dirs_to_check {
        if !dir.exists() || !dir.is_dir() {
            missing_dirs.push(dir);
        }
    }
    if !missing_dirs.is_empty() {
        return Err(anyhow::anyhow!(
            "Missing required directories: {:?}",
            missing_dirs
        ))
        .with_context(|| format!("Failed to access required directories: {:?}", missing_dirs));
    }

    // Use provided gpkg file or find one in the config directory
    let gpkg_file = if let Some(hf) = args.hf {
        if !hf.exists() {
            return Err(anyhow::anyhow!(
                "Specified hydrofabric file does not exist: {:?}",
                hf
            ));
        }
        hf
    } else {
        config_dir
            .read_dir()
            .context("Failed to read config directory")?
            .filter_map(Result::ok)
            .find(|entry| entry.path().extension().map_or(false, |ext| ext == "gpkg"))
            .ok_or_else(|| anyhow::anyhow!("No .gpkg file found in config directory"))?
            .path()
    };
    let cfg = Config {
        csv_dir,
        gpkg_file,
        internal_timestep_seconds: validate_timestep(args.internal_timestep_seconds)?,
        output_dir,
        kernel: args.kernel,
        num_threads: args.num_threads,
        secant_bracket: if args.fast_converge {
            SecantBracket::TIGHT
        } else {
            SecantBracket::WIDE
        },
    };
    print_banner(&cfg);
    Ok(cfg)
}

#[cfg(test)]
mod tests {
    // Same-file tests for CLI module
    use super::*;

    // Test Args parsing with default values
    #[test]
    fn test_args_parsing_defaults() {
        let args = Args::parse_from(["test", "test_route_dir"]);
        assert_eq!(args.route_dir, PathBuf::from("test_route_dir"));
        assert_eq!(args.internal_timestep_seconds, 300);
        match args.kernel {
            MuskingumCungeKernel::TRouteModernized => {}
            _ => panic!("Expected default kernel to be TRouteModernized"),
        }
    }
    // Test Args parsing with custom values
    #[test]
    fn test_args_parsing_custom() {
        let args = Args::parse_from([
            "test",
            "test_route_dir",
            "-t",
            "600",
            "-k",
            "t-route-legacy",
        ]);
        assert_eq!(args.route_dir, PathBuf::from("test_route_dir"));
        assert_eq!(args.internal_timestep_seconds, 600);
        match args.kernel {
            MuskingumCungeKernel::TRouteLegacy => {}
            _ => panic!("Expected kernel to be TRouteLegacy"),
        }
    }
    // Impossible to test get_args(), as it pulls from the program's actual command line arguments, which we can't easily manipulate.
    // #[test]
    // fn test_get_args_invalid_root() {
    //     let result = get_args();
    //     assert!(result.is_err());
    // }

    /// A timestep that does not divide the forcing step would run the routing
    /// clock slow, so it has to be rejected rather than silently truncated.
    #[test]
    fn test_timestep_must_divide_forcing_step() {
        for good in [60, 300, 450, 900, 1800, 3600] {
            assert_eq!(validate_timestep(good).unwrap(), good, "{} should be valid", good);
        }
        for bad in [0, 700, 500, 2400, 5000] {
            let err = validate_timestep(bad).unwrap_err().to_string();
            assert!(
                err.contains("does not divide"),
                "{} should be rejected, got: {}",
                bad,
                err
            );
        }
    }

    /// Every accepted timestep must give a whole number of internal steps that
    /// add back up to exactly one forcing step.
    #[test]
    fn test_accepted_timesteps_reconstruct_the_forcing_step() {
        for dt in 1..=EXTERNAL_TIMESTEP_SECONDS {
            if validate_timestep(dt).is_ok() {
                let steps = EXTERNAL_TIMESTEP_SECONDS / dt;
                assert_eq!(steps * dt, EXTERNAL_TIMESTEP_SECONDS, "dt={} drifts", dt);
            }
        }
    }
}
