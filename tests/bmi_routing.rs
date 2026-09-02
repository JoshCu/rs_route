//! Routing with the catchment models run in process must give the same answer as routing from
//! the CSV files those models would have written.
//!
//! The models here are a synthetic one-bucket BMI model compiled on the fly, so this needs no
//! hydrological model libraries installed -- only a C compiler.

#![cfg(feature = "bmi")]

use std::path::{Path, PathBuf};
use std::process::Command;

const LOCATION: &str = "cat-486888";
const N_STEPS: usize = 25; // 24 hourly intervals, both ends included
const START_EPOCH: i64 = 1_262_304_000; // 2010-01-01 00:00:00 UTC
const GPKG: &str = "tests/one_cat/config/cat-486888_subset.gpkg";

fn precip_mm_per_hour(step: usize) -> f32 {
    0.4 + (step % 6) as f32 * 0.35
}

fn build_model(out_dir: &Path) -> Option<PathBuf> {
    let src = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/bucket_bmi.c");
    let lib = out_dir.join("libbucketbmi.so");
    for cc in ["cc", "gcc", "clang"] {
        let ok = Command::new(cc)
            .args(["-shared", "-fPIC", "-O1", "-o"])
            .arg(&lib)
            .arg(&src)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if ok {
            return Some(lib);
        }
    }
    eprintln!("skipping: no working C compiler found");
    None
}

fn write_forcings(path: &Path) {
    let mut file = netcdf::create(path).unwrap();
    file.add_dimension("catchment-id", 1).unwrap();
    file.add_dimension("time", N_STEPS).unwrap();

    file.add_string_variable("ids", &["catchment-id"]).unwrap();
    file.variable_mut("ids")
        .unwrap()
        .put_string(LOCATION, 0)
        .unwrap();

    file.add_variable::<i64>("Time", &["catchment-id", "time"])
        .unwrap();
    let times: Vec<i64> = (0..N_STEPS).map(|s| START_EPOCH + s as i64 * 3600).collect();
    file.variable_mut("Time")
        .unwrap()
        .put_values(&times, (0, ..))
        .unwrap();

    file.add_variable::<f32>("RAINRATE", &["catchment-id", "time"])
        .unwrap();
    let mut rain = file.variable_mut("RAINRATE").unwrap();
    rain.put_attribute("units", "mm h-1").unwrap();
    let values: Vec<f32> = (0..N_STEPS).map(precip_mm_per_hour).collect();
    rain.put_values(&values, (0, ..)).unwrap();
}

const REALIZATION: &str = r#"{
  "global": {
    "formulations": [{
      "name": "bmi_multi",
      "params": {
        "main_output_variable": "Q_OUT",
        "output_variables": ["Q_OUT"],
        "modules": [{
          "name": "bmi_c",
          "params": {
            "model_type_name": "Bucket",
            "library_file": "libs/libbucketbmi.so",
            "init_config": "./config/cat_config/Bucket/{{id}}.ini",
            "registration_function": "register_bmi_bucket",
            "main_output_variable": "Q_OUT",
            "variables_names_map": { "precip_rate": "RAINRATE" }
          }
        }]
      }
    }],
    "forcing": { "path": "./forcings/forcings.nc" }
  },
  "time": {
    "start_time": "2010-01-01 00:00:00",
    "end_time": "2010-01-02 00:00:00",
    "output_interval": 3600
  }
}
"#;

/// A bmi-driver data directory carrying the hydrofabric this crate's test dataset uses.
fn make_data_dir(root: &Path) -> Option<()> {
    std::fs::create_dir_all(root.join("config/cat_config/Bucket")).unwrap();
    std::fs::create_dir_all(root.join("forcings")).unwrap();
    std::fs::create_dir_all(root.join("libs")).unwrap();
    std::fs::create_dir_all(root.join("outputs/ngen")).unwrap();

    build_model(&root.join("libs"))?;
    write_forcings(&root.join("forcings/forcings.nc"));

    let gpkg = Path::new(env!("CARGO_MANIFEST_DIR")).join(GPKG);
    std::fs::copy(&gpkg, root.join("config/cat-486888_subset.gpkg")).unwrap();

    std::fs::write(
        root.join(format!("config/cat_config/Bucket/{LOCATION}.ini")),
        "drain_fraction=0.2\ninitial_storage=0.005\n",
    )
    .unwrap();
    std::fs::write(root.join("config/realization.json"), REALIZATION).unwrap();
    Some(())
}

fn bmi_source(root: &Path) -> rs_route::io::flows::FlowSource {
    rs_route::io::flows::FlowSource::Bmi {
        data_dir: root.to_path_buf(),
        config: None,
        variable: "Q_OUT".to_string(),
    }
}

#[test]
fn bmi_provider_returns_flows_in_cubic_metres_per_second() {
    let tmp = tempfile::tempdir().unwrap();
    let Some(()) = make_data_dir(tmp.path()) else {
        return;
    };

    let source = bmi_source(tmp.path());
    let (steps, start) = source.simulation_span(486888).unwrap();
    assert_eq!(steps, 24);
    assert_eq!(start.and_utc().timestamp(), START_EPOCH);

    let mut provider = source.provider().unwrap();
    let area_sqkm = 14.710_95_f32;
    let flows = provider.load(486888, area_sqkm).unwrap();
    assert_eq!(flows.len(), steps + 1);

    // The model reports Q_OUT in m h-1. Reproduce the expected first value independently:
    // depth per hour -> volume per second over the catchment.
    let mut storage = 0.005_f64;
    storage += precip_mm_per_hour(0) as f64 * 0.001;
    let q_depth = storage * 0.2;
    let expected = (q_depth * (area_sqkm as f64) * 1_000_000.0 / 3600.0) as f32;
    assert!(
        (flows[0] - expected).abs() < 1e-3,
        "got {}, expected {}",
        flows[0],
        expected
    );
    assert!(flows.iter().all(|f| *f > 0.0));
}

#[test]
fn bmi_and_csv_routing_agree() {
    let tmp = tempfile::tempdir().unwrap();
    let Some(()) = make_data_dir(tmp.path()) else {
        return;
    };

    // Same models, same forcings: once through a CSV file, once straight into the router.
    let csv_dir = tmp.path().join("outputs/ngen");
    let mut sim = bmi_driver::Simulation::open(tmp.path()).unwrap();
    let q = sim.run_variable(LOCATION, "Q_OUT").unwrap();
    let mut csv = String::from("Time Step,Time,Q_OUT\n");
    for (step, value) in q.iter().enumerate() {
        csv.push_str(&format!("{step},2010-01-01 00:00:00,{value:.9}\n"));
    }
    std::fs::write(csv_dir.join(format!("{LOCATION}.csv")), csv).unwrap();

    let area_sqkm = 14.710_95_f32;
    let mut from_csv = rs_route::io::flows::FlowSource::Csv {
        dir: csv_dir,
        variable: Some("Q_OUT".to_string()),
    }
    .provider()
    .unwrap();
    let mut from_bmi = bmi_source(tmp.path()).provider().unwrap();

    let a = from_csv.load(486888, area_sqkm).unwrap();
    let b = from_bmi.load(486888, area_sqkm).unwrap();

    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        // The CSV round-trip costs some precision; nothing else should differ.
        assert!(
            (x - y).abs() <= x.abs() * 1e-5 + 1e-4,
            "step {i}: csv {x} vs bmi {y}"
        );
    }
}

#[test]
fn unknown_location_is_an_error() {
    let tmp = tempfile::tempdir().unwrap();
    let Some(()) = make_data_dir(tmp.path()) else {
        return;
    };

    let mut provider = bmi_source(tmp.path()).provider().unwrap();
    assert!(provider.load(999_999, 1.0).is_err());
}
