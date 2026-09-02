# rs-route

A Rust implementation of Muskingum-Cunge channel routing for [NextGen](https://github.com/NOAA-OWP/ngen) hydrological modeling frameworks, inspired by [t-route](https://github.com/NOAA-OWP/t-route).

## Features

- **Multiple routing kernels**: Pure Rust, modernized Fortran (t-route), legacy Fortran (t-route), and C implementations
- **Parallel wave-front routing**: Processes independent network branches concurrently using topological ordering
- **NetCDF and CSV output**: Produces t-route-compatible NetCDF output files
- **GeoPackage input**: Reads network topology and channel parameters from NextGen hydrofabric GeoPackage databases

## Project Structure

```
src/
├── main.rs            # Entry point and simulation orchestration
├── cli.rs             # Command-line interface (clap)
├── config.rs          # Configuration structures
├── network.rs         # Network topology and database operations
├── state.rs           # Node status tracking
├── routing.rs         # Parallel wave-front routing engine
├── io/
│   ├── csv.rs         # CSV reading/writing
│   ├── netcdf.rs      # NetCDF output
│   └── results.rs     # Simulation results storage
└── kernel/
    └── muskingum/
        ├── mod.rs           # Kernel dispatcher
        ├── rs_route/
        │   └── mc_kernel.rs # Pure Rust Muskingum-Cunge implementation
        ├── t_route.rs       # Fortran t-route FFI bindings
        ├── t-route/         # Fortran source (modernized + legacy)
        ├── c_mc.rs          # C Muskingum-Cunge FFI bindings
        └── c_mc/            # C source
```

## Dependencies

### System libraries

- `libhdf5-dev`
- `libnetcdf-dev`
- `libsqlite3-dev`
- `gfortran`
- `gcc`

On Ubuntu/Debian:
```bash
sudo apt install -y libhdf5-dev libnetcdf-dev libsqlite3-dev gfortran gcc
```

### Rust

Requires Rust 1.85+ (edition 2024). Install via [rustup](https://rustup.rs/).

## Building and Running

```bash
# Build in release mode
cargo build --release

# Run routing on a NextGen output directory
cargo run --release -- <path/to/ngen-run-directory>

# Build with in-process catchment models (see below)
cargo build --release --features bmi
```

### CLI Options

```
Usage: rs-route [OPTIONS] <ROUTE_DIR>

Arguments:
  <ROUTE_DIR>  Path to NextGen run directory

Options:
  -i, --internal-timestep-seconds <N>  Internal routing timestep in seconds [default: 300]
  -k, --kernel <KERNEL>                Routing kernel to use [default: t-route-modernized]
                                       [possible values: route-rs, t-route-modernized,
                                       t-route-legacy, c-muskingum-cunge]
```

### Running the catchment models directly

With the `bmi` feature, rs-route can run the catchment models itself through
[bmi-driver](https://github.com/JoshCu/bmi-driver) rather than reading the CSV files they would
have written:

```bash
rs-route <route_dir> --bmi-dir <bmi_data_dir>
```

`<bmi_data_dir>` is a bmi-driver data directory -- the one holding `config/`, `forcings/` and the
model libraries. Nothing is written to disk between the models and the router, and the units of
the routed variable come from the models rather than being assumed. `--bmi-config` points at a
realization config other than `<bmi_data_dir>/config/realization.json`, and `--flow-variable`
selects a model output other than `Q_OUT` (it applies to the CSV path too).

Each worker thread runs its own copy of the models. That is safe for models keeping their state
in the instance the registration function allocates, and unsafe for anything using Fortran `save`
variables or C file-scope statics -- those instances will quietly share state. Use `-n 1`, or the
CSV path, if you are unsure which kind you have.

The `bmi-python` feature adds Python BMI models, and needs Python development headers at build
time.

### Expected directory structure

The target directory should follow the NextGen output convention:
```
<route_dir>/
├── config/
│   └── *.gpkg          # Hydrofabric GeoPackage
└── outputs/
    ├── ngen/            # NextGen model output CSVs (cat-*.csv)
    └── troute/          # Routing output directory
```

## Testing

```bash
cargo test
```

A small test dataset is included in `tests/one_cat/` for integration testing.

```bash
cargo test --features bmi
```

also runs the in-process model tests, which compile a synthetic BMI model
(`tests/fixtures/bucket_bmi.c`) during the test run and check that routing through a CSV file and
routing the models directly agree. They need a C compiler and nothing else.

## License

[MIT](LICENSE)
