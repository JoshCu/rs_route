# rs-route

A Rust implementation of Muskingum-Cunge channel routing for [NextGen](https://github.com/NOAA-OWP/ngen) hydrological modeling frameworks, inspired by [t-route](https://github.com/NOAA-OWP/t-route).

## Features

- **Multiple routing kernels**: Pure Rust (scalar and SIMD), modernized Fortran (t-route), legacy Fortran (t-route), and C implementations
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
```

### CLI Options

```
Usage: rs-route [OPTIONS] <ROUTE_DIR>

Arguments:
  <ROUTE_DIR>  Path to NextGen run directory

Options:
  -i, --internal-timestep-seconds <N>  Internal routing timestep in seconds [default: 300]
  -k, --kernel <KERNEL>                Routing kernel to use [default: t-route-modernized]
                                       [possible values: route-rs, route-rs-simd,
                                       t-route-modernized, t-route-legacy,
                                       c-muskingum-cunge]
  -n, --num-threads <N>                Worker threads [default: number of CPUs]
      --fast-converge                  Start the depth solver from a tighter bracket
```

### Performance

The Muskingum-Cunge kernel is latency-bound rather than throughput-bound: one
secant iteration is ~250 cycles for ~40 flops, because it is a single chain of
dependent divides and square roots. Two options trade a little accuracy for
throughput:

| Option | Speedup | Cost |
| --- | --- | --- |
| `-k route-rs-simd` | 3.3x (AVX2), 4.0x (AVX-512) | 99.9% of calls within 1e-4 of the scalar kernel |
| `--fast-converge` | a further 1.45x | total flow shifted ~0.07% |

`route-rs-simd` routes 16 reaches per timestep in lockstep, so it helps most
where the network is wide; when the wavefront narrows to fewer than 16 ready
reaches the spare lanes idle. It also holds 16 nodes' inputs and results per
worker rather than one, so peak memory is higher.

Note that per-reach hydrographs are not reproducible bit-for-bit across kernels,
and this is a property of the model rather than of any one kernel: the solver
stops at a 1% relative tolerance on depth, which then feeds back into the next
timestep. Perturbing the scalar kernel's starting depth by a nanometre diverges
further than switching to the SIMD kernel does. Compare aggregate flows, not
individual timesteps.

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

## License

[MIT](LICENSE)
