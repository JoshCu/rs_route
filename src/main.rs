use anyhow::Result;

use rs_route::cli::get_args;
use rs_route::run_routing;

fn main() -> Result<()> {
    let config = get_args()?;
    run_routing(config, false)
}
