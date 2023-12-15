use cudarc::driver::*;
use simulator::{get_device, get_fluid};

const ROWS: usize = 100;
const COLS: usize = 100;
const DT: f32 = 0.01;

fn main() -> Result<(), DriverError> {
    let dev = get_device(0)?;
    let mut fluid = get_fluid(dev.clone(), ROWS, COLS)?;

    let cfg = LaunchConfig {
        grid_dim: (10, 10, 1),
        block_dim: (10, 10, 1),
        shared_mem_bytes: 0,
    };

    fluid.step(dev.clone(), cfg, DT)?;
    let result = fluid.smoke(dev.clone())?;

    for y in 0..ROWS {
        for x in 0..COLS {
            print!("{}", result[y * COLS + x]);
        }
        println!();
    }

    Ok(())
}
