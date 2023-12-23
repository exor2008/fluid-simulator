use cudarc::driver::*;
use simulator::{get_device, get_fluid};

const X_SIZE: usize = 180;
const Y_SIZE: usize = 100;
const Z_SIZE: usize = 120;
const DT: f32 = 0.01;

fn main() -> Result<(), DriverError> {
    let dev = get_device(0)?;
    let mut fluid = get_fluid(dev.clone(), X_SIZE, Y_SIZE, Z_SIZE)?;

    let cfg = LaunchConfig {
        grid_dim: (18, 10, 12),
        block_dim: (10, 10, 10),
        shared_mem_bytes: 0,
    };

    fluid.step(dev.clone(), cfg, DT)?;
    let result = fluid.smoke(dev.clone())?;

    // fluid.step(dev.clone(), cfg, DT)?;
    // let result = fluid.smoke(dev.clone())?;

    // for r in result {
    //     if r != 0.0 {
    //         print!("{} ", r);
    //     }
    // }

    // for y in 0..ROWS {
    //     for x in 0..COLS {
    //         print!("{}", result[y * COLS + x]);
    //     }
    //     println!();
    // }

    Ok(())
}
