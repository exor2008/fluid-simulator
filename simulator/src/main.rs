use cudarc::{driver::*, nvrtc::compile_ptx};
use simulator::Fluid;

const ROWS: usize = 100;
const COLS: usize = 100;
const DT: f32 = 0.01;

fn main() -> Result<(), DriverError> {
    let dev = CudaDevice::new(0)?;

    const PTX_SRC: &str = include_str!("fluid.cu");

    let ptx = compile_ptx(PTX_SRC).unwrap();

    dev.load_ptx(
        ptx,
        "fluid",
        &[
            "divergence",
            "pressure",
            "incompress",
            "advect_velocity",
            "advect_smoke",
        ],
    )?;

    let cfg = LaunchConfig {
        grid_dim: (10, 10, 1),
        block_dim: (10, 10, 1),
        shared_mem_bytes: 0,
    };

    let mut fluid = Fluid::new(dev.clone(), ROWS, COLS).unwrap();

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
