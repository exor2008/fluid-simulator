use cudarc::{driver::*, nvrtc::compile_ptx};

const ROWS: usize = 100;
const COLS: usize = 100;
const SIZE: usize = ROWS * COLS;

fn main() -> Result<(), DriverError> {
    let dev = CudaDevice::new(0)?;

    const PTX_SRC: &str = include_str!("fluid.cu");

    let ptx = compile_ptx(PTX_SRC).unwrap();

    dev.load_ptx(ptx, "fluid", &["divergence"])?;

    let divergence = dev.get_func("fluid", "divergence").unwrap();

    let mut u_host = vec![0f32; SIZE];
    let v_host = vec![0f32; SIZE];
    let div_host = vec![0f32; SIZE];

    for y in 0..ROWS {
        for x in 0..COLS {
            if x == 5 {
                u_host[y * COLS + x] = 2.0;
            }
        }
    }

    let u_dev = dev.htod_copy(u_host.into())?;
    let v_dev = dev.htod_copy(v_host.into())?;
    let mut div_dev = dev.htod_copy(div_host.into())?;

    let cfg = LaunchConfig {
        grid_dim: (10, 10, 1),
        block_dim: (10, 10, 1),
        shared_mem_bytes: 0,
    };

    unsafe { divergence.launch(cfg, (&mut div_dev, &u_dev, &v_dev, ROWS, COLS)) }?;

    let div_host = dev.sync_reclaim(div_dev)?;

    for y in 0..ROWS {
        for x in 0..COLS {
            print!("{}", div_host[y * COLS + x]);
        }
        println!();
    }

    Ok(())
}
