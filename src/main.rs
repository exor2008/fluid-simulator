use std::mem::swap;

use cudarc::{driver::*, nvrtc::compile_ptx};

const ROWS: usize = 100;
const COLS: usize = 100;
const SIZE: usize = ROWS * COLS;
const ITERATIONS: usize = 40;
const H: f32 = 1.0 / 100.0;
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

    let mut u_host = vec![0f32; SIZE];
    let v_host = vec![0f32; SIZE];
    let new_u_host = vec![0f32; SIZE];
    let new_v_host = vec![0f32; SIZE];
    let mut smoke_host = vec![0f32; SIZE];
    let new_smoke_host = vec![0f32; SIZE];
    let div_host = vec![0f32; SIZE];
    let pressure_a_host = vec![0f32; SIZE];
    let pressure_b_host = vec![0f32; SIZE];

    for y in 0..ROWS {
        for x in 0..COLS {
            if x == 5 {
                u_host[y * COLS + x] = 2.0;
                smoke_host[y * COLS + x] = 1.0;
            }
        }
    }

    let mut u_dev = dev.htod_copy(u_host.into())?;
    let mut v_dev = dev.htod_copy(v_host.into())?;
    let mut new_u_dev = dev.htod_copy(new_u_host.into())?;
    let mut new_v_dev = dev.htod_copy(new_v_host.into())?;
    let mut smoke_dev = dev.htod_copy(smoke_host.into())?;
    let mut new_smoke_dev = dev.htod_copy(new_smoke_host.into())?;
    let mut div_dev = dev.htod_copy(div_host.into())?;
    let mut pressure_a_dev = dev.htod_copy(pressure_a_host.into())?;
    let mut pressure_b_dev = dev.htod_copy(pressure_b_host.into())?;

    let cfg = LaunchConfig {
        grid_dim: (10, 10, 1),
        block_dim: (10, 10, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        // Divergence
        let divergence = dev.get_func("fluid", "divergence").unwrap();
        divergence.launch(cfg, (&mut div_dev, &u_dev, &v_dev, ROWS, COLS))?;
        dev.synchronize()?;

        // Pressure Jacobi
        for _ in 0..ITERATIONS {
            let pressure = dev.get_func("fluid", "pressure").unwrap();

            pressure.launch(
                cfg,
                (
                    &mut pressure_a_dev,
                    &mut pressure_b_dev,
                    &div_dev,
                    ROWS,
                    COLS,
                ),
            )?;
            dev.synchronize()?;

            swap(&mut pressure_a_dev, &mut pressure_b_dev);
        }

        // Solve Incompressibility
        let incompress = dev.get_func("fluid", "incompress").unwrap();
        incompress.launch(cfg, (&mut u_dev, &mut v_dev, &pressure_a_dev, ROWS, COLS))?;
        dev.synchronize()?;

        // Advect velocity
        let advect_velocity = dev.get_func("fluid", "advect_velocity").unwrap();
        advect_velocity.launch(
            cfg,
            (
                &u_dev,
                &v_dev,
                &mut new_u_dev,
                &mut new_v_dev,
                &smoke_dev,
                DT,
                H,
                ROWS,
                COLS,
            ),
        )?;
        swap(&mut u_dev, &mut new_u_dev);
        swap(&mut v_dev, &mut new_v_dev);
        dev.synchronize()?;

        // Advect smoke
        let advect_smoke = dev.get_func("fluid", "advect_smoke").unwrap();
        advect_smoke.launch(
            cfg,
            (
                &smoke_dev,
                &mut new_smoke_dev,
                &u_dev,
                &v_dev,
                DT,
                H,
                ROWS,
                COLS,
            ),
        )?;
        swap(&mut smoke_dev, &mut new_smoke_dev);
    };

    let result = dev.sync_reclaim(smoke_dev)?;

    for y in 0..ROWS {
        for x in 0..COLS {
            print!("{}", result[y * COLS + x]);
        }
        println!();
    }

    Ok(())
}
