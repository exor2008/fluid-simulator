use cudarc::{driver::*, nvrtc::compile_ptx};
use rand::prelude::*;
use std::time::Instant;

fn main() -> Result<(), DriverError> {
    let dev = CudaDevice::new(0)?;

    const PTX_SRC: &str = include_str!("sin.cu");

    let ptx = compile_ptx(PTX_SRC).unwrap();

    dev.load_ptx(ptx, "sin", &["sin_kernel"])?;

    let f = dev.get_func("sin", "sin_kernel").unwrap();

    let mut a_host = vec![0f32; 360_000_000];
    let mut rng = rand::thread_rng();
    for i in 0..a_host.len() {
        a_host[i] = rng.gen();
    }

    let start_time = Instant::now();

    let mut result = vec![0f32; 360_000_000];
    for i in 0..a_host.len() {
        result[i] = a_host[i].sin();
    }
    let elapsed_time = start_time.elapsed();
    println!("CPU: {:?}", elapsed_time);

    let start_time = Instant::now();
    let a_dev = dev.htod_copy(a_host.into())?;
    let mut b_dev = a_dev.clone();

    let n = 360_000_000;
    let cfg = LaunchConfig::for_num_elems(n);

    unsafe { f.launch(cfg, (&mut b_dev, &a_dev, n as i32)) }?;

    let a_host_2 = dev.sync_reclaim(a_dev)?;
    let b_host = dev.sync_reclaim(b_dev)?;

    let elapsed_time = start_time.elapsed();

    // println!("Found {:?}", b_host);
    println!("GPU {:?}", elapsed_time);
    println!("CPU {:?}", result[1000]);
    println!("GPU {:?}", b_host[1000]);
    // println!("Expected {:?}", a_host.map(f32::sin));
    // assert_eq!(result, b_host);

    Ok(())
}
