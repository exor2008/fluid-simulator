use rocket::response::stream::ByteStream;
use rocket::serde::{json::Json, Serialize};
use rocket::tokio::select;
use rocket::tokio::time::{self, Duration};
use rocket::Shutdown;
use simulator::LaunchConfig;
use simulator::{get_device, get_fluid};
use std::mem::{self, transmute};
use std::vec;

#[macro_use]
extern crate rocket;
const X: usize = 140;
const Y: usize = 80;
const Z: usize = 40;
const SIZE: usize = X * Y * Z;

#[derive(Serialize)]
#[serde(crate = "rocket::serde")]
pub struct Size {
    x: usize,
    y: usize,
    z: usize,
}

#[get("/size")]
pub fn get_size() -> Json<Size> {
    let size = Size { x: X, y: Y, z: Z };
    Json(size)
}

#[get("/fluid/stream")]
pub fn stream(mut shutdown: Shutdown) -> ByteStream![Vec<u8>] {
    let mut interval = time::interval(Duration::from_millis(1));

    let dev = get_device(0).unwrap();

    let mut fluid = get_fluid(dev.clone(), X, Y, Z).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (20, 20, 20),
        block_dim: (10, 10, 10),
        shared_mem_bytes: 0,
    };

    ByteStream! {
        loop {
            fluid.step(dev.clone(), cfg, 0.01).unwrap();
            let result = fluid.smoke(dev.clone()).unwrap();
            select! {
                _ = interval.tick() => {
                    let bytes: Vec<u8> = vec_to_bytes(result);
                    yield bytes;
                    interval.tick().await;
                }
                _ = &mut shutdown => {
                    yield vec![0u8; SIZE];
                    break;
                }
            }
        }
    }
}

fn vec_to_bytes<T>(vec: Vec<T>) -> Vec<u8> {
    return unsafe {
        let float_ptr: *const T = vec.as_ptr();
        let byte_ptr: *const u8 = transmute(float_ptr);
        let byte_slice: &[u8] =
            std::slice::from_raw_parts(byte_ptr, vec.len() * mem::size_of::<T>());
        byte_slice.to_vec()
    };
}
