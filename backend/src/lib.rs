use rocket::response::stream::ByteStream;
use rocket::serde::{json::Json, Serialize};
use rocket::tokio::select;
use rocket::tokio::time::{self, Duration};
use rocket::Shutdown;
use std::mem::{self, transmute};
use std::vec;

#[macro_use]
extern crate rocket;
const X: usize = 256;
const Y: usize = 64;
const Z: usize = 64;
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
    let mut interval = time::interval(Duration::from_micros(1));

    ByteStream! {
        loop {
            let random_values: Vec<f32> = (0..SIZE).map(|_| rand::random()).collect();
            select! {
                _ = interval.tick() => {
                    let bytes: Vec<u8> = unsafe {
                        let float_ptr: *const f32 = random_values.as_ptr();
                        let byte_ptr: *const u8 = transmute(float_ptr);
                        let byte_slice: &[u8] =
                            std::slice::from_raw_parts(byte_ptr, random_values.len() * mem::size_of::<f32>());
                        byte_slice.to_vec()
                    };
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
