#![feature(slice_flatten)]

use rocket::fs::FileServer;
use rocket::response::stream::ByteStream;
use rocket::tokio::select;
use rocket::tokio::time::{self, Duration};
use rocket::Shutdown;
use std::mem::{self, transmute};
use std::vec;

#[macro_use]
extern crate rocket;

#[get("/fluid/stream")]
fn stream(mut shutdown: Shutdown) -> ByteStream![Vec<u8>] {
    const SIZE: usize = 256 * 256 * 256;
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

#[launch]
fn rocket() -> _ {
    rocket::build()
        .mount("/", FileServer::from("static/"))
        .mount("/", routes![stream])
}
