use rocket::form::FromForm;
use rocket::fs::TempFile;
use rocket::tokio::sync::Mutex;
use simulator::FluidData;
use simulator::{Fluid, LaunchConfig};
use std::mem::{self, transmute};

// #[derive(Debug, FromForm, Default)]
pub struct Size {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

#[derive(FromForm)]
pub struct Input<'a> {
    pub x: usize,
    pub y: usize,
    pub z: usize,
    pub gltf: TempFile<'a>,
}

pub struct Config {
    pub data: Mutex<FluidData>,
    pub stream_on: Mutex<bool>,
    pub size: Mutex<Size>,
    pub launch: Mutex<LaunchConfig>,
}

impl Default for Config {
    fn default() -> Self {
        let cfg = LaunchConfig {
            grid_dim: (3, 3, 3),
            block_dim: (3, 3, 3),
            shared_mem_bytes: 0,
        };

        let size = Size { x: 3, y: 3, z: 3 };

        Config {
            data: Default::default(),
            stream_on: Mutex::new(true),
            size: Mutex::new(size),
            launch: Mutex::new(cfg),
        }
    }
}

#[derive(Default)]
pub struct FluidState {
    pub fluid: Mutex<Fluid>,
}

impl FluidState {
    pub fn vec_to_bytes<T>(vec: Vec<T>) -> Vec<u8> {
        return unsafe {
            let float_ptr: *const T = vec.as_ptr();
            let byte_ptr: *const u8 = transmute(float_ptr);
            let byte_slice: &[u8] =
                std::slice::from_raw_parts(byte_ptr, vec.len() * mem::size_of::<T>());
            byte_slice.to_vec()
        };
    }

    pub fn get_launch_config() -> LaunchConfig {
        LaunchConfig {
            grid_dim: (18, 10, 12),
            block_dim: (10, 10, 10),
            shared_mem_bytes: 0,
        }
    }
}
