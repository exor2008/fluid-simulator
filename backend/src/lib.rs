use crate::fluid::{Config, Size};
use fluid::{FluidState, Input};
use rocket::form::Form;
use rocket::http::Status;
use rocket::response::stream::ByteStream;
use rocket::tokio::io;
use rocket::tokio::select;
use rocket::tokio::time::{self, Duration};
use rocket::Shutdown;
use rocket::State;
use simulator::FluidData;
use simulator::{get_device, gltf, Fluid};
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;
use std::vec;

#[macro_use]
extern crate rocket;
const X: usize = 180;
const Y: usize = 100;
const Z: usize = 120;
const SIZE: usize = X * Y * Z;

pub mod fluid;

#[post("/init", data = "<input>")]
pub async fn init<'a>(
    input: Form<Input<'a>>,
    fluid_state: &State<FluidState>,
    config: &State<Config>,
) {
    let size = Size {
        x: input.x,
        y: input.y,
        z: input.z,
    };

    let mut stream = input.gltf.open().await.unwrap();
    let mut bin: Vec<u8> = vec![];
    io::copy(&mut stream, &mut bin).await.unwrap();

    let (doc, buffers, _) = gltf::import_slice(bin).unwrap();

    let dev = get_device(0).unwrap();
    Fluid::init_dev(dev.clone()).unwrap();

    let mut fluid = fluid_state.fluid.lock().await;
    *fluid = Fluid::from_gltf(Arc::clone(&dev), size.x, size.y, size.z, doc, buffers).unwrap();

    let mut launch = config.launch.lock().await;
    *launch = FluidState::get_launch_config();
}

#[get("/fluid/stream")]
pub async fn stream<'a>(
    mut shutdown: Shutdown,
    fluid_state: &'a State<FluidState>,
    config: &'a State<Config>,
) -> ByteStream![Vec<u8> + 'a] {
    let mut interval = time::interval(Duration::from_millis(1));
    let dev = get_device(0).unwrap();
    Fluid::init_dev(dev.clone()).unwrap();

    ByteStream! {
        loop {
            if *config.stream_on.lock().await {
                let cfg = config.launch.lock().await;
                let to_draw = config.data.lock().await;

                let gravity = config.gravity.load(Relaxed);
                fluid_state
                    .fluid
                    .lock()
                    .await
                    .step(Arc::clone(&dev), *cfg, 0.01, gravity)
                    .unwrap();

                let result = fluid_state
                    .fluid
                    .lock()
                    .await
                    .get_to_draw(Arc::clone(&dev), *to_draw, *cfg)
                    .unwrap();

                select! {
                    _ = interval.tick() => {
                        let bytes: Vec<u8> = FluidState::vec_to_bytes(result.clone());
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
}

#[post("/pause")]
pub async fn pause<'a>(config: &State<Config>) {
    *config.stream_on.lock().await = false;
}

#[post("/resume")]
pub async fn resume<'a>(config: &State<Config>) {
    *config.stream_on.lock().await = true;
}

#[post("/reset")]
pub async fn reset<'a>(fluid_state: &State<FluidState>) {
    let dev = get_device(0).unwrap();
    Fluid::init_dev(dev.clone()).unwrap();

    let mut fluid = fluid_state.fluid.lock().await;
    fluid.reset(dev).unwrap();
}

#[post("/switch/<data>")]
pub async fn switch<'a>(data: &str, config: &State<Config>) -> Status {
    *config.data.lock().await = match data {
        "smoke" => FluidData::Smoke,
        "pressure" => FluidData::Pressure,
        "block" => FluidData::Block,
        "speed" => FluidData::Speed,
        "speed_smoke" => FluidData::SpeedSmoke,
        &_ => return Status::NotFound,
    };

    Status::Ok
}

#[post("/gravity/<value>")]
pub fn gravity<'a>(value: f32, config: &State<Config>) {
    config.gravity.store(value, Relaxed);
}
