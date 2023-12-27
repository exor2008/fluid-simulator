#![feature(slice_flatten)]

use backend::fluid::FluidState;
use backend::{fluid::Config, init, pause, resume, stream};
use rocket::fs::FileServer;

#[macro_use]
extern crate rocket;

#[launch]
fn rocket() -> _ {
    rocket::build()
        .mount("/", FileServer::from("static/"))
        .mount("/", routes![stream, init, resume, pause])
        .manage(FluidState::default())
        .manage(Config::default())
}
