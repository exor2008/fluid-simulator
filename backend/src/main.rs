#![feature(slice_flatten)]

use backend::fluid::FluidState;
use backend::{fluid::Config, gravity, init, pause, reset, resume, stream, switch};
use rocket::fs::FileServer;

#[macro_use]
extern crate rocket;

#[launch]
fn rocket() -> _ {
    rocket::build()
        .mount("/", FileServer::from("static/"))
        .mount(
            "/",
            routes![stream, init, resume, pause, reset, switch, gravity],
        )
        .manage(FluidState::default())
        .manage(Config::default())
}
