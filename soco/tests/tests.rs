#![allow(clippy::module_inception)]

use simple_logger::SimpleLogger;
use std::sync::Once;

static INIT: Once = Once::new();

#[macro_use]
extern crate approx;

#[cfg(test)]
mod algorithms;
#[cfg(test)]
mod convert;
#[cfg(test)]
mod streaming;

mod factories;
mod utils;

fn init() {
    INIT.call_once(|| SimpleLogger::new().init().unwrap());
}
