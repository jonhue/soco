use std::cmp::{max, min};

pub fn ipos(x: i32) -> i32 {
    if x >= 0 {
        x
    } else {
        0
    }
}

pub fn project(x: i32, a: i32, b: i32) -> i32 {
    max(a, min(b, x))
}

pub fn is_2pow(x: i32) -> bool {
    x != 0 && x & (x - 1) == 0
}
