pub fn log(n: i32) -> u32 {
    return (n as f64).log(2.).floor() as i32;
}

pub fn discrete_pos(value: i32) -> f64 {
    if value >= 0 {
        return value as f64;
    } else {
        return 0.;
    }
}
