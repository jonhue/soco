pub fn discrete_pos(value: i32) -> f64 {
    if value >= 0 {
        return value as f64;
    } else {
        return 0.;
    }
}
