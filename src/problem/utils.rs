pub fn ipos(x: i32) -> i32 {
    if x >= 0 {
        return x;
    } else {
        return 0;
    }
}

pub fn is_2pow(x: i32) -> bool {
    return x != 0 && x & (x - 1) == 0;
}
