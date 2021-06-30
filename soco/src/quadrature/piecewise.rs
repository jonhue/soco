use crate::breakpoints::Breakpoints;
use crate::quadrature::integral;
use crate::result::Result;
use crate::TOLERANCE;
use ordered_float::OrderedFloat;

pub fn piecewise_integral(
    breakpoints: &Breakpoints,
    from: f64,
    to: f64,
    f_: impl Fn(f64) -> f64,
) -> Result<f64> {
    let f = |x| f_(x);

    let b = if from.is_infinite() && to.is_infinite() {
        0.
    } else if from.is_infinite() {
        to
    } else if to.is_infinite() {
        from
    } else {
        (to - from) / 2.
    };
    let i = breakpoints
        .bs
        .iter()
        .position(|x| x.into_inner() > b)
        .unwrap_or(breakpoints.bs.len()) as i32;
    let prev_i = if i > 0
        && breakpoints.bs.get(i as usize - 1) == Some(&OrderedFloat(b))
    {
        i - 2
    } else {
        i - 1
    };
    let l = __piecewise_integral(
        Direction::Left,
        &breakpoints,
        b,
        from,
        f,
        prev_i,
    )?;
    let r = __piecewise_integral(
        Direction::Right,
        &breakpoints,
        b,
        to,
        f,
        i as i32,
    )?;
    Ok(l + r)
}

#[derive(Debug)]
enum Direction {
    Left,
    Right,
}

fn __piecewise_integral(
    dir: Direction,
    breakpoints: &Breakpoints,
    b: f64,
    to: f64,
    f_: impl Fn(f64) -> f64,
    // index of next breakpoint in `breakpoints`
    i: i32,
) -> Result<f64> {
    match dir {
        Direction::Left => {
            if b <= to {
                return Ok(0.);
            }
        }
        Direction::Right => {
            if b >= to {
                return Ok(0.);
            }
        }
    }

    let f = |x| f_(x);
    let mut next_i = i;

    let c = if i >= 0 {
        breakpoints.bs.get(i as usize)
    } else {
        None
    };
    let (x, y) = match &breakpoints.next {
        None => (None, None),
        Some(next) => next(b),
    };
    let next_b = match c {
        None => match dir {
            Direction::Left => x,
            Direction::Right => y,
        },
        Some(c_) => {
            let c = c_.into_inner();
            match dir {
                Direction::Left => match x {
                    None => {
                        next_i -= 1;
                        Some(c)
                    }
                    Some(x) => {
                        if c < x {
                            next_i -= 1;
                            Some(x)
                        } else {
                            Some(c)
                        }
                    }
                },
                Direction::Right => match y {
                    None => {
                        next_i += 1;
                        Some(c)
                    }
                    Some(y) => {
                        if c > y {
                            Some(y)
                        } else {
                            next_i += 1;
                            Some(c)
                        }
                    }
                },
            }
        }
    };

    match dir {
        Direction::Left => {
            if next_b.is_none() || next_b.unwrap() <= to {
                return integral(to, b, f);
            }
        }
        Direction::Right => {
            if next_b.is_none() || next_b.unwrap() >= to {
                return integral(b, to, f);
            }
        }
    };

    let result = match dir {
        Direction::Left => integral(next_b.unwrap(), b, f)?,
        Direction::Right => integral(b, next_b.unwrap(), f)?,
    };
    if result < TOLERANCE {
        return Ok(result);
    }

    Ok(result
        + __piecewise_integral(
            dir,
            breakpoints,
            next_b.unwrap(),
            to,
            f_,
            next_i,
        )?)
}

#[test]
fn piecewise_integral_empty_breakpoints() {
    let result =
        piecewise_integral(&Breakpoints::empty(), 0., f64::INFINITY, |x| {
            std::f64::consts::E.powf(-x)
        })
        .unwrap();
    assert_abs_diff_eq!(result, 1., epsilon = TOLERANCE);
}

#[test]
fn piecewise_integral_default_breakpoints() {
    let result = piecewise_integral(
        &Breakpoints::from(vec![0., 1.]),
        0.,
        f64::INFINITY,
        |x| std::f64::consts::E.powf(-x),
    )
    .unwrap();
    assert_abs_diff_eq!(result, 1., epsilon = TOLERANCE);
}

#[test]
fn piecewise_integral_right() {
    let result =
        piecewise_integral(&Breakpoints::grid(), 0., f64::INFINITY, |x| {
            std::f64::consts::E.powf(-x)
        })
        .unwrap();
    assert_abs_diff_eq!(result, 1., epsilon = TOLERANCE);
}

#[test]
fn piecewise_integral_left() {
    let result =
        piecewise_integral(&Breakpoints::grid(), f64::NEG_INFINITY, 0., |x| {
            std::f64::consts::E.powf(x)
        })
        .unwrap();
    assert_abs_diff_eq!(result, 1., epsilon = TOLERANCE);
}
