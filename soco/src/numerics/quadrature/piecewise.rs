use crate::breakpoints::Breakpoints;
use crate::numerics::quadrature::integral;
use crate::numerics::PRECISION;
use crate::result::Result;
use ordered_float::OrderedFloat;

/// Number of consecutive integrations below precision before the piecewise integration is stopped.
static CONVERGENCE_THRESHOLD: i32 = 10;

/// Computes the piecewise integral of `f_` over the interval `[from, to]` with respect to some breakpoints.
/// Note that this function terminates when either the entire interval was integrated OR piecewise integrals converge to `0`.
pub fn piecewise_integral(
    breakpoints: &Breakpoints,
    from: f64,
    to: f64,
    f_: impl Fn(f64) -> f64,
) -> Result<f64> {
    let f = |x| f_(x);

    // determine initial breakpoint
    let init = if from.is_infinite() && to.is_infinite() {
        0.
    } else if from.is_infinite() {
        to
    } else if to.is_infinite() {
        from
    } else {
        (to - from) / 2.
    };

    // find initial indices of next and previous breakpoints in `breakpoints.bs`
    let i = breakpoints
        .bs
        .iter()
        .position(|x| x.into_inner() > init)
        .unwrap_or(breakpoints.bs.len()) as i32;
    let prev_i = if i > 0
        && breakpoints.bs.get(i as usize - 1) == Some(&OrderedFloat(init))
    {
        i - 2
    } else {
        i - 1
    };

    // compute piecewise integrals in both directions
    let l = __piecewise_integral(
        Direction::Left,
        &breakpoints,
        init,
        from,
        f,
        prev_i,
        0,
    )?;
    let r = __piecewise_integral(
        Direction::Right,
        &breakpoints,
        init,
        to,
        f,
        i,
        0,
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
    // number of consecutive integrations below tolerance
    n: i32,
) -> Result<f64> {
    // check if limit of integration is reached
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
    let mut next_n = n;

    // obtain the next fixed breakpoint
    let fixed_b = if i >= 0 {
        breakpoints.bs.get(i as usize)
    } else {
        None
    };

    // obtain the next dynamic breakpoints (which depend on the direction)
    let (left_b, right_b) = match &breakpoints.next {
        None => (None, None),
        Some(next) => next(b),
    };

    // choose the next breakpoint; and update `i` if fixed breakpoint is chosen
    let next_b = match fixed_b {
        None => match dir {
            Direction::Left => left_b,
            Direction::Right => right_b,
        },
        Some(fixed_b_) => {
            let fixed_b = fixed_b_.into_inner();
            match dir {
                Direction::Left => match left_b {
                    None => {
                        next_i -= 1;
                        Some(fixed_b)
                    }
                    Some(left_b) => {
                        if fixed_b < left_b {
                            Some(left_b)
                        } else {
                            next_i -= 1;
                            Some(fixed_b)
                        }
                    }
                },
                Direction::Right => match right_b {
                    None => {
                        next_i += 1;
                        Some(fixed_b)
                    }
                    Some(right_b) => {
                        if fixed_b > right_b {
                            Some(right_b)
                        } else {
                            next_i += 1;
                            Some(fixed_b)
                        }
                    }
                },
            }
        }
    };

    // if there is no new breakpoint or the new breakpoint is beyond the limit of integration, then compute the remaining integral and return
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
    // otherwise, compute the integral over the new interval
    let result = match dir {
        Direction::Left => integral(next_b.unwrap(), b, f)?,
        Direction::Right => integral(b, next_b.unwrap(), f)?,
    };

    // check if convergence threshold is reached
    if result < PRECISION {
        next_n += 1;
        if next_n >= CONVERGENCE_THRESHOLD {
            return Ok(result);
        }
    }

    // compute next interval
    Ok(result
        + __piecewise_integral(
            dir,
            breakpoints,
            next_b.unwrap(),
            to,
            f_,
            next_i,
            next_n,
        )?)
}

#[test]
fn piecewise_integral_empty_breakpoints() {
    let result =
        piecewise_integral(&Breakpoints::empty(), 0., f64::INFINITY, |x| {
            std::f64::consts::E.powf(-x)
        })
        .unwrap();
    assert_abs_diff_eq!(result, 1., epsilon = PRECISION);
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
    assert_abs_diff_eq!(result, 1., epsilon = PRECISION);
}

#[test]
fn piecewise_integral_right() {
    let result =
        piecewise_integral(&Breakpoints::grid(1.), 0., f64::INFINITY, |x| {
            std::f64::consts::E.powf(-x)
        })
        .unwrap();
    assert_abs_diff_eq!(result, 1., epsilon = PRECISION);
}

#[test]
fn piecewise_integral_left() {
    let result = piecewise_integral(
        &Breakpoints::grid(1.),
        f64::NEG_INFINITY,
        0.,
        |x| std::f64::consts::E.powf(x),
    )
    .unwrap();
    assert_abs_diff_eq!(result, 1., epsilon = PRECISION);
}
