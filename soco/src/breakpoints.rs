use ordered_float::OrderedFloat;
use std::sync::Arc;

type FiniteBreakpoints = Vec<OrderedFloat<f64>>;
type NextBreakpointFn = Arc<dyn Fn(f64) -> (Option<f64>, Option<f64>)>;

/// Sorted non-continuous or non-smooth points of the probability distribution.
#[derive(Clone)]
pub struct Breakpoints {
    pub bs: FiniteBreakpoints,
    pub next: Option<NextBreakpointFn>,
    // /// Vector of breakpoints, if finite.
    // Finite(FiniteBreakpoints),
    // /// If infinite, a vector of breakpoints, an initial breakpoint and functions which given a breakpoint returns the previous and next breakpoints, respectively.
    // /// The function will be called to obtain the next breakpoint until the piecewise integrals converge to `0`.
    // Infinite(FiniteBreakpoints, NextBreakpointFn, NextBreakpointFn),
}
impl Breakpoints {
    pub fn empty() -> Self {
        Breakpoints {
            bs: vec![],
            next: None,
        }
    }

    pub fn from(bs: Vec<f64>) -> Self {
        Self::empty().add(&bs)
    }

    pub fn grid() -> Self {
        Breakpoints {
            bs: vec![],
            next: Some(Arc::new(|b| {
                (Some((b - 1.).ceil()), Some((b + 1.).floor()))
            })),
        }
    }

    pub fn add(&self, bs: &Vec<f64>) -> Self {
        let mut breakpoints = self.clone();
        for &b in bs {
            if !breakpoints.bs.contains(&OrderedFloat(b)) {
                breakpoints.bs.push(OrderedFloat(b));
            }
        }
        breakpoints.bs.sort_unstable();
        breakpoints
    }
}
