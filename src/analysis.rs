use crate::lib::types::{DiscreteHomProblem, DiscreteSchedule};
use crate::lib::utils::ipos;

pub fn discrete_objective_function(
    p: &DiscreteHomProblem,
    xs: &DiscreteSchedule,
) -> f64 {
    let mut cost = 0.;
    for t in 0..p.t_end as usize {
        let prev_x = if t > 0 { xs[t - 1] } else { 0 };
        cost += (p.f)(t as i32, xs[t])
            .expect("f should be total on its domain")
            + p.beta * ipos(xs[t] - prev_x);
    }
    return cost;
}
