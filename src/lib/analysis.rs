use crate::lib::types::{DiscreteHomProblem, DiscreteSchedule};
use crate::lib::utils::ipos;

pub trait ObjectiveFunction {
    fn objective_function(&self, xs: &DiscreteSchedule) -> f64;
}

impl<'a> ObjectiveFunction for DiscreteHomProblem<'a> {
    fn objective_function(&self, xs: &DiscreteSchedule) -> f64 {
        let mut cost = 0.;
        for t in 1..=self.t_end as usize {
            let prev_x = if t > 1 { xs[t - 2] } else { 0 };
            cost += (self.f)(t as i32, xs[t - 1])
                .expect("f should be total on its domain")
                + self.beta * ipos(xs[t - 1] - prev_x) as f64;
        }
        return cost;
    }
}
