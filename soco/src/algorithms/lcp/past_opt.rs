use nlopt::Algorithm;
use nlopt::Nlopt;
use nlopt::ObjFn;
use nlopt::Target;

use crate::problem::{HomProblem, Online};
use crate::PRECISION;

impl<'a, T> Online<HomProblem<'a, T>> {
    /// Convex (continuous) cost optimization.
    pub fn past_opt<'b>(
        &'b self,
        objective_function: impl ObjFn<&'b HomProblem<'a, T>>,
    ) -> Vec<f64> {
        let n = self.p.t_end as usize - 1;
        if n == 0 {
            return vec![0.];
        }

        let mut xs = vec![0.0; n];
        let mut opt = Nlopt::new(
            Algorithm::Bobyqa,
            n,
            objective_function,
            Target::Minimize,
            &self.p,
        );
        opt.set_lower_bound(0.).unwrap();
        opt.set_upper_bound(self.p.m as f64).unwrap();
        opt.set_xtol_rel(PRECISION).unwrap();

        opt.optimize(&mut xs).unwrap();
        xs
    }
}
