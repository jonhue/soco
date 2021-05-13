use nlopt::Algorithm;
use nlopt::Nlopt;
use nlopt::ObjFn;
use nlopt::Target;

use crate::algorithms::graph_search::Path;
use crate::algorithms::offline::uni_dimensional::optimal_graph_search::{
    make_pow_of_2, optimal_graph_search,
};
use crate::objective::Objective;
use crate::problem::{
    ContinuousSmoothedConvexOptimization, DiscreteSmoothedConvexOptimization,
};
use crate::result::{Error, Result};
use crate::utils::{assert, is_pow_of_2};
use crate::PRECISION;

pub trait Bounded<T> {
    /// Computes the number of servers at time `t` simulating up to time `t_end` resulting in the lowest possible cost.
    fn find_lower_bound(&self, t: i32, t_start: i32) -> Result<T>;

    /// Computes the number of servers at time `t` simulating up to time `t_end` resulting in the highest possible cost.
    fn find_upper_bound(&self, t: i32, t_start: i32) -> Result<T>;
}

impl Bounded<f64> for ContinuousSmoothedConvexOptimization<'_> {
    fn find_lower_bound(&self, t: i32, t_start: i32) -> Result<f64> {
        let objective_function =
            |xs: &[f64],
             _: Option<&mut [f64]>,
             p: &mut &ContinuousSmoothedConvexOptimization<'_>|
             -> f64 {
                p.objective_function(&xs.iter().map(|&x| vec![x]).collect())
                    .unwrap()
            };

        self.find_bound(objective_function, t, t_start)
    }

    fn find_upper_bound(&self, t: i32, t_start: i32) -> Result<f64> {
        let objective_function =
            |xs: &[f64],
             _: Option<&mut [f64]>,
             p: &mut &ContinuousSmoothedConvexOptimization<'_>|
             -> f64 {
                p.inverted_objective_function(
                    &xs.iter().map(|&x| vec![x]).collect(),
                )
                .unwrap()
            };

        self.find_bound(objective_function, t, t_start)
    }
}

impl ContinuousSmoothedConvexOptimization<'_> {
    fn find_bound<'a>(
        &'a self,
        objective_function: impl ObjFn<&'a ContinuousSmoothedConvexOptimization<'a>>,
        t: i32,
        t_start: i32,
    ) -> Result<f64> {
        assert(self.d == 1, Error::UnsupportedProblemDimension)?;
        assert(t <= self.t_end, Error::LcpBoundComputationExceedsDomain)?;

        if t <= 0 {
            return Ok(0.);
        }

        let n = (self.t_end - t_start) as usize;
        let mut xs = vec![0.0; n];
        let mut opt = Nlopt::new(
            Algorithm::Bobyqa,
            n,
            objective_function,
            Target::Minimize,
            self,
        );
        opt.set_lower_bound(0.)?;
        opt.set_upper_bound(self.bounds[0])?;
        opt.set_xtol_rel(PRECISION)?;

        opt.optimize(&mut xs)?;
        Ok(xs[(t - t_start) as usize - 1])
    }
}

impl Bounded<i32> for DiscreteSmoothedConvexOptimization<'_> {
    fn find_lower_bound(&self, t: i32, t_start: i32) -> Result<i32> {
        let alg = |p: &DiscreteSmoothedConvexOptimization<'_>| {
            optimal_graph_search(p, false)
        };
        self.find_bound(alg, t, t_start)
    }

    fn find_upper_bound(&self, t: i32, t_start: i32) -> Result<i32> {
        let alg = |p: &DiscreteSmoothedConvexOptimization<'_>| {
            optimal_graph_search(p, true)
        };
        self.find_bound(alg, t, t_start)
    }
}

impl DiscreteSmoothedConvexOptimization<'_> {
    fn find_bound(
        &self,
        alg: impl Fn(&'_ DiscreteSmoothedConvexOptimization<'_>) -> Result<Path>,
        t: i32,
        t_start: i32,
    ) -> Result<i32> {
        assert(self.d == 1, Error::UnsupportedProblemDimension)?;
        assert(t <= self.t_end, Error::LcpBoundComputationExceedsDomain)?;

        if t <= 0 {
            return Ok(0);
        }

        let tmp;
        let p = if is_pow_of_2(self.bounds[0]) {
            self
        } else {
            tmp = make_pow_of_2(self)?;
            &tmp
        };
        let reset_p = p.reset(t_start);

        let Path(xs, _) = alg(&reset_p)?;
        Ok(xs[(t - t_start) as usize - 1][0])
    }
}
