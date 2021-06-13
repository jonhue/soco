use crate::algorithms::graph_search::Path;
use crate::algorithms::offline::multi_dimensional::approx_graph_search::{
    approx_graph_search, Options as ApproxOptions,
};
use crate::algorithms::offline::uni_dimensional::optimal_graph_search::{
    make_pow_of_2, optimal_graph_search, Options as OptOptions,
};
use crate::algorithms::offline::OfflineOptions;
use crate::config::Config;
use crate::convert::ResettableProblem;
use crate::objective::Objective;
use crate::problem::{
    FractionalSimplifiedSmoothedConvexOptimization,
    IntegralSimplifiedSmoothedConvexOptimization,
};
use crate::result::{Error, Result};
use crate::utils::{assert, is_pow_of_2};
use crate::PRECISION;
use nlopt::{Algorithm, Nlopt, ObjFn, Target};

pub trait Bounded<T> {
    /// Computes the number of servers at time `t` starting from `t_start` simulating up to time `t_end` resulting in the lowest possible cost.
    ///
    /// `use_approx` may only be set when computing integral bounds.
    fn find_lower_bound(
        &self,
        t: i32,
        t_start: i32,
        use_approx: Option<&'_ ApproxOptions>,
    ) -> Result<T>;

    /// Computes the number of servers at time `t` starting from `t_start` simulating up to time `t_end` resulting in the highest possible cost.
    ///
    /// `use_approx` may only be set when computing integral bounds.
    fn find_upper_bound(
        &self,
        t: i32,
        t_start: i32,
        use_approx: Option<&'_ ApproxOptions>,
    ) -> Result<T>;
}

impl Bounded<f64> for FractionalSimplifiedSmoothedConvexOptimization<'_> {
    fn find_lower_bound(
        &self,
        t: i32,
        t_start: i32,
        use_approx: Option<&'_ ApproxOptions>,
    ) -> Result<f64> {
        assert(
            use_approx.is_none(),
            Error::UnsupportedArgument(
                "Approximation can only be used for integral bounds."
                    .to_string(),
            ),
        )?;

        let objective_function =
            |xs: &[f64],
             _: Option<&mut [f64]>,
             p: &mut &FractionalSimplifiedSmoothedConvexOptimization<'_>|
             -> f64 {
                p.objective_function(
                    &xs.iter().map(|&x| Config::single(x)).collect(),
                )
                .unwrap()
            };
        self.find_bound(objective_function, t, t_start)
    }

    fn find_upper_bound(
        &self,
        t: i32,
        t_start: i32,
        use_approx: Option<&'_ ApproxOptions>,
    ) -> Result<f64> {
        assert(
            use_approx.is_none(),
            Error::UnsupportedArgument(
                "Approximation can only be used for integral bounds."
                    .to_string(),
            ),
        )?;

        let objective_function =
            |xs: &[f64],
             _: Option<&mut [f64]>,
             p: &mut &FractionalSimplifiedSmoothedConvexOptimization<'_>|
             -> f64 {
                p.inverted_objective_function(
                    &xs.iter().map(|&x| Config::single(x)).collect(),
                )
                .unwrap()
            };
        self.find_bound(objective_function, t, t_start)
    }
}

impl FractionalSimplifiedSmoothedConvexOptimization<'_> {
    fn find_bound<'a>(
        &'a self,
        objective_function: impl ObjFn<
            &'a FractionalSimplifiedSmoothedConvexOptimization<'a>,
        >,
        t: i32,
        t_start: i32,
    ) -> Result<f64> {
        assert(self.d == 1, Error::UnsupportedProblemDimension)?;
        assert(t <= self.t_end, Error::LcpBoundComputationExceedsDomain)?;

        if t <= 0 {
            return Ok(0.);
        }

        let n = (self.t_end - t_start) as usize;
        let mut xs = vec![0.; n];
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

impl Bounded<i32> for IntegralSimplifiedSmoothedConvexOptimization<'_> {
    fn find_lower_bound(
        &self,
        t: i32,
        t_start: i32,
        use_approx: Option<&'_ ApproxOptions>,
    ) -> Result<i32> {
        self.find_bound(t, t_start, false, use_approx)
    }

    fn find_upper_bound(
        &self,
        t: i32,
        t_start: i32,
        use_approx: Option<&'_ ApproxOptions>,
    ) -> Result<i32> {
        self.find_bound(t, t_start, true, use_approx)
    }
}

impl IntegralSimplifiedSmoothedConvexOptimization<'_> {
    fn find_bound(
        &self,
        t: i32,
        t_start: i32,
        inverted: bool,
        use_approx: Option<&'_ ApproxOptions>,
    ) -> Result<i32> {
        assert(self.d == 1, Error::UnsupportedProblemDimension)?;
        assert(t <= self.t_end, Error::LcpBoundComputationExceedsDomain)?;

        if t <= 0 {
            return Ok(0);
        }

        let mut p = self.reset(t_start);
        let Path(xs, _) = match use_approx {
            None => {
                if !is_pow_of_2(self.bounds[0]) {
                    p = make_pow_of_2(self)?;
                }
                optimal_graph_search(&p, &OptOptions { inverted })?
            }
            Some(options) => {
                approx_graph_search(&p, options, &OfflineOptions { inverted })?
            }
        };

        Ok(xs[(t - t_start) as usize - 1][0])
    }
}
