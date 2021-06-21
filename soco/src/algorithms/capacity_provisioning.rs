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
use crate::convex_optimization::find_minimizer;
use crate::objective::Objective;
use crate::problem::{
    FractionalSimplifiedSmoothedConvexOptimization,
    IntegralSimplifiedSmoothedConvexOptimization,
};
use crate::result::{Error, Result};
use crate::utils::{assert, is_pow_of_2};

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

        let objective = |xs: &[f64]| -> f64 {
            self.objective_function(
                &xs.iter().map(|&x| Config::single(x)).collect(),
            )
            .unwrap()
        };
        self.find_bound(objective, t, t_start)
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

        let objective = |xs: &[f64]| -> f64 {
            self.inverted_objective_function(
                &xs.iter().map(|&x| Config::single(x)).collect(),
            )
            .unwrap()
        };
        self.find_bound(objective, t, t_start)
    }
}

impl FractionalSimplifiedSmoothedConvexOptimization<'_> {
    fn find_bound(
        &self,
        objective: impl Fn(&[f64]) -> f64,
        t: i32,
        t_start: i32,
    ) -> Result<f64> {
        assert(self.d == 1, Error::UnsupportedProblemDimension)?;
        assert(t <= self.t_end, Error::LcpBoundComputationExceedsDomain)?;

        if t <= 0 {
            return Ok(0.);
        }

        let bounds =
            vec![(0., self.bounds[0]); (self.t_end - t_start) as usize];
        let (xs, _) = find_minimizer(objective, &bounds)?;
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
        let Path { xs, .. } = match use_approx {
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
