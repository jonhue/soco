use crate::algorithms::offline::uni_dimensional::optimal_graph_search::{
    optimal_graph_search, Options as OptimalGraphSearchOptions,
};
use crate::algorithms::offline::{
    OfflineAlgorithm, OfflineOptions, OfflineResult,
};
use crate::config::Config;
use crate::convert::ResettableProblem;
use crate::numerics::convex_optimization::find_minimizer;
use crate::objective::Objective;
use crate::problem::{
    FractionalSimplifiedSmoothedConvexOptimization,
    IntegralSimplifiedSmoothedConvexOptimization,
};
use crate::result::{Failure, Result};
use crate::utils::assert;
use noisy_float::prelude::*;

pub trait Bounded<T> {
    /// Computes the number of servers at time `t` starting from `t_start` with initial condition `x_start` simulating up to time `t_end` resulting in the lowest possible cost.
    fn find_lower_bound(
        &self,
        alpha: f64,
        t: i32,
        t_start: i32,
        x_start: T,
    ) -> Result<T>;

    /// Computes the number of servers at time `t` starting from `t_start` with initial condition `x_start` simulating up to time `t_end` resulting in the highest possible cost.
    fn find_upper_bound(
        &self,
        alpha: f64,
        t: i32,
        t_start: i32,
        x_start: T,
    ) -> Result<T>;
}

impl Bounded<f64> for FractionalSimplifiedSmoothedConvexOptimization<'_> {
    fn find_lower_bound(
        &self,
        alpha: f64,
        t: i32,
        t_start: i32,
        x_start: f64,
    ) -> Result<f64> {
        self.find_bound(alpha, false, t, t_start, x_start)
    }

    fn find_upper_bound(
        &self,
        alpha: f64,
        t: i32,
        t_start: i32,
        x_start: f64,
    ) -> Result<f64> {
        self.find_bound(alpha, true, t, t_start, x_start)
    }
}

impl FractionalSimplifiedSmoothedConvexOptimization<'_> {
    fn find_bound(
        &self,
        alpha: f64,
        inverted: bool,
        t: i32,
        t_start: i32,
        x_start: f64,
    ) -> Result<f64> {
        assert!(t <= self.t_end);
        assert(self.d == 1, Failure::UnsupportedProblemDimension(self.d))?;

        if t <= 0 {
            return Ok(0.);
        }

        let p = self.reset(t_start);
        let objective = |xs: &[f64]| -> N64 {
            n64(p
                ._objective_function_with_default(
                    &xs.iter().map(|&x| Config::single(x)).collect(),
                    &Config::single(x_start),
                    alpha,
                    inverted,
                )
                .unwrap())
        };
        let bounds = vec![(0., p.bounds[0]); p.t_end as usize];
        let (xs, _) = find_minimizer(objective, &bounds)?;
        Ok(xs[(t - t_start) as usize - 1])
    }
}

impl Bounded<i32> for IntegralSimplifiedSmoothedConvexOptimization<'_> {
    fn find_lower_bound(
        &self,
        alpha: f64,
        t: i32,
        t_start: i32,
        x_start: i32,
    ) -> Result<i32> {
        self.find_bound(alpha, false, t, t_start, x_start)
    }

    fn find_upper_bound(
        &self,
        alpha: f64,
        t: i32,
        t_start: i32,
        x_start: i32,
    ) -> Result<i32> {
        self.find_bound(alpha, true, t, t_start, x_start)
    }
}

impl IntegralSimplifiedSmoothedConvexOptimization<'_> {
    fn find_bound(
        &self,
        alpha: f64,
        inverted: bool,
        t: i32,
        t_start: i32,
        x_start: i32,
    ) -> Result<i32> {
        assert!(t <= self.t_end);
        assert(self.d == 1, Failure::UnsupportedProblemDimension(self.d))?;

        if t <= 0 {
            return Ok(0);
        }

        let p = self.reset(t_start);
        let result = optimal_graph_search.solve(
            p,
            OptimalGraphSearchOptions { x_start },
            OfflineOptions::new(inverted, alpha, None),
        )?;
        let xs = result.xs();

        Ok(xs[(t - t_start) as usize - 1][0])
    }
}
