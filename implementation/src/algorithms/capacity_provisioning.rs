use crate::algorithms::offline::uni_dimensional::optimal_graph_search::{
    optimal_graph_search, Options as OptimalGraphSearchOptions,
};
use crate::algorithms::offline::{
    OfflineAlgorithm, OfflineOptions, OfflineResult,
};
use crate::config::Config;
use crate::convert::ResettableProblem;
use crate::model::{ModelOutputFailure, ModelOutputSuccess};
use crate::numerics::convex_optimization::{find_minimizer, WrappedObjective};
use crate::problem::{
    FractionalSimplifiedSmoothedConvexOptimization,
    IntegralSimplifiedSmoothedConvexOptimization, Problem,
};
use crate::result::{Failure, Result};
use crate::utils::assert;

pub trait Bounded<T>
where
    T: std::fmt::Debug + Clone,
    Self: std::fmt::Debug,
{
    fn find_lower_bound(
        &self,
        w: i32,
        t: i32,
        t_start: i32,
        x_start: T,
    ) -> Result<T> {
        self.find_alpha_unfair_lower_bound(1., w, t, t_start, x_start)
    }

    /// Computes the number of servers at time $t$ starting from $t_start$ with initial condition $x_start$ simulating up to time $t_end$ resulting in the lowest possible cost.
    fn find_alpha_unfair_lower_bound(
        &self,
        alpha: f64,
        w: i32,
        t: i32,
        t_start: i32,
        x_start: T,
    ) -> Result<T>;

    fn find_upper_bound(
        &self,
        w: i32,
        t: i32,
        t_start: i32,
        x_start: T,
    ) -> Result<T> {
        self.find_alpha_unfair_upper_bound(1., w, t, t_start, x_start)
    }

    /// Computes the number of servers at time $t$ starting from $t_start$ with initial condition $x_start$ simulating up to time $t_end$ resulting in the highest possible cost.
    fn find_alpha_unfair_upper_bound(
        &self,
        alpha: f64,
        w: i32,
        t: i32,
        t_start: i32,
        x_start: T,
    ) -> Result<T>;
}

impl<C, D> Bounded<f64>
    for FractionalSimplifiedSmoothedConvexOptimization<'_, C, D>
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    fn find_alpha_unfair_lower_bound(
        &self,
        alpha: f64,
        w: i32,
        t: i32,
        t_start: i32,
        x_start: f64,
    ) -> Result<f64> {
        self.find_bound(alpha, false, w, t, t_start, x_start)
    }

    fn find_alpha_unfair_upper_bound(
        &self,
        alpha: f64,
        w: i32,
        t: i32,
        t_start: i32,
        x_start: f64,
    ) -> Result<f64> {
        self.find_bound(alpha, true, w, t, t_start, x_start)
    }
}

#[derive(Clone)]
struct ObjectiveData<'a, C, D> {
    p: FractionalSimplifiedSmoothedConvexOptimization<'a, C, D>,
    alpha: f64,
    inverted: bool,
    x_start: f64,
}

impl<C, D> FractionalSimplifiedSmoothedConvexOptimization<'_, C, D>
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    fn find_bound(
        &self,
        alpha: f64,
        inverted: bool,
        w: i32,
        t: i32,
        t_start: i32,
        x_start: f64,
    ) -> Result<f64> {
        assert!(t <= self.t_end + w);
        assert(self.d == 1, Failure::UnsupportedProblemDimension(self.d))?;

        if t <= 0 {
            return Ok(0.);
        }

        let mut p = self.reset(t_start);
        p.t_end += w;
        let objective = WrappedObjective::new(
            ObjectiveData {
                p: p.clone(),
                alpha,
                inverted,
                x_start,
            },
            |xs, data| {
                data.p
                    ._objective_function_with_default(
                        &xs.iter().map(|&x| Config::single(x)).collect(),
                        &Config::single(data.x_start),
                        data.alpha,
                        data.inverted,
                    )
                    .unwrap()
                    .cost
            },
        );
        let bounds = vec![(0., p.bounds[0]); p.t_end as usize];
        let (xs, _) = find_minimizer(objective, bounds);
        Ok(xs[(t - t_start) as usize - 1])
    }
}

impl<C, D> Bounded<i32>
    for IntegralSimplifiedSmoothedConvexOptimization<'_, C, D>
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    fn find_alpha_unfair_lower_bound(
        &self,
        alpha: f64,
        w: i32,
        t: i32,
        t_start: i32,
        x_start: i32,
    ) -> Result<i32> {
        self.find_bound(alpha, false, w, t, t_start, x_start)
    }

    fn find_alpha_unfair_upper_bound(
        &self,
        alpha: f64,
        w: i32,
        t: i32,
        t_start: i32,
        x_start: i32,
    ) -> Result<i32> {
        self.find_bound(alpha, true, w, t, t_start, x_start)
    }
}

impl<C, D> IntegralSimplifiedSmoothedConvexOptimization<'_, C, D>
where
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    fn find_bound(
        &self,
        alpha: f64,
        inverted: bool,
        w: i32,
        t: i32,
        t_start: i32,
        x_start: i32,
    ) -> Result<i32> {
        assert!(t <= self.t_end + w);
        assert(self.d == 1, Failure::UnsupportedProblemDimension(self.d))?;

        if t <= 0 {
            return Ok(0);
        }

        let mut p = self.reset(t_start);
        p.t_end += w;
        let result = optimal_graph_search.solve(
            p,
            OptimalGraphSearchOptions { x_start },
            OfflineOptions::new(inverted, alpha, None),
        )?;
        let xs = result.xs();

        Ok(xs[(t - t_start) as usize - 1][0])
    }
}