use bacon_sci::differentiate::second_derivative;
use bacon_sci::integrate::integrate;
use nlopt::Algorithm;
use nlopt::Nlopt;
use nlopt::Target;
use std::f64::{INFINITY, NEG_INFINITY};
use std::sync::Arc;

use crate::online::{Online, OnlineSolution};
use crate::problem::ContinuousHomProblem;
use crate::result::Result;
use crate::schedule::ContinuousSchedule;
use crate::PRECISION;

/// Probability distribution over the number of servers.
pub type Memory<'a> = Arc<dyn Fn(f64) -> f64 + 'a>;

static STEP_SIZE: f64 = 1e-16;

impl<'a> Online<ContinuousHomProblem<'a>> {
    /// Deterministic Online Algorithm
    pub fn det(
        &'a self,
        xs: &ContinuousSchedule,
        ps: &Vec<Memory<'a>>,
    ) -> Result<OnlineSolution<f64, Memory<'a>>> {
        let t = xs.len() as i32 + 1;
        let prev_p = if ps.is_empty() {
            Arc::new(|j| if j == 0. { 1. } else { 0. })
        } else {
            ps[ps.len() - 1].clone()
        };

        let x_m = self.find_minimizer(t)?;
        let x_r = self.find_right_bound(t, &prev_p, x_m)?;
        let x_l = self.find_left_bound(t, &prev_p, x_m)?;

        let p: Arc<dyn Fn(f64) -> f64> = Arc::new(move |j| {
            if j >= x_l && j <= x_r {
                prev_p(j)
                    + second_derivative(
                        |j: f64| (self.p.f)(t, j).unwrap(),
                        j,
                        STEP_SIZE,
                    ) / 2.
            } else {
                0.
            }
        });

        let x = expected_value(&p, x_l, x_r)?;
        Ok((x, p))
    }

    /// Determines minimizer of `f` with a convex optimization.
    fn find_minimizer(&self, t: i32) -> Result<f64> {
        let objective_function =
            |xs: &[f64], _: Option<&mut [f64]>, _: &mut ()| -> f64 {
                (self.p.f)(t, xs[0]).unwrap()
            };
        let mut xs = [0.0];

        let mut opt = Nlopt::new(
            Algorithm::Bobyqa,
            1,
            objective_function,
            Target::Minimize,
            (),
        );
        opt.set_lower_bound(0.)?;
        opt.set_upper_bound(self.p.m as f64)?;
        opt.set_xtol_rel(PRECISION)?;

        opt.optimize(&mut xs)?;
        Ok(xs[0])
    }

    /// Determines `x_r` with a convex optimization.
    fn find_right_bound(
        &self,
        t: i32,
        prev_p: &Memory<'a>,
        x_m: f64,
    ) -> Result<f64> {
        let objective_function =
            |xs: &[f64], _: Option<&mut [f64]>, _: &mut ()| -> f64 { xs[0] };
        let mut xs = [x_m];

        let mut opt = Nlopt::new(
            Algorithm::Bobyqa,
            1,
            objective_function,
            Target::Maximize,
            (),
        );
        opt.set_lower_bound(0.)?;
        opt.set_upper_bound(self.p.m as f64)?;
        opt.set_xtol_rel(PRECISION)?;

        opt.add_equality_constraint(
            |xs: &[f64], _: Option<&mut [f64]>, _: &mut ()| -> f64 {
                let l = integrate(
                    x_m,
                    xs[0],
                    |j: f64| {
                        second_derivative(
                            |j: f64| (self.p.f)(t, j).unwrap(),
                            j,
                            STEP_SIZE,
                        )
                    },
                    PRECISION,
                )
                .unwrap();
                let r =
                    integrate(xs[0], INFINITY, |j: f64| prev_p(j), PRECISION)
                        .unwrap();
                l / 2. - r
            },
            (),
            PRECISION,
        )?;

        opt.optimize(&mut xs)?;
        Ok(xs[0])
    }

    /// Determines `x_l` with a convex optimization.
    fn find_left_bound(
        &self,
        t: i32,
        prev_p: &Memory<'a>,
        x_m: f64,
    ) -> Result<f64> {
        let objective_function =
            |xs: &[f64], _: Option<&mut [f64]>, _: &mut ()| -> f64 { xs[0] };
        let mut xs = [x_m];

        let mut opt = Nlopt::new(
            Algorithm::Bobyqa,
            1,
            objective_function,
            Target::Minimize,
            (),
        );
        opt.set_lower_bound(0.)?;
        opt.set_upper_bound(self.p.m as f64)?;
        opt.set_xtol_rel(PRECISION)?;

        opt.add_equality_constraint(
            |xs: &[f64], _: Option<&mut [f64]>, _: &mut ()| -> f64 {
                let l = integrate(
                    xs[0],
                    x_m,
                    |j: f64| {
                        second_derivative(
                            |j: f64| (self.p.f)(t, j).unwrap(),
                            j,
                            STEP_SIZE,
                        )
                    },
                    PRECISION,
                )
                .unwrap();
                let r = integrate(
                    NEG_INFINITY,
                    xs[0],
                    |j: f64| prev_p(j),
                    PRECISION,
                )
                .unwrap();
                l / 2. - r
            },
            (),
            PRECISION,
        )?;

        opt.optimize(&mut xs)?;
        Ok(xs[0])
    }
}

fn expected_value(p: &Memory, a: f64, b: f64) -> Result<f64> {
    Ok(integrate(a, b, |j: f64| j * p(j), PRECISION)?)
}
