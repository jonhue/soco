//! Utilities modeling the cost of server load.

use std::sync::Arc;

/// Function mapping the load of a server to some cost metric.
pub type LoadFn<'a> = Arc<dyn Fn(f64) -> f64 + 'a>;

pub mod bansal {
    //! Loss as described by Bansal et al.

    use std::sync::Arc;

    use crate::cost::data_center::load::LoadFn;

    /// Returns the power consumed by a server as a function of load `l`
    /// according to the formula `l^a + b` where `l^a` models the dynamic power
    /// while `b` models the static/leakage power.
    ///
    /// * `a > 1`
    /// * `b >= 0`
    pub fn energy_loss(a: f64, b: f64) -> LoadFn<'static> {
        Arc::new(move |l| l.powf(a) + b)
    }
}

pub mod lin {
    //! Loss as described by Lin et al.

    use std::sync::Arc;

    use crate::cost::data_center::load::LoadFn;
    use crate::utils::pos;

    /// Returns the power consumed by a server as a function of load `l`
    /// according to the formula `e_0 + e_1 * l` where `e_1 * l` models the dynamic power
    /// while `e_0` models the static/leakage power.
    pub fn energy_loss(e_0: f64, e_1: f64) -> LoadFn<'static> {
        Arc::new(move |l| e_0 + e_1 * l)
    }

    /// Returns the revenue loss given average delay `d` and load `l`
    /// according to the formula `d_1 * l * (d - d_0)^+` where `d_0` is the minimum
    /// delay users can detect and `d_1` is a constant.
    pub fn revenue_loss<'a>(
        d_0: f64,
        d_1: f64,
    ) -> Arc<dyn Fn(f64) -> LoadFn<'a>> {
        Arc::new(move |d| Arc::new(move |l| d_1 * l * pos(d - d_0)))
    }

    /// Returns the average delay of a server modeled by an M/GI/1 Processor Sharing queue
    /// where the service rate of ther server is assumed to be `1`.
    pub fn avg_delay() -> LoadFn<'static> {
        Arc::new(|l| 1. / (1. - l))
    }

    /// Given functions to compute revenue loss (`r`), energy loss (`e`), and average delay (`d`), returns the cumulative loss.
    pub fn loss<'a>(
        r: &'a Arc<dyn Fn(f64) -> LoadFn<'a>>,
        e: &'a LoadFn<'a>,
        d: &'a LoadFn<'a>,
    ) -> LoadFn<'a> {
        Arc::new(move |l| r(d(l))(l) + e(l))
    }
}
