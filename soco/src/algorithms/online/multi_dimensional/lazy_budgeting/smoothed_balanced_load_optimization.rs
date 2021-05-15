use ordered_float::OrderedFloat;
use std::sync::Arc;

use crate::algorithms::graph_search::Path;
use crate::algorithms::offline::multi_dimensional::approx_graph_search::{
    approx_graph_search, Options as ApproxOptions,
};
use crate::algorithms::offline::multi_dimensional::optimal_graph_search::optimal_graph_search;
use crate::algorithms::offline::OfflineOptions;
use crate::config::Config;
use crate::cost::CostFn;
use crate::online::Online;
use crate::online::OnlineSolution;
use crate::problem::{
    IntegralSmoothedBalancedLoadOptimization, SmoothedBalancedLoadOptimization,
};
use crate::result::{Error, Result};
use crate::schedule::IntegralSchedule;
use crate::utils::assert;

static DEFAULT_EPSILON: f64 = 0.25;

/// Maps dimension to the number of active instances at some time `t`.
pub type Memory = Vec<i32>;

pub struct Options<'a> {
    /// Whether to use an approximation to find the optimal schedule.
    pub use_approx: Option<&'a ApproxOptions>,
    /// `epsilon > 0`. Defaults to `0.25`.
    pub epsilon: Option<f64>,
}

/// Lazy Budgeting for Smoothed Balanced-Load Optimization
pub fn lb<'a>(
    o: &'a Online<IntegralSmoothedBalancedLoadOptimization>,
    xs: &IntegralSchedule,
    _: &Vec<()>,
    options: &Options,
) -> Result<OnlineSolution<i32, ()>> {
    assert(o.w == 0, Error::UnsupportedPredictionWindow)?;

    let epsilon = options.epsilon.unwrap_or(DEFAULT_EPSILON);
    let t = xs.t_end() + 1;
    let n = determine_sub_time_slots(&o.p, t, epsilon)?;
    let mod_p = modify_problem(&o.p, t, epsilon)?;
    let mut mod_o = Online { w: 0, p: mod_p };
    let init_u = mod_o.p.t_end;

    let (xs, _) = mod_o.offline_stream(alg_b, n, options)?;

    let config = determine_config(&mod_o.p, xs, init_u, n)?;
    Ok(OnlineSolution(config, ()))
}

fn determine_config(
    p: &IntegralSmoothedBalancedLoadOptimization,
    xs: IntegralSchedule,
    init_u: i32,
    n: i32,
) -> Result<Config<i32>> {
    let mut min_u = init_u;
    let mut min_c = hitting_cost(p, init_u, &xs[init_u as usize - 1])?;
    for u in init_u + 1..=init_u + n {
        let c = hitting_cost(p, u, &xs[u as usize - 1])?;
        if c < min_c {
            min_u = u;
            min_c = c;
        }
    }

    Ok(xs[min_u as usize - 1].clone())
}

fn hitting_cost(
    p: &IntegralSmoothedBalancedLoadOptimization,
    t: i32,
    x: &Config<i32>,
) -> Result<f64> {
    let mut result = 0.;
    for k in 0..p.d as usize {
        result += p.hitting_cost[k](t, x[k]).ok_or(Error::CostFnMustBeTotal)?;
    }
    Ok(result)
}

fn modify_problem<'a>(
    p: &'a IntegralSmoothedBalancedLoadOptimization<'a>,
    t: i32,
    epsilon: f64,
) -> Result<IntegralSmoothedBalancedLoadOptimization<'a>> {
    let mut time_slot_mapping: Vec<i32> = vec![]; // maps each time `u` to the corresponding time `t`
    let mut load: Vec<i32> = vec![];
    let mut ns: Vec<i32> = vec![];
    for t in 1..=t - 1 {
        let n = determine_sub_time_slots(p, t, epsilon)?;
        for _ in 0..n {
            time_slot_mapping.push(t);
            load.push(p.load[t as usize - 1]);
        }
        ns.push(n);
    }
    let t_end = ns.iter().sum::<i32>() + 1;

    let hitting_cost = (0..p.d as usize)
        .map(|k| -> CostFn<'a, i32> {
            let time_slot_mapping = time_slot_mapping.clone();
            let ns = ns.clone();
            Arc::new(move |u, x| {
                let t = time_slot_mapping[u as usize - 1];
                Some(p.hitting_cost[k](t, x)? / ns[t as usize - 1] as f64)
            })
        })
        .collect();

    Ok(SmoothedBalancedLoadOptimization {
        d: p.d,
        t_end,
        bounds: p.bounds.clone(),
        switching_cost: p.switching_cost.clone(),
        hitting_cost,
        load,
    })
}

fn determine_sub_time_slots(
    p: &IntegralSmoothedBalancedLoadOptimization,
    t: i32,
    epsilon: f64,
) -> Result<i32> {
    let n = p.d as f64 / epsilon;

    let fractions = (0..p.d as usize)
        .map(|k| -> Result<OrderedFloat<f64>> {
            let l = p.hitting_cost[k](t, 0).ok_or(Error::CostFnMustBeTotal)?;
            Ok(OrderedFloat(l / p.switching_cost[k]))
        })
        .collect::<Result<Vec<OrderedFloat<f64>>>>()?;
    let max_frac = fractions.iter().max().unwrap().into_inner();

    Ok((n * max_frac).ceil() as i32)
}

fn alg_b(
    o: &Online<IntegralSmoothedBalancedLoadOptimization>,
    xs: &IntegralSchedule,
    ms: &Vec<Memory>,
    options: &Options,
) -> Result<OnlineSolution<i32, Memory>> {
    let t = xs.t_end() + 1;
    let opt_x = find_optimal_config(&o.p, options.use_approx)?;
    let mut m = vec![0; o.p.d as usize];
    let mut x = if xs.is_empty() {
        Config::repeat(0, o.p.d)
    } else {
        xs.now().clone()
    };

    for k in 0..o.p.d as usize {
        x[k] -= deactivated_quantity(
            &o.p.hitting_cost[k],
            o.p.switching_cost[k],
            ms,
            t,
            k,
        )?;
        if x[k] < opt_x[k] {
            m[k] = opt_x[k] - x[k];
            x[k] = opt_x[k];
        }
    }

    Ok(OnlineSolution(x, m))
}

fn deactivated_quantity(
    hitting_cost: &CostFn<'_, i32>,
    switching_cost: f64,
    ms: &Vec<Memory>,
    t_now: i32,
    k: usize,
) -> Result<i32> {
    let mut result = 0;
    for t in 1..=t_now - 1 {
        let cum_l =
            cumulative_idle_hitting_cost(hitting_cost, t + 1, t_now - 1)?;
        let l = hitting_cost(t_now, 0).ok_or(Error::CostFnMustBeTotal)?;

        if cum_l <= switching_cost && switching_cost < cum_l + l {
            result += ms[t as usize - 1][k];
        }
    }
    Ok(result)
}

fn cumulative_idle_hitting_cost(
    hitting_cost: &CostFn<'_, i32>,
    from: i32,
    to: i32,
) -> Result<f64> {
    let mut result = 0.;
    for t in from..=to {
        result += hitting_cost(t, 0).ok_or(Error::CostFnMustBeTotal)?;
    }
    Ok(result)
}

fn find_optimal_config(
    p: &IntegralSmoothedBalancedLoadOptimization,
    use_approx: Option<&ApproxOptions>,
) -> Result<Config<i32>> {
    let sco_p = p.to_sco();
    let Path(xs, _) = match use_approx {
        None => {
            optimal_graph_search(&sco_p, &OfflineOptions { inverted: false })?
        }
        Some(options) => approx_graph_search(
            &sco_p,
            options,
            &OfflineOptions { inverted: false },
        )?,
    };
    Ok(xs.now().clone())
}
