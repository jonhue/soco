use crate::algorithms::graph_search::Path;
use crate::algorithms::offline::multi_dimensional::approx_graph_search::{
    approx_graph_search, Options as ApproxGraphSearch,
};
use crate::algorithms::offline::multi_dimensional::optimal_graph_search::optimal_graph_search;
use crate::algorithms::offline::OfflineAlgorithm;
use crate::algorithms::online::{IntegralStep, Step};
use crate::config::{Config, IntegralConfig};
use crate::cost::{CallableCostFn, CostFn};
use crate::problem::{
    IntegralSmoothedBalancedLoadOptimization, Online,
    SmoothedBalancedLoadOptimization,
};
use crate::result::{Failure, Result};
use crate::schedule::{IntegralSchedule, Schedule};
use crate::utils::assert;
use ordered_float::OrderedFloat;

/// Schedule and memory of internally used algorithm.
pub type Memory = (IntegralSchedule, Vec<AlgBMemory>);

/// Maps dimension to the number of active instances for some sub time slot `u`.
type AlgBMemory = Vec<i32>;

#[derive(Clone)]
pub struct Options {
    /// Whether to use an approximation to find the optimal schedule.
    pub use_approx: Option<ApproxGraphSearch>,
    /// `epsilon > 0`. Defaults to `0.25`.
    pub epsilon: f64,
}
impl Default for Options {
    fn default() -> Self {
        Options {
            use_approx: None,
            epsilon: 0.25,
        }
    }
}

/// Lazy Budgeting for Smoothed Balanced-Load Optimization
pub fn lb(
    o: &Online<IntegralSmoothedBalancedLoadOptimization>,
    xs: &mut IntegralSchedule,
    ms: &mut Vec<Memory>,
    options: Options,
) -> Result<IntegralStep<Memory>> {
    assert(o.w == 0, Failure::UnsupportedPredictionWindow(o.w))?;

    let t = xs.t_end() + 1;
    let n = determine_sub_time_slots(&o.p, t, options.epsilon)?;
    let mut mod_o = Online {
        w: 0,
        p: modify_problem(&o.p, t, options.epsilon)?,
    };
    let init_u = mod_o.p.t_end;

    if ms.is_empty() {
        ms.push((Schedule::empty(), vec![]));
    }
    let (mut algb_xs, mut algb_ms) = ms[0].clone();
    mod_o.offline_stream_from(alg_b, n, options, &mut algb_xs, &mut algb_ms)?;
    ms[0] = (algb_xs, algb_ms);

    let config = determine_config(&mod_o.p, xs, init_u, n);
    Ok(Step(config, None))
}

fn determine_config(
    p: &IntegralSmoothedBalancedLoadOptimization,
    xs: &IntegralSchedule,
    init_u: i32,
    n: i32,
) -> IntegralConfig {
    let mut min_u = init_u;
    let mut min_c = p.hit_cost(init_u, xs[init_u as usize - 1].clone());
    for u in init_u + 1..=init_u + n {
        let c = p.hit_cost(u, xs[u as usize - 1].clone());
        if c < min_c {
            min_u = u;
            min_c = c;
        }
    }

    xs[min_u as usize - 1].clone()
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
            CostFn::new(move |u, x| {
                let t = time_slot_mapping[u as usize - 1];
                p.hitting_cost[k].call(t, x, p.bounds[k])
                    / ns[t as usize - 1] as f64
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
            let l = p.hitting_cost[k].call(t, 0, p.bounds[k]);
            Ok(OrderedFloat(l / p.switching_cost[k]))
        })
        .collect::<Result<Vec<OrderedFloat<f64>>>>()?;
    let max_frac = fractions.iter().max().unwrap().into_inner();

    Ok((n * max_frac).ceil() as i32)
}

fn alg_b(
    o: Online<IntegralSmoothedBalancedLoadOptimization>,
    t: i32,
    xs: &IntegralSchedule,
    prev_m: AlgBMemory,
    options: Options,
) -> Result<IntegralStep<AlgBMemory>> {
    let opt_x = find_optimal_config(&o.p, options.use_approx)?;
    let mut m = vec![0; o.p.d as usize];
    let mut x = if xs.is_empty() {
        Config::repeat(0, o.p.d)
    } else {
        xs.now()
    };

    for k in 0..o.p.d as usize {
        x[k] -= deactivated_quantity(
            o.p.bounds[k],
            &o.p.hitting_cost[k],
            o.p.switching_cost[k],
            &prev_m,
            t,
            k,
        );
        if x[k] < opt_x[k] {
            m[k] = opt_x[k] - x[k];
            x[k] = opt_x[k];
        }
    }

    Ok(Step(x, Some(m)))
}

fn deactivated_quantity(
    bound: i32,
    hitting_cost: &CostFn<'_, i32>,
    switching_cost: f64,
    m: &AlgBMemory,
    t_now: i32,
    k: usize,
) -> i32 {
    let mut result = 0;
    for t in 1..=t_now - 1 {
        let cum_l =
            cumulative_idle_hitting_cost(bound, hitting_cost, t + 1, t_now - 1);
        let l = hitting_cost.call(t_now, 0, bound);

        if cum_l <= switching_cost && switching_cost < cum_l + l {
            result += m[k];
        }
    }
    result
}

fn cumulative_idle_hitting_cost(
    bound: i32,
    hitting_cost: &CostFn<'_, i32>,
    from: i32,
    to: i32,
) -> f64 {
    let mut result = 0.;
    for t in from..=to {
        result += hitting_cost.call(t, 0, bound);
    }
    result
}

fn find_optimal_config(
    p: &IntegralSmoothedBalancedLoadOptimization,
    use_approx: Option<ApproxGraphSearch>,
) -> Result<IntegralConfig> {
    let ssco_p = p.to_ssco();
    let Path { xs, .. } = match use_approx {
        None => optimal_graph_search.solve(ssco_p, (), false)?,
        Some(options) => approx_graph_search.solve(ssco_p, options, false)?,
    };
    Ok(xs.now())
}
