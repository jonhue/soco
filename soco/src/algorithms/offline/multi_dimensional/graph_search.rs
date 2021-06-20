use crate::algorithms::graph_search::{Path, Paths};
use crate::algorithms::offline::OfflineOptions;
use crate::config::{Config, IntegralConfig};
use crate::convert::RelaxableProblem;
use crate::convex_optimization::find_minimizer_of_hitting_cost;
use crate::objective::scalar_movement;
use crate::problem::IntegralSimplifiedSmoothedConvexOptimization;
use crate::result::{Error, Result};
use crate::schedule::Schedule;
use num::ToPrimitive;
use ordered_float::OrderedFloat;
use pathfinding::prelude::astar;
use std::collections::HashMap;

/// Vertice in the graph denoting time `t` and the value `x` at time `t`.
/// The boolean flag indicates whether the vertice belongs to the powering up (`true`) or powering down (`false`) phase.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct Vertice {
    t: i32,
    config: IntegralConfig,
    powering_up: bool,
}

/// Graph-Based Integral Algorithm
pub fn graph_search<'a>(
    p: &'a IntegralSimplifiedSmoothedConvexOptimization<'a>,
    configs: &Vec<IntegralConfig>,
    offline_options: &OfflineOptions,
) -> Result<Path> {
    let mut paths: Paths<Vertice> = HashMap::new();
    let initial_vertice = Vertice {
        t: 1,
        config: Config::repeat(0, p.d),
        powering_up: true,
    };
    let final_vertice = Vertice {
        t: p.t_end + 1,
        config: Config::repeat(0, p.d),
        powering_up: true,
    };
    let initial_path = Path(Schedule::empty(), 0.);
    paths.insert(initial_vertice.clone(), initial_path);

    // initial time step (t = 1)
    let from = vec![initial_vertice];
    for config in configs {
        let to = Vertice {
            t: 2,
            config: config.clone(),
            powering_up: true,
        };
        find_shortest_subpath(
            p,
            configs,
            offline_options.inverted,
            &mut paths,
            &from,
            &to,
        )?;
    }

    // intermediate time steps (1 < t < T)
    if p.t_end > 2 {
        for t in 2..p.t_end {
            for config in configs {
                let to = Vertice {
                    t: t + 1,
                    config: config.clone(),
                    powering_up: true,
                };
                let from = configs
                    .iter()
                    .map(|config| Vertice {
                        t,
                        config: config.clone(),
                        powering_up: true,
                    })
                    .collect();
                find_shortest_subpath(
                    p,
                    configs,
                    offline_options.inverted,
                    &mut paths,
                    &from,
                    &to,
                )?;
            }
        }
    }

    // final time step (t = T)
    if p.t_end > 1 {
        let from = configs
            .iter()
            .map(|config| Vertice {
                t: p.t_end,
                config: config.clone(),
                powering_up: true,
            })
            .collect();
        find_shortest_subpath(
            p,
            configs,
            offline_options.inverted,
            &mut paths,
            &from,
            &final_vertice,
        )?;
    }

    Ok(paths
        .get(&final_vertice)
        .ok_or(Error::PathsShouldBeCached)?
        .clone())
}

/// computes the minimum hitting cost of the relaxed problem
fn find_minimum_hitting_cost(
    p: &IntegralSimplifiedSmoothedConvexOptimization<'_>,
    t: i32,
) -> Result<f64> {
    let relaxed_p = p.to_f();
    let bounds = vec![0.; relaxed_p.bounds.len()]
        .into_iter()
        .zip(relaxed_p.bounds.into_iter())
        .collect();
    Ok(find_minimizer_of_hitting_cost(t, &relaxed_p.hitting_cost, &bounds)?.1)
}

fn find_shortest_subpath(
    p: &IntegralSimplifiedSmoothedConvexOptimization<'_>,
    configs: &Vec<IntegralConfig>,
    inverted: bool,
    paths: &mut Paths<Vertice>,
    from: &Vec<Vertice>,
    to: &Vertice,
) -> Result<()> {
    // find minimum hitting cost of relaxed problem to under-approximate distance to the goal
    let minimum_hitting_cost = find_minimum_hitting_cost(p, to.t - 1)?;

    let mut picked_source = &from[0];
    let mut picked_cost = f64::INFINITY;
    let mut picked_vs = vec![];
    for source in from {
        let (vs, cost) = astar(
            source,
            |v| v.successors(p, configs, inverted),
            |v| v.heuristic(to, p, minimum_hitting_cost, inverted),
            |v| *v == *to,
        )
        .ok_or(Error::SubpathShouldBePresent)?;
        let prev_cost = paths.get(source).ok_or(Error::PathsShouldBeCached)?.1;
        let new_cost = prev_cost + cost.into_inner();
        if new_cost < picked_cost {
            picked_source = source;
            picked_cost = new_cost;
            picked_vs = vs;
        };
    }

    // the configuration of the first power-down vertice is the optimal config for this time step
    let x = picked_vs
        .iter()
        .find(|&v| !v.powering_up)
        .unwrap()
        .config
        .clone();
    update_paths(paths, picked_source, to, x, picked_cost)
}

impl Vertice {
    /// Heuristic function under-approximating the cost to the goal assuming the goal is a vertice from the powering up phase.
    ///
    /// This function exploits the observation that
    /// * each dimension must be powered up to match the value in the goal config;
    /// * hitting costs must be paid once; and that
    /// * the only allowed vertice in layer `t + 1` is the goal.
    fn heuristic(
        &self,
        to: &Vertice,
        p: &IntegralSimplifiedSmoothedConvexOptimization<'_>,
        minimum_hitting_cost: f64,
        inverted: bool,
    ) -> OrderedFloat<f64> {
        assert!(to.powering_up, "Only vertices from the powering up phase may be used as goals with this heuristic function.");

        if self.t == to.t {
            if *self == *to {
                return OrderedFloat(0.);
            } else {
                // allow only one vertice in layer `t + 1`
                return OrderedFloat(f64::INFINITY);
            }
        }

        let mut cost = 0.;

        // hitting costs
        if self.powering_up {
            // one could also add the smallest hitting cost of any of the configurations
            // which result from powering-up additional servers (i.e. using the current configuration as a lower bound),
            // but that may be very inefficient.
            cost += minimum_hitting_cost;
        }

        // switching costs
        for k in 0..p.d as usize {
            let delta = ToPrimitive::to_f64(&scalar_movement(
                to.config[k],
                self.config[k],
                inverted,
            ))
            .unwrap();
            cost += p.switching_cost[k] * delta;
        }

        OrderedFloat(cost)
    }

    fn successors(
        &self,
        p: &IntegralSimplifiedSmoothedConvexOptimization<'_>,
        configs: &Vec<IntegralConfig>,
        inverted: bool,
    ) -> Vec<(Vertice, OrderedFloat<f64>)> {
        let mut successors = vec![];
        if self.powering_up {
            // edges paying hitting cost
            successors.push((
                Vertice {
                    t: self.t,
                    config: self.config.clone(),
                    powering_up: false,
                },
                OrderedFloat(
                    (p.hitting_cost)(self.t, self.config.clone()).unwrap(),
                ),
            ));
            // edges for powering up
            for k in 0..p.d as usize {
                let mut config = self.config.clone();
                let vs = collect_dimension_range(configs, k);
                let i = vs.iter().position(|&v| v == config[k]).unwrap();
                if i < vs.len() - 1 {
                    let delta = ToPrimitive::to_f64(&scalar_movement(
                        vs[i + 1],
                        vs[i],
                        inverted,
                    ))
                    .unwrap();
                    let cost = p.switching_cost[k] * delta;
                    config[k] = vs[i + 1];
                    successors.push((
                        Vertice {
                            t: self.t,
                            config,
                            powering_up: true,
                        },
                        OrderedFloat(cost),
                    ));
                }
            }
        } else {
            // edges for powering down
            for k in 0..p.d as usize {
                let mut config = self.config.clone();
                let vs = collect_dimension_range(configs, k);
                let i = vs.iter().position(|&v| v == config[k]).unwrap();
                if i > 0 {
                    config[k] = vs[i - 1];
                    successors.push((
                        Vertice {
                            t: self.t,
                            config,
                            powering_up: false,
                        },
                        OrderedFloat(0.),
                    ));
                }
            }
            // edges for moving to the next time step
            if self.t <= p.t_end {
                successors.push((
                    Vertice {
                        t: self.t + 1,
                        config: self.config.clone(),
                        powering_up: true,
                    },
                    OrderedFloat(0.),
                ));
            }
        }
        successors
    }
}

fn collect_dimension_range(
    configs: &Vec<IntegralConfig>,
    k: usize,
) -> Vec<i32> {
    let mut vs = vec![0; configs.len()];
    for (i, x) in configs.iter().enumerate() {
        vs[i] = x[k];
    }
    vs
}

fn update_paths(
    paths: &mut Paths<Vertice>,
    from: &Vertice,
    to: &Vertice,
    x: IntegralConfig,
    c: f64,
) -> Result<()> {
    let prev_xs = &paths.get(from).ok_or(Error::PathsShouldBeCached)?.0;
    let xs = prev_xs.extend(x);

    paths.insert(to.clone(), Path(xs, c));
    Ok(())
}
