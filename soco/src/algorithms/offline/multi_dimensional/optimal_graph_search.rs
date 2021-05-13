use ordered_float::OrderedFloat;
use pathfinding::prelude::astar;
use std::collections::HashMap;

use crate::algorithms::graph_search::{Path, Paths};
use crate::config::Config;
use crate::problem::IntegralSmoothedConvexOptimization;
use crate::result::{Error, Result};
use crate::schedule::Schedule;
use crate::utils::pos;

/// Vertice in the graph denoting time `t` and the value `x` at time `t`.
/// The boolean flag indicates whether the vertice belongs to the powering up (`true`) or powering down (`false`) phase.
#[derive(Clone, Eq, Hash, PartialEq)]
pub struct Vertice {
    t: i32,
    config: Config<i32>,
    powering_up: bool,
}

/// Graph-Based Optimal Integral Algorithm
pub fn optimal_graph_search<'a>(
    p: &'a IntegralSmoothedConvexOptimization<'a>,
) -> Result<Path> {
    let mut paths: Paths<Vertice> = HashMap::new();
    let initial_vertice = Vertice {
        t: 1,
        config: Config::repeat(0, p.d),
        powering_up: true,
    };
    let initial_path = Path(Schedule::empty(), 0.);
    paths.insert(initial_vertice, initial_path);

    let configs = build_configs(p);
    for t in 1..p.t_end {
        for x in &configs {
            let to = Vertice {
                t: t + 1,
                config: x.clone(),
                powering_up: true,
            };
            for y in &configs {
                let from = Vertice {
                    t,
                    config: y.clone(),
                    powering_up: true,
                };
                find_shortest_subpath(p, &mut paths, &from, &to)?;
            }
        }
    }

    let final_vertice = Vertice {
        t: p.t_end + 1,
        config: Config::repeat(0, p.d),
        powering_up: false,
    };
    for config in configs {
        let from = Vertice {
            t: p.t_end,
            config,
            powering_up: true,
        };
        find_shortest_subpath(p, &mut paths, &from, &final_vertice)?;
    }

    Ok(paths
        .get(&final_vertice)
        .ok_or(Error::PathsShouldBeCached)?
        .clone())
}

/// Computes all configurations for some time `t`.
fn build_configs(
    p: &IntegralSmoothedConvexOptimization<'_>,
) -> Vec<Config<i32>> {
    let mut configs: Vec<Vec<i32>> = vec![vec![]];
    for k in 1..=p.d {
        for j in 0..=p.bounds[k as usize] {
            for v in configs.iter_mut() {
                v.push(j);
            }
        }
    }
    configs
        .iter()
        .map(|config| Config::new(config.clone()))
        .collect()
}

fn find_shortest_subpath(
    p: &IntegralSmoothedConvexOptimization<'_>,
    paths: &mut Paths<Vertice>,
    from: &Vertice,
    to: &Vertice,
) -> Result<()> {
    let (vs, c) = astar(
        from,
        |v| v.successors(p),
        |v| v.heuristic(to, p),
        |v| *v == *to,
    )
    .ok_or(Error::SubpathShouldBePresent)?;
    // the configuration of the first power-down vertice is the optimal config for this time step
    let x = vs.iter().find(|&v| !v.powering_up).unwrap().config.clone();
    update_paths(paths, from, to, x, c.into_inner())
}

impl Vertice {
    /// Heuristic function under-approximating the cost to the goal assuming the goal is a powering up vertice.
    ///
    /// This function exploits the observation that
    /// * each dimension must be powered up to match the value in the goal config;
    /// * the only allowed vertice in layer `t + 1` is the goal; and that
    /// * at the powering down vertice, the value in each dimension must be greater equals the goal config.
    fn heuristic(
        &self,
        to: &Vertice,
        p: &IntegralSmoothedConvexOptimization<'_>,
    ) -> OrderedFloat<f64> {
        assert!(to.powering_up, "Only powering up vertices may be used as goals with this heuristic function.");

        let mut cost = 0.;

        // allow only one vertice in layer `t + 1`
        if self.t == to.t && *self != *to {
            return OrderedFloat(f64::INFINITY);
        }

        // when already powering down, the value in each dimension must be greater equals the goal config
        if !self.powering_up {
            for k in 0..p.d as usize {
                if self.config[k] < to.config[k] {
                    return OrderedFloat(f64::INFINITY);
                }
            }
        }

        // switching costs
        for k in 0..p.d as usize {
            cost +=
                p.switching_cost[k] * pos(to.config[k] - self.config[k]) as f64;
        }

        // one could also add the smallest hitting cost of any of the configurations,
        // but depending on the computational complexity of the convex cost function that may be very inefficient.

        OrderedFloat(cost)
    }

    fn successors(
        &self,
        p: &IntegralSmoothedConvexOptimization<'_>,
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
                    (p.hitting_cost)(self.t, self.config.to_vec()).unwrap(),
                ),
            ));
            // edges for powering up
            for k in 0..p.d as usize {
                let mut x = self.config.to_vec();
                if x[k] < p.bounds[k] {
                    x[k] += 1;
                    successors.push((
                        Vertice {
                            t: self.t,
                            config: Config::new(x),
                            powering_up: true,
                        },
                        OrderedFloat(p.switching_cost[k]),
                    ));
                }
            }
        } else {
            // edges for powering down
            for k in 0..p.d as usize {
                let mut x = self.config.to_vec();
                if x[k] > 0 {
                    x[k] -= 1;
                    successors.push((
                        Vertice {
                            t: self.t,
                            config: Config::new(x),
                            powering_up: false,
                        },
                        OrderedFloat(0.),
                    ));
                }
            }
            // edges for moving to the next time step
            if self.t < p.t_end {
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

fn update_paths(
    paths: &mut Paths<Vertice>,
    from: &Vertice,
    to: &Vertice,
    x: Config<i32>,
    c: f64,
) -> Result<()> {
    let prev_xs = &paths.get(from).ok_or(Error::PathsShouldBeCached)?.0;
    let xs = prev_xs.extend(x);

    paths.insert(to.clone(), Path(xs, c));
    Ok(())
}
