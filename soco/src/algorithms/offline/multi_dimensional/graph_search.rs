use log::debug;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::algorithms::offline::graph_search::{
    read_cache, Cache, CachedPath, Path, Paths,
};
use crate::algorithms::offline::multi_dimensional::Values;
use crate::algorithms::offline::OfflineOptions;
use crate::config::{Config, IntegralConfig};
use crate::objective::scalar_movement;
use crate::problem::IntegralSimplifiedSmoothedConvexOptimization;
use crate::result::{Failure, Result};
use crate::schedule::IntegralSchedule;
use crate::utils::assert;
use serde_derive::{Deserialize, Serialize};
use std::collections::HashMap;

/// Tracks the value as well as the index in which the value appears in the respective dimension.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
struct InternalConfig {
    config: IntegralConfig,
    indices: Config<usize>,
}

/// Vertice in the graph denoting time `t` and the value `x` at time `t`.
/// The boolean flag indicates whether the vertice belongs to the powering up (`true`) or powering down (`false`) phase.
/// The algorithm only keeps the most recent two layers in memory.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct Vertice {
    config: InternalConfig,
    powering_up: bool,
}

/// Edge consisting of source and weight.
#[derive(Clone, Debug)]
struct Edge {
    from: Vertice,
    cost: f64,
}

/// Graph-Based Integral Algorithm
pub fn graph_search(
    p: IntegralSimplifiedSmoothedConvexOptimization<'_>,
    values: Values,
    cache: Option<Cache<Vertice>>,
    OfflineOptions { inverted, alpha, l }: OfflineOptions,
) -> Result<CachedPath<Cache<Vertice>>> {
    assert(l.is_none(), Failure::UnsupportedLConstrainedMovement)?;

    let (t_init, mut paths) = read_cache(cache, || (1, HashMap::new()));

    debug!("from time slot `{}` to time slot `{}`", t_init, p.t_end);

    for t in t_init..=p.t_end {
        handle_layer(&p, alpha, inverted, t, true, &values, &mut paths, None)?;
        debug!("handled first layer at time `{}`", t);
        handle_layer(&p, alpha, inverted, t, false, &values, &mut paths, None)?;
        debug!("handled second layer at time `{}`", t);
    }

    Ok(CachedPath {
        path: paths[&Vertice {
            config: build_base_config(p.d, &p.bounds, &values, true),
            powering_up: false,
        }]
            .clone(),
        cache: Cache { t: p.t_end, paths },
    })
}

struct HandleLayerState {
    k: i32,
    configs: Vec<InternalConfig>,
}

#[allow(clippy::too_many_arguments)]
fn handle_layer(
    p: &IntegralSimplifiedSmoothedConvexOptimization,
    alpha: f64,
    inverted: bool,
    t: i32,
    powering_up: bool,
    values: &Values,
    paths: &mut Paths<Vertice>,
    state_: Option<HandleLayerState>,
) -> Result<()> {
    let base_k = 1;
    let mut state = state_.unwrap_or_else(|| HandleLayerState {
        k: base_k,
        configs: vec![build_base_config(p.d, &p.bounds, values, powering_up)],
    });

    if state.k < 1 || state.k > p.d {
        return Ok(());
    }
    let mut added_configs = vec![];

    let dir = if powering_up {
        Direction::Next
    } else {
        Direction::Previous
    };
    for base_config in state.configs.iter() {
        let mut config = if state.k == base_k {
            base_config.clone()
        } else {
            build_config(dir, &values, &base_config, state.k).unwrap()
        };
        if state.k != base_k {
            added_configs.push(config.clone());
        }
        loop {
            handle_config(
                p,
                alpha,
                inverted,
                t,
                state.k,
                (powering_up, &config),
                values,
                paths,
            );
            config = match build_config(dir, &values, &config, state.k) {
                None => break,
                Some(config) => {
                    added_configs.push(config.clone());
                    config
                }
            };
        }
    }

    state.configs.append(&mut added_configs);
    state.k += 1;
    handle_layer(
        p,
        alpha,
        inverted,
        t,
        powering_up,
        values,
        paths,
        Some(state),
    )
}

/// updates paths up to vertex representing some config; then returns next config
#[allow(clippy::too_many_arguments)]
fn handle_config(
    p: &IntegralSimplifiedSmoothedConvexOptimization,
    alpha: f64,
    inverted: bool,
    t: i32,
    k: i32,
    (powering_up, config): (bool, &InternalConfig),
    values: &Values,
    paths: &mut Paths<Vertice>,
) {
    // find all immediate predecessors
    let predecessors = find_immediate_predecessors(
        p,
        alpha,
        inverted,
        t,
        powering_up,
        k,
        &config,
        values,
    );

    // determine shortest path
    let opt_predecessor = find_optimal_predecessor(predecessors, paths);

    // update paths
    update_paths(
        paths,
        opt_predecessor,
        &Vertice {
            config: config.clone(),
            powering_up,
        },
        config.config == IntegralConfig::repeat(0, p.d),
    );
}

fn build_base_config(
    d: i32,
    bounds: &Vec<i32>,
    values: &Values,
    bottom: bool,
) -> InternalConfig {
    if bottom {
        InternalConfig {
            config: IntegralConfig::repeat(0, d),
            indices: Config::repeat(0, d),
        }
    } else {
        InternalConfig {
            config: IntegralConfig::new(bounds.clone()),
            indices: Config::new(values.bound_indices.clone()),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Direction {
    Next,
    Previous,
}

/// selects the next config w.r.t. some dimension based on a given config in a given direction
fn build_config(
    dir: Direction,
    values: &Values,
    current_config: &InternalConfig,
    k_: i32,
) -> Option<InternalConfig> {
    let k = k_ as usize - 1;
    let mut config = current_config.clone();
    let i = match dir {
        Direction::Next => {
            if current_config.indices[k] == values.bound_indices[k] {
                return None;
            } else {
                current_config.indices[k] + 1
            }
        }
        Direction::Previous => {
            if current_config.indices[k] == 0 {
                return None;
            } else {
                current_config.indices[k] - 1
            }
        }
    };

    config.config[k] = values.values[i];
    config.indices[k] = i;
    Some(config)
}

#[allow(clippy::too_many_arguments)]
fn find_immediate_predecessors(
    p: &IntegralSimplifiedSmoothedConvexOptimization,
    alpha: f64,
    inverted: bool,
    t: i32,
    powering_up: bool,
    k: i32,
    config: &InternalConfig,
    all_values: &Values,
) -> Vec<Edge> {
    let mut predecessors: Vec<Edge> = (1..=k)
        .into_par_iter()
        .filter_map(|l| {
            let prev_config = build_config(
                if powering_up {
                    Direction::Previous
                } else {
                    Direction::Next
                },
                all_values,
                config,
                l,
            );
            match prev_config {
                None => None,
                Some(prev_config) => Some(Edge {
                    from: Vertice {
                        config: prev_config.clone(),
                        powering_up,
                    },
                    cost: if powering_up && !inverted
                        || !powering_up && inverted
                    {
                        alpha
                            * p.switching_cost[l as usize - 1]
                            * scalar_movement(
                                config.config[l as usize - 1],
                                prev_config.config[l as usize - 1],
                                inverted,
                            ) as f64
                    } else {
                        0.
                    },
                }),
            }
        })
        .collect();

    if t > 1 || !powering_up {
        let inaction = Edge {
            from: Vertice {
                config: config.clone(),
                powering_up: !powering_up,
            },
            cost: if powering_up {
                0.
            } else {
                p.hit_cost(t, config.config.clone()).raw()
            },
        };
        if config.config.to_vec() == vec![0, 0] {
            println!("{};{}", powering_up, inaction.cost)
        }
        predecessors.push(inaction);
    }

    predecessors
}

fn find_optimal_predecessor(
    predecessors: Vec<Edge>,
    paths: &mut Paths<Vertice>,
) -> Option<(Edge, Path)> {
    let r = predecessors
        .into_par_iter()
        .fold(
            || None,
            |picked: Option<(Edge, Path)>, predecessor| {
                let path = &paths[&predecessor.from];
                let new_cost = path.cost + predecessor.cost;

                // take smallest possible action if costs are equal
                let picked_cost = picked.clone().map_or_else(
                    || f64::INFINITY,
                    |(picked_predecessor, path)| {
                        path.cost + picked_predecessor.cost
                    },
                );
                if new_cost < picked_cost {
                    Some((predecessor, path.clone()))
                } else {
                    picked
                }
            },
        )
        .reduce(
            || None,
            |a, b| match a {
                Some(a) => Some(match b {
                    Some(b) => {
                        if a.0.cost + a.1.cost <= b.0.cost + b.1.cost {
                            a
                        } else {
                            b
                        }
                    }
                    None => a,
                }),
                None => b,
            },
        );
    debug!("{:?}", r);
    r
}

fn update_paths(
    paths: &mut Paths<Vertice>,
    predecessor: Option<(Edge, Path)>,
    to: &Vertice,
    is_initial_config: bool,
) {
    let path = match predecessor {
        None => {
            if is_initial_config {
                Path {
                    xs: IntegralSchedule::empty(),
                    cost: 0.,
                }
            } else {
                panic!("Problem is infeasible. Did not find a predecessor with a finite cost.")
            }
        }
        Some((predecessor, path)) => {
            let xs = if predecessor.from.powering_up && !to.powering_up {
                path.xs.extend(to.config.config.clone())
            } else {
                path.xs.clone()
            };
            Path {
                xs,
                cost: path.cost + predecessor.cost,
            }
        }
    };
    paths.insert(to.clone(), path);
}
