use crate::algorithms::graph_search::{Path, Paths};
use crate::algorithms::offline::OfflineOptions;
use crate::config::{Config, IntegralConfig};
use crate::objective::scalar_movement;
use crate::problem::IntegralSimplifiedSmoothedConvexOptimization;
use crate::result::{Error, Result};
use crate::schedule::IntegralSchedule;
use crate::utils::assert;
use std::collections::HashMap;

/// Lists for each dimension the number of possible values of a config, from smallest to largest.
pub type Values = Vec<Vec<i32>>;

/// Tracks the value as well as the index in which the value appears in the respective dimension.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct InternalConfig {
    config: IntegralConfig,
    indices: Config<usize>,
}

/// Vertice in the graph denoting time `t` and the value `x` at time `t`.
/// The boolean flag indicates whether the vertice belongs to the powering up (`true`) or powering down (`false`) phase.
/// The algorithm only keeps the most recent two layers in memory.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct Vertice {
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
pub fn graph_search<'a>(
    p: &'a IntegralSimplifiedSmoothedConvexOptimization<'a>,
    all_values: Values,
    offline_options: &OfflineOptions,
) -> Result<Path> {
    assert(all_values.len() as i32 == p.d, Error::DimensionInconsistent)?;

    let mut paths: Paths<Vertice> = HashMap::new();
    for t in 1..=p.t_end {
        handle_layer(
            p,
            offline_options.inverted,
            t,
            true,
            &all_values,
            &mut paths,
            None,
        )?;
        handle_layer(
            p,
            offline_options.inverted,
            t,
            false,
            &all_values,
            &mut paths,
            None,
        )?;
    }

    Ok(paths
        .get(&Vertice {
            config: build_base_config(&all_values, true),
            powering_up: false,
        })
        .ok_or(Error::PathsShouldBeCached)?
        .clone())
}

struct HandleLayerState {
    k: i32,
    configs: Vec<InternalConfig>,
}

fn handle_layer(
    p: &IntegralSimplifiedSmoothedConvexOptimization,
    inverted: bool,
    t: i32,
    powering_up: bool,
    all_values: &Values,
    paths: &mut Paths<Vertice>,
    state_: Option<HandleLayerState>,
) -> Result<()> {
    let mut state = state_.unwrap_or_else(|| {
        let configs = vec![build_base_config(all_values, powering_up)];
        if powering_up {
            HandleLayerState { k: 1, configs }
        } else {
            HandleLayerState { k: p.d, configs }
        }
    });

    if state.k < 1 || state.k > p.d {
        return Ok(());
    }
    let mut added_configs = vec![];

    for base_config in state.configs.iter() {
        let mut config = base_config.clone();
        loop {
            // find all immediate predecessors
            let predecessors = find_immediate_predecessors(
                p,
                inverted,
                t,
                powering_up,
                state.k as usize - 1,
                &config,
                all_values,
            )?;

            // determine shortest path
            let opt_predecessor =
                find_optimal_predecessor(inverted, predecessors, paths)?;

            // update paths
            update_paths(
                paths,
                opt_predecessor,
                &Vertice {
                    config: config.clone(),
                    powering_up,
                },
            )?;

            // build the next considered config
            config = match build_config(
                if powering_up {
                    Direction::Next
                } else {
                    Direction::Previous
                },
                &all_values,
                &config,
                state.k as usize - 1,
            ) {
                None => break,
                Some(config) => {
                    added_configs.push(config.clone());
                    config
                }
            };
        }
    }

    state.configs.append(&mut added_configs);
    if powering_up {
        state.k += 1
    } else {
        state.k -= 1
    };
    handle_layer(p, inverted, t, powering_up, all_values, paths, Some(state))
}

fn build_base_config(all_values: &Values, bottom: bool) -> InternalConfig {
    let mut config = IntegralConfig::empty();
    let mut indices = Config::empty();
    for values in all_values {
        let i = if bottom { 0 } else { values.len() - 1 };
        config.push(values[i]);
        indices.push(i);
    }
    InternalConfig { config, indices }
}

enum Direction {
    Next,
    Previous,
}

/// selects the next config w.r.t. some dimension based on a given config in a given direction
fn build_config(
    dir: Direction,
    all_values: &Values,
    current_config: &InternalConfig,
    k: usize,
) -> Option<InternalConfig> {
    let mut config = current_config.clone();
    let i = match dir {
        Direction::Next => {
            if current_config.indices[k] == all_values[k].len() - 1 {
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

    config.config[k] = all_values[k][i];
    config.indices[k] = i;
    Some(config)
}

fn find_immediate_predecessors(
    p: &IntegralSimplifiedSmoothedConvexOptimization,
    inverted: bool,
    t: i32,
    powering_up: bool,
    k: usize,
    config: &InternalConfig,
    all_values: &Values,
) -> Result<Vec<Edge>> {
    let mut predecessors = vec![];

    if t > 1 || !powering_up {
        let inaction = Edge {
            from: Vertice {
                config: config.clone(),
                powering_up: !powering_up,
            },
            cost: if powering_up {
                0.
            } else {
                (p.hitting_cost)(t, config.config.clone())
                    .ok_or(Error::CostFnMustBeTotal)?
            },
        };
        predecessors.push(inaction);
    }

    for l in 0..=k {
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
            None => (),
            Some(prev_config) => {
                let action = Edge {
                    from: Vertice {
                        config: prev_config.clone(),
                        powering_up,
                    },
                    cost: if powering_up && !inverted
                        || !powering_up && inverted
                    {
                        p.switching_cost[l]
                            * scalar_movement(
                                config.config[l],
                                prev_config.config[l],
                                inverted,
                            ) as f64
                    } else {
                        0.
                    },
                };
                predecessors.push(action);
            }
        }
    }

    Ok(predecessors)
}

fn find_optimal_predecessor(
    inverted: bool,
    predecessors: Vec<Edge>,
    paths: &mut Paths<Vertice>,
) -> Result<Option<Edge>> {
    let mut picked_predecessor: Option<Edge> = None;
    for predecessor in predecessors {
        let prev_cost = paths
            .get(&predecessor.from)
            .ok_or(Error::PathsShouldBeCached)?
            .cost;
        let new_cost = prev_cost + predecessor.cost;

        // take smallest possible action if costs are equal
        let picked_cost = picked_predecessor.clone().map_or_else(
            || f64::INFINITY,
            |picked_predecessor| picked_predecessor.cost,
        );
        if !inverted && new_cost < picked_cost
            || inverted && new_cost <= picked_cost
        {
            picked_predecessor = Some(predecessor);
        }
    }
    Ok(picked_predecessor)
}

fn update_paths(
    paths: &mut Paths<Vertice>,
    predecessor: Option<Edge>,
    to: &Vertice,
) -> Result<()> {
    let path = match predecessor {
        None => Path {
            xs: IntegralSchedule::empty(),
            cost: 0.,
        },
        Some(predecessor) => {
            let prev = &paths
                .get(&predecessor.from)
                .ok_or(Error::PathsShouldBeCached)?;
            let xs = if predecessor.from.powering_up && !to.powering_up {
                prev.xs.extend(to.config.config.clone())
            } else {
                prev.xs.clone()
            };
            Path {
                xs,
                cost: prev.cost + predecessor.cost,
            }
        }
    };
    paths.insert(to.clone(), path);
    Ok(())
}
