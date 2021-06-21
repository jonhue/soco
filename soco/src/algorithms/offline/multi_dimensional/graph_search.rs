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
            config: build_base_config(p.d),
            powering_up: false,
        })
        .ok_or(Error::PathsShouldBeCached)?
        .clone())
}

struct HandleLayerState {
    k: usize,
    configs: Vec<InternalConfig>,
}

impl Default for HandleLayerState {
    fn default() -> Self {
        HandleLayerState {
            k: 0,
            configs: vec![],
        }
    }
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
    let mut state = state_.unwrap_or_default();

    if state.k as i32 >= p.d {
        return Ok(());
    }
    if state.configs.is_empty() {
        state.configs.push(build_base_config(p.d));
    }
    let mut added_configs = vec![];

    for base_config in state.configs.iter() {
        for i in 0..all_values[state.k].len() {
            // build the next considered config
            let config = build_next_config(
                base_config,
                state.k,
                &all_values[state.k],
                i,
            );

            // find all immediate predecessors
            let predecessors = find_immediate_predecessors(
                p,
                inverted,
                t,
                powering_up,
                state.k,
                &config,
                all_values,
            )?;

            // determine shortest path
            let opt_predecessor =
                find_optimal_predecessor(predecessors, paths)?;

            // update paths
            update_paths(
                paths,
                opt_predecessor,
                &Vertice {
                    config: config.clone(),
                    powering_up,
                },
            )?;

            added_configs.push(config);
        }
    }

    state.configs.append(&mut added_configs);
    state.k += 1;
    handle_layer(p, inverted, t, powering_up, all_values, paths, Some(state))
}

fn build_base_config(d: i32) -> InternalConfig {
    InternalConfig {
        config: IntegralConfig::repeat(0, d),
        indices: Config::repeat(0, d),
    }
}

/// selects the previous config w.r.t. some dimension based on a given config
fn build_previous_config(
    all_values: &Values,
    current_config: &InternalConfig,
    k: usize,
) -> Option<InternalConfig> {
    let mut config = current_config.clone();
    let i = if current_config.indices[k] == 0 {
        return None;
    } else {
        current_config.indices[k] - 1
    };

    config.config[k] = all_values[k][i];
    config.indices[k] = i;
    Some(config)
}

/// given a config, a dimension, and the next value of that dimension, returns the updated config
fn build_next_config(
    prev_config: &InternalConfig,
    k: usize,
    values: &Vec<i32>,
    i: usize,
) -> InternalConfig {
    let mut config = prev_config.clone();
    config.config[k] = values[i];
    config.indices[k] = i;
    config
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

    for l in 0..k {
        let prev_config = build_previous_config(all_values, config, l);
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

        if new_cost
            < picked_predecessor.clone().map_or_else(
                || f64::INFINITY,
                |picked_predecessor| picked_predecessor.cost,
            )
        {
            picked_predecessor = Some(predecessor);
        }
    }
    Ok(picked_predecessor)
}

fn update_paths(
    paths: &mut Paths<Vertice>,
    edge: Option<Edge>,
    to: &Vertice,
) -> Result<()> {
    let path = match edge {
        None => Path {
            xs: IntegralSchedule::empty(),
            cost: 0.,
        },
        Some(edge) => {
            let prev_xs =
                &paths.get(&edge.from).ok_or(Error::PathsShouldBeCached)?.xs;
            let xs = if !to.powering_up {
                prev_xs.extend(to.config.config.clone())
            } else {
                prev_xs.clone()
            };
            Path {
                xs,
                cost: edge.cost,
            }
        }
    };
    paths.insert(to.clone(), path);
    Ok(())
}
