from typing import List, Tuple
from soco.data_center.offline import (
    OfflineOptions,
    brcp,
    optimal_graph_search_1d,
    OptimalGraphSearch1dOptions,
    optimal_graph_search,
    OptimalGraphSearchOptions,
    approx_graph_search,
    ApproxGraphSearchOptions,
    convex_optimization,
    static_integral,
)
from soco.data_center.model import DataCenterModel
import numpy as np


def evaluate_1d(model: DataCenterModel, inp: List[List[int]]) -> Tuple[float, float]:
    options = OfflineOptions(False, 1, None)

    # _, cost_brcp, runtime_brcp = brcp(model, inp, options)
    # (
    #     _,
    #     cost_optimal_graph_search_1d,
    #     runtime_optimal_graph_search_1d,
    # ) = optimal_graph_search_1d(model, inp, OptimalGraphSearch1dOptions(0), options)
    _, cost_optimal_graph_search, runtime_optimal_graph_search = optimal_graph_search(
        model, inp, OptimalGraphSearchOptions(), options
    )
    _, cost_co, runtime_co = convex_optimization(model, inp, options)

    # sanity checks
    # assert cost_optimal_graph_search_1d == cost_optimal_graph_search
    # assert cost_brcp == cost_co
    # assert cost_brcp <= cost_optimal_graph_search_1d

    return (
        cost_co[0],
        cost_optimal_graph_search[0],
        runtime_co,
        runtime_optimal_graph_search,
    )


def evaluate_static(
    model: DataCenterModel, inp: List[List[int]]
) -> Tuple[float, float]:
    options = OfflineOptions(False, 1, 0)

    _, cost_fractional, runtime_co = convex_optimization(model, inp, options)
    _, cost_integral, runtime_static_integral = static_integral(model, inp, options)

    # sanity checks
    # assert cost_fractional <= cost_integral

    return cost_fractional[0], cost_integral[0], runtime_co, runtime_static_integral


# def evaluate(model: DataCenterModel, inp: List[List[int]]) -> Tuple[float, float]:
#     _, cost_optimal_graph_search = optimal_graph_search(model, inp, False)
#     _, cost_co = convex_optimization(model, inp, False)

#     # sanity checks
#     assert cost_co <= cost_optimal_graph_search


def evaluate_approx_graph_search(
    model: DataCenterModel, inp: List[List[int]], gammas: List[float]
) -> Tuple[np.array, np.array, float]:
    options = OfflineOptions(False, 1, None)

    # _, cost_optimal_graph_search = optimal_graph_search(model, inp, False)

    costs = []
    runtimes = []
    for gamma in gammas:
        _, cost, runtime = approx_graph_search(
            model, inp, ApproxGraphSearchOptions(gamma), options
        )

        # # sanity checks
        # assert cost >= cost_optimal_graph_search

        costs.append(cost[0])
        runtimes.append(runtime)

    return gammas, np.array(costs), runtimes
