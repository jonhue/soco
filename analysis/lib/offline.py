from typing import List, Tuple
from soco.data_center.offline import (
    brcp,
    optimal_graph_search_1d,
    OptimalGraphSearch1dOptions,
    optimal_graph_search,
    approx_graph_search,
    ApproxGraphSearchOptions,
    co,
)
from soco.data_center.model import DataCenterModel
import numpy as np


def evaluate_1d(model: DataCenterModel, inp: List[List[int]]) -> Tuple[float, float]:
    _, cost_brcp = brcp(model, inp, False)
    _, cost_optimal_graph_search_1d = optimal_graph_search_1d(
        model, inp, OptimalGraphSearch1dOptions(0), False
    )
    _, cost_optimal_graph_search = optimal_graph_search(model, inp, False)
    _, cost_co = co(model, inp, False)

    # # sanity checks
    # assert cost_optimal_graph_search_1d == cost_optimal_graph_search
    # assert cost_brcp == cost_co
    # assert cost_brcp <= cost_optimal_graph_search_1d

    return cost_brcp, cost_optimal_graph_search_1d


def evaluate(model: DataCenterModel, inp: List[List[int]]) -> Tuple[float, float]:
    _, cost_optimal_graph_search = optimal_graph_search(model, inp, False)
    # _, cost_approx_graph_search = approx_graph_search(model, inp, ApproxGraphSearchOptions(1.1), False)
    _, cost_co = co(model, inp, False)

    # sanity checks
    assert cost_co <= cost_optimal_graph_search


def evaluate_approx_graph_search(
    model: DataCenterModel, inp: List[List[int]]
) -> Tuple[np.array, np.array, float]:
    _, cost_optimal_graph_search = optimal_graph_search(model, inp, False)

    gammas = np.arange(1.1, 3, 0.1)
    costs = []
    for gamma in gammas:
        _, cost = approx_graph_search(
            model, inp, ApproxGraphSearchOptions(gamma), False
        )

        # sanity checks
        assert cost >= cost_optimal_graph_search

        costs.append(cost)

    return gammas, np.array(costs), cost_optimal_graph_search
