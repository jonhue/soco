from typing import List, Tuple
from soco.data_center.online import (
    stop,
)
from soco.data_center.model import DataCenterModel
import numpy as np
from tqdm import tqdm


def evaluate(
    alg,
    model: DataCenterModel,
    offline_inp: List[List[int]],
    online_inp: List[List[List[List[int]]]],
) -> Tuple[float, float]:
    addr = "127.0.0.1:3449"
    xs, initial_cost, m = alg.start(addr, model, offline_inp, 0)
    cost = initial_cost
    for i in tqdm(range(len(online_inp))):
        x, cost, m = alg.next(addr, online_inp[i])
    stop(addr)

    return initial_cost, cost


# def evaluate(model: DataCenterModel, inp: List[List[int]]) -> Tuple[float, float]:
#     _, cost_optimal_graph_search = optimal_graph_search(model, inp, False)
#     # _, cost_approx_graph_search = approx_graph_search(model, inp, ApproxGraphSearchOptions(1.1), False)
#     _, cost_co = co(model, inp, False)

#     # sanity checks
#     assert cost_co <= cost_optimal_graph_search


# def evaluate_approx_graph_search(
#     model: DataCenterModel, inp: List[List[int]]
# ) -> Tuple[np.array, np.array, float]:
#     _, cost_optimal_graph_search = optimal_graph_search(model, inp, False)

#     gammas = np.arange(1.1, 3, 0.1)
#     costs = []
#     for gamma in gammas:
#         _, cost = approx_graph_search(
#             model, inp, ApproxGraphSearchOptions(gamma), False
#         )

#         # sanity checks
#         assert cost >= cost_optimal_graph_search

#         costs.append(cost)

#     return gammas, np.array(costs), cost_optimal_graph_search
