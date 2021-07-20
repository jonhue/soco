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
