from typing import List, Tuple
from soco.data_center.online import (
    stop,
    lazy_capacity_provisioning,
    memoryless,
    probabilistic,
    randomized,
    randomly_biased_greedy,
)
from soco.data_center.model import DataCenterModel
from tqdm import tqdm


ADDR = "127.0.0.1:3449"


def evaluate(
    alg,
    online_inp: List[List[List[List[int]]]],
    fractional,
    integral,
    m,
    initial_runtime,
) -> Tuple[float, float]:
    initial_cost = fractional[1][0]
    initial_int_cost = integral[1][0]
    cost = initial_cost
    int_cost = initial_int_cost
    energy_cost = integral[1][1].energy_cost if integral[1][1] is not None else 0
    revenue_loss = integral[1][1].revenue_loss if integral[1][1] is not None else 0
    assert int_cost >= energy_cost + revenue_loss
    runtimes = []
    for i in tqdm(range(len(online_inp))):
        fractional, integral, m, runtime = alg.next(ADDR, online_inp[i])
        cost += fractional[1][0]
        int_cost += integral[1][0]
        energy_cost += integral[1][1].energy_cost
        revenue_loss += integral[1][1].revenue_loss
        assert int_cost >= energy_cost + revenue_loss
        runtimes.append(runtime)
    stop(ADDR)

    switching_cost = int_cost - energy_cost - revenue_loss
    return (
        initial_cost,
        cost,
        initial_int_cost,
        int_cost,
        energy_cost,
        revenue_loss,
        switching_cost,
        initial_runtime,
        runtimes,
    )


def evaluate_fractional_lazy_capacity_provisioning(
    model: DataCenterModel,
    offline_inp: List[List[int]],
    online_inp: List[List[List[List[int]]]],
    w: int = 0,
) -> Tuple[float, float]:
    (
        fractional,
        integral,
        m,
        initial_runtime,
    ) = lazy_capacity_provisioning.fractional.start(ADDR, model, offline_inp, w)
    return evaluate(
        lazy_capacity_provisioning.fractional,
        online_inp,
        fractional,
        integral,
        m,
        initial_runtime,
    )


def evaluate_integral_lazy_capacity_provisioning(
    model: DataCenterModel,
    offline_inp: List[List[int]],
    online_inp: List[List[List[List[int]]]],
    w: int = 0,
) -> Tuple[float, float]:
    (
        fractional,
        integral,
        m,
        initial_runtime,
    ) = lazy_capacity_provisioning.integral.start(ADDR, model, offline_inp, w)
    return evaluate(
        lazy_capacity_provisioning.integral,
        online_inp,
        fractional,
        integral,
        m,
        initial_runtime,
    )


def evaluate_memoryless(
    model: DataCenterModel,
    offline_inp: List[List[int]],
    online_inp: List[List[List[List[int]]]],
) -> Tuple[float, float]:
    fractional, integral, m, initial_runtime = memoryless.start(
        ADDR, model, offline_inp, 0
    )
    return evaluate(memoryless, online_inp, fractional, integral, m, initial_runtime)


def evaluate_probabilistic(
    model: DataCenterModel,
    offline_inp: List[List[int]],
    online_inp: List[List[List[List[int]]]],
) -> Tuple[float, float]:
    options = probabilistic.Options(probabilistic.Breakpoints([]))
    fractional, integral, m, initial_runtime = probabilistic.start(
        ADDR, model, offline_inp, 0, options
    )
    return evaluate(probabilistic, online_inp, fractional, integral, m, initial_runtime)


def evaluate_randomized_probabilistic(
    model: DataCenterModel,
    offline_inp: List[List[int]],
    online_inp: List[List[List[List[int]]]],
) -> Tuple[float, float]:
    fractional, integral, m, initial_runtime = randomized.probabilistic.start(
        ADDR, model, offline_inp, 0
    )
    return evaluate(
        randomized.probabilistic, online_inp, fractional, integral, m, initial_runtime
    )


def evaluate_randomized_randomly_biased_greedy(
    model: DataCenterModel,
    offline_inp: List[List[int]],
    online_inp: List[List[List[List[int]]]],
) -> Tuple[float, float]:
    fractional, integral, m, initial_runtime = randomized.randomly_biased_greedy.start(
        ADDR, model, offline_inp, 0
    )
    return evaluate(
        randomized.randomly_biased_greedy,
        online_inp,
        fractional,
        integral,
        m,
        initial_runtime,
    )


def evaluate_randomly_biased_greedy(
    model: DataCenterModel,
    offline_inp: List[List[int]],
    online_inp: List[List[List[List[int]]]],
    theta: float = 1,
) -> Tuple[float, float]:
    options = randomly_biased_greedy.Options(theta)
    fractional, integral, m, initial_runtime = randomly_biased_greedy.start(
        ADDR, model, offline_inp, 0, options
    )
    return evaluate(
        randomly_biased_greedy, online_inp, fractional, integral, m, initial_runtime
    )
