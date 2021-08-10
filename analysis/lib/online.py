from typing import List, Tuple
from soco.data_center.online import stop
from soco.data_center.online.uni_dimensional import (
    lazy_capacity_provisioning,
    memoryless,
    probabilistic,
    randomized,
    randomly_biased_greedy,
)
from soco.data_center.online.multi_dimensional import (
    lazy_budgeting,
    online_balanced_descent,
    online_gradient_descent,
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
    initial_xs = integral[0]
    xs = initial_xs.copy()
    ms = []
    for i in tqdm(range(len(online_inp))):
        fractional, integral, m, runtime = alg.next(ADDR, online_inp[i])
        cost = fractional[1][0]
        int_cost = integral[1][0]
        if integral[1][1] is not None:
            energy_cost = integral[1][1].energy_cost
            revenue_loss = integral[1][1].revenue_loss
        assert int_cost >= energy_cost + revenue_loss
        xs.append(integral[0])
        ms.append(m)
        runtimes.append(runtime)
    stop(ADDR)

    switching_cost = int_cost - energy_cost - revenue_loss

    assert len(xs) - len(initial_xs) == len(online_inp)
    print(f"Resulting schedule: {xs}")

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
        ms,
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


def evaluate_lazy_budgeting_slo(
    model: DataCenterModel,
    offline_inp: List[List[int]],
    online_inp: List[List[List[List[int]]]],
    randomized: bool = False,
) -> Tuple[float, float]:
    options = lazy_budgeting.smoothed_load_optimization.Options(randomized)
    (
        fractional,
        integral,
        m,
        initial_runtime,
    ) = lazy_budgeting.smoothed_load_optimization.start(
        ADDR, model, offline_inp, 0, options
    )
    return evaluate(
        lazy_budgeting.smoothed_load_optimization,
        online_inp,
        fractional,
        integral,
        m,
        initial_runtime,
    )


def evaluate_lazy_budgeting_sblo(
    model: DataCenterModel,
    offline_inp: List[List[int]],
    online_inp: List[List[List[List[int]]]],
    epsilon: float = 0.25,
) -> Tuple[float, float]:
    options = lazy_budgeting.smoothed_balanced_load_optimization.Options(epsilon)
    (
        fractional,
        integral,
        m,
        initial_runtime,
    ) = lazy_budgeting.smoothed_balanced_load_optimization.start(
        ADDR, model, offline_inp, 0, options
    )
    return evaluate(
        lazy_budgeting.smoothed_balanced_load_optimization,
        online_inp,
        fractional,
        integral,
        m,
        initial_runtime,
    )


def evaluate_pobd(
    model: DataCenterModel,
    offline_inp: List[List[int]],
    online_inp: List[List[List[List[int]]]],
    beta: float = 0.5,
) -> Tuple[float, float]:
    options = online_balanced_descent.primal.Options.euclidean_squared(beta)
    (
        fractional,
        integral,
        m,
        initial_runtime,
    ) = online_balanced_descent.primal.start(ADDR, model, offline_inp, 0, options)
    return evaluate(
        online_balanced_descent.primal,
        online_inp,
        fractional,
        integral,
        m,
        initial_runtime,
    )


def evaluate_dobd(
    model: DataCenterModel,
    offline_inp: List[List[int]],
    online_inp: List[List[List[List[int]]]],
    eta: float = 1,
) -> Tuple[float, float]:
    options = online_balanced_descent.dual.Options.euclidean_squared(eta)
    (
        fractional,
        integral,
        m,
        initial_runtime,
    ) = online_balanced_descent.dual.start(ADDR, model, offline_inp, 0, options)
    return evaluate(
        online_balanced_descent.dual,
        online_inp,
        fractional,
        integral,
        m,
        initial_runtime,
    )


def evaluate_ogd(
    model: DataCenterModel,
    offline_inp: List[List[int]],
    online_inp: List[List[List[List[int]]]],
) -> Tuple[float, float]:
    options = online_gradient_descent.Options.sqrt()
    (
        fractional,
        integral,
        m,
        initial_runtime,
    ) = online_gradient_descent.start(ADDR, model, offline_inp, 0, options)
    return evaluate(
        online_gradient_descent,
        online_inp,
        fractional,
        integral,
        m,
        initial_runtime,
    )
