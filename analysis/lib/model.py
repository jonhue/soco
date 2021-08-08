from typing import Dict, List, Literal, Tuple, Union
from soco.data_center.model import (
    LinearEnergyConsumptionModel,
    SimplifiedLinearEnergyConsumptionModel,
    NonLinearEnergyConsumptionModel,
    LinearEnergyCostModel,
    QuotasEnergyCostModel,
    MinimalDetectableDelayRevenueLossModel,
    SwitchingCost,
    Location,
    ServerType,
    Source,
    JobType,
    DataCenterModel,
)

EnergyConsumptionModel = Union[
    LinearEnergyConsumptionModel,
    SimplifiedLinearEnergyConsumptionModel,
    NonLinearEnergyConsumptionModel,
]
EnergyConsumptionModels = Dict[str, EnergyConsumptionModel]  # per server type

EnergyCostModel = Union[
    LinearEnergyCostModel,
    QuotasEnergyCostModel,
]
EnergyCostModels = Dict[str, EnergyCostModel]  # per location

RevenueLossModel = Union[
    MinimalDetectableDelayRevenueLossModel,
]
RevenueLossModels = Dict[str, RevenueLossModel]  # per job type

SwitchingCosts = Dict[str, SwitchingCost]  # per server type

DEFAULT_KEY = ""


def single(
    delta: float,
    server_types: List[ServerType],
    m: Dict[str, int],  # number of servers per server type
    job_types: List[JobType],
    energy_consumption_model: EnergyConsumptionModels,
    energy_cost_model: EnergyCostModel,
    revenue_loss_model: RevenueLossModels,
    switching_cost_model: SwitchingCosts,
) -> DataCenterModel:
    """Models a single data center."""
    return DataCenterModel(
        delta,
        [Location(DEFAULT_KEY, m)],
        server_types,
        [Source(DEFAULT_KEY, lambda _t, _location: ROUTING_DELAY)],
        job_types,
        energy_consumption_model,
        {DEFAULT_KEY: energy_cost_model},
        revenue_loss_model,
        switching_cost_model,
    )


def network(
    delta: float,
    locations: List[Location],
    server_types: List[ServerType],
    sources: List[Source],
    job_types: List[JobType],
    energy_consumption_model: EnergyConsumptionModels,
    energy_cost_model: EnergyCostModels,
    revenue_loss_model: RevenueLossModels,
    switching_cost_model: SwitchingCosts,
) -> DataCenterModel:
    """Models a network of data centers."""
    return DataCenterModel(
        delta,
        locations,
        server_types,
        sources,
        job_types,
        energy_consumption_model,
        energy_cost_model,
        revenue_loss_model,
        switching_cost_model,
    )


def linear_energy_cost(
    delta: float,
    server_types: List[ServerType],
    m: Dict[str, int],  # number of servers per server type
    job_types: List[JobType],
    energy_cost: float,  # average cost of a unit of energy
    energy_consumption_model: Dict[
        str, Tuple[float, float]
    ],  # (phi_min, phi_max) per server type
    revenue_loss_model: Dict[str, Tuple[float, float]],  # (gamma, delta) per job type
    switching_cost_model: SwitchingCosts,
) -> DataCenterModel:
    """Models a single data center using a linear energy consumption model."""
    return single(
        delta,
        server_types,
        m,
        job_types,
        {
            server_type: LinearEnergyConsumptionModel(phi_min, phi_max)
            for server_type, (phi_min, phi_max) in energy_consumption_model.items()
        },
        LinearEnergyCostModel(lambda _t: energy_cost),
        {
            job_type: MinimalDetectableDelayRevenueLossModel(gamma, delta)
            for job_type, (gamma, delta) in revenue_loss_model.items()
        },
        switching_cost_model,
    )


def nonlinear_energy_cost(
    delta: float,
    server_types: List[ServerType],
    m: Dict[str, int],  # number of servers per server type
    job_types: List[JobType],
    energy_cost: float,  # average cost of a unit of energy
    energy_consumption_model: Dict[
        str, Tuple[float, float, float]
    ],  # (phi_min, alpha, beta) per server type
    revenue_loss_model: Dict[str, Tuple[float, float]],  # (gamma, delta) per job type
    switching_cost_model: SwitchingCosts,
) -> DataCenterModel:
    """Models a single data center using a nonlinear energy consumption model."""
    return single(
        delta,
        server_types,
        m,
        job_types,
        {
            server_type: NonLinearEnergyConsumptionModel(phi_min, alpha, beta)
            for server_type, (phi_min, alpha, beta) in energy_consumption_model.items()
        },
        LinearEnergyCostModel(lambda _t: energy_cost),
        {
            job_type: MinimalDetectableDelayRevenueLossModel(gamma, delta)
            for job_type, (gamma, delta) in revenue_loss_model.items()
        },
        switching_cost_model,
    )


Trace = Union[
    Literal["facebook-2009-0"],
    Literal["facebook-2009-1"],
    Literal["facebook-2010"],
    Literal["lanl-mustang"],
    Literal["microsoft-fiddle"],
    Literal["alibaba"],
]
FACEBOOK_2009_0 = "facebook-2009-0"
FACEBOOK_2009_1 = "facebook-2009-1"
FACEBOOK_2010 = "facebook-2010"
LANL_MUSTANG = "lanl-mustang"
MICROSOFT_FIDDLE = "microsoft-fiddle"
ALIBABA = "alibaba"

MINUTE = 60
HOUR = 60 * 60

ROUTING_DELAY = 10

ENERGY_MODEL_WIERMAN = {
    "energy_cost": 1,
    "phi_min": 1,
    "phi_max": 1,
}
ENERGY_MODEL_ALTERNATIVE = {
    "energy_cost_per_hour": 0.0677,
    "phi_min": 0.5,
    "phi_max": 1,
}


def build_model(
    trace: Trace,
    energy_model,
    delta: float = 10 * MINUTE,
    normalized_switching_cost_in_hours: float = 1,  # in hours instead of time slots
    revenue_loss: float = 0.1,
) -> DataCenterModel:
    """Model 1 from the paper customized to each trace."""
    normalized_switching_cost = normalized_switching_cost_in_hours * HOUR / delta
    energy_cost = (
        energy_model["energy_cost"]
        if "energy_cost" in energy_model
        else energy_model["energy_cost_per_hour"] * delta / HOUR
    )
    if trace == FACEBOOK_2009_0 or trace == FACEBOOK_2009_1 or trace == FACEBOOK_2010:
        server_type = ServerType(DEFAULT_KEY, 1)
        job_type = JobType(DEFAULT_KEY, lambda _server_type: delta / 2)
        return linear_energy_cost(
            delta,
            [server_type],
            {DEFAULT_KEY: 1000 if trace == FACEBOOK_2010 else 600},
            [job_type],
            energy_cost,
            {DEFAULT_KEY: (energy_model["phi_min"], energy_model["phi_max"])},
            {DEFAULT_KEY: (revenue_loss, 2.5 * delta / 2)},
            {
                DEFAULT_KEY: SwitchingCost.from_normalized(
                    delta, normalized_switching_cost, energy_cost, energy_model["phi_min"]
                )
            },
        )
    if trace == LANL_MUSTANG:
        server_type = ServerType(DEFAULT_KEY, 1)
        job_type = JobType(DEFAULT_KEY, lambda _server_type: delta / 2)
        return linear_energy_cost(
            delta,
            [server_type],
            {DEFAULT_KEY: 1600},
            [job_type],
            energy_cost,
            {DEFAULT_KEY: (energy_model["phi_min"], energy_model["phi_max"])},
            {DEFAULT_KEY: (revenue_loss, 2.5 * delta / 2)},
            {
                DEFAULT_KEY: SwitchingCost.from_normalized(
                    delta, normalized_switching_cost, energy_cost, energy_model["phi_min"]
                )
            },
        )
    if trace == MICROSOFT_FIDDLE:
        GPU2 = "gpu2"
        GPU8 = "gpu8"
        gpu2 = ServerType(GPU2, 1)
        gpu8 = ServerType(GPU8, 1)
        job_type = JobType(
            DEFAULT_KEY,
            lambda server_type: delta / 2 if server_type == GPU2 else delta / (2 * 4),
        )
        return linear_energy_cost(
            delta,
            [gpu2, gpu8],
            {GPU2: 321, GPU8: 231},
            [job_type],
            energy_cost,
            {
                GPU2: (energy_model["phi_min"], energy_model["phi_max"]),
                GPU8: (
                    4 * energy_model["phi_min"],
                    4 * energy_model["phi_max"],
                ),
            },
            {DEFAULT_KEY: (revenue_loss, 2.5 * delta / 2)},
            {
                GPU2: SwitchingCost.from_normalized(
                    delta, normalized_switching_cost, energy_cost, energy_model["phi_min"]
                ),
                GPU8: SwitchingCost.from_normalized(
                    delta, 10 * normalized_switching_cost, energy_cost, energy_model["phi_min"]
                )
            },
        )
    if trace == ALIBABA:
        VERY_LONG = "very_long"
        LONG = "long"
        MEDIUM = "medium"
        SHORT = "short"
        server_type = ServerType(DEFAULT_KEY, 1)
        very_long_runtime = delta / 2
        very_long = JobType(VERY_LONG, lambda _server_type: very_long_runtime)
        long_runtime = very_long_runtime / 2.5
        long = JobType(LONG, lambda _server_type: long_runtime)
        medium_runtime = long_runtime / 2.5
        medium = JobType(MEDIUM, lambda _server_type: medium_runtime)
        short_runtime = medium_runtime / 2.5
        short = JobType(SHORT, lambda _server_type: short_runtime)
        return linear_energy_cost(
            delta,
            [server_type],
            {DEFAULT_KEY: 4000},
            [short, medium, long, very_long],
            energy_cost,
            {DEFAULT_KEY: (energy_model["phi_min"], energy_model["phi_max"])},
            {
                SHORT: (revenue_loss, 2.5 * short_runtime),
                MEDIUM: (revenue_loss, 2.5 * medium_runtime),
                LONG: (revenue_loss, 2.5 * long_runtime),
                VERY_LONG: (revenue_loss, 2.5 * very_long_runtime),
            },
            {
                DEFAULT_KEY: SwitchingCost.from_normalized(
                    delta, normalized_switching_cost, energy_cost, energy_model["phi_min"]
                )
            },
        )
    else:
        assert False, "Invalid trace."
