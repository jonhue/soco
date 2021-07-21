from typing import Dict, List, Tuple, Union
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
    gamma: float,
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
        gamma,
        [Location(DEFAULT_KEY, m)],
        server_types,
        [Source(DEFAULT_KEY, lambda _t, _location: 0)],
        job_types,
        energy_consumption_model,
        {DEFAULT_KEY: energy_cost_model},
        revenue_loss_model,
        switching_cost_model,
    )


def network(
    delta: float,
    gamma: float,
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
        gamma,
        locations,
        server_types,
        sources,
        job_types,
        energy_consumption_model,
        energy_cost_model,
        revenue_loss_model,
        switching_cost_model,
    )


def model1(
    delta: float,
    server_types: List[ServerType],
    m: Dict[str, int],  # number of servers per server type
    job_types: List[JobType],
    energy_consumption_model: Dict[str, float],  # phi_max per server type
    switching_cost_model: SwitchingCosts,
) -> DataCenterModel:
    """Models a single data center using a linear energy consumption model, disregarding delay/revenue loss."""
    return single(
        delta,
        0,
        server_types,
        m,
        job_types,
        {
            key: SimplifiedLinearEnergyConsumptionModel(phi_max)
            for key, phi_max in energy_consumption_model.items()
        },
        LinearEnergyCostModel(
            lambda _t: 1.0
        ),  # energy cost is given directly by energy consumption
        {
            job_type.key: MinimalDetectableDelayRevenueLossModel(0)
            for job_type in job_types
        },  # irrelevant
        switching_cost_model,
    )


def model2(
    delta: float,
    server_types: List[ServerType],
    m: Dict[str, int],  # number of servers per server type
    job_types: List[JobType],
    energy_consumption_model: Dict[
        str, Tuple[float, float, float]
    ],  # (phi_min, alpha, beta) per server type
    switching_cost_model: SwitchingCosts,
) -> DataCenterModel:
    """Models a single data center using a nonlinear energy consumption model, disregarding delay/revenue loss."""
    return single(
        delta,
        0,
        server_types,
        m,
        job_types,
        {
            key: NonLinearEnergyConsumptionModel(phi_min, alpha, beta)
            for key, (phi_min, alpha, beta) in energy_consumption_model.items()
        },
        LinearEnergyCostModel(
            lambda _t: 1.0
        ),  # energy cost is given directly by energy consumption
        {
            job_type.key: MinimalDetectableDelayRevenueLossModel(0)
            for job_type in job_types
        },  # irrelevant
        switching_cost_model,
    )
