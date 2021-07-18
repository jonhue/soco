use crate::model::data_center::{
    model::{
        DataCenterModel, DataCenterOfflineInput, DataCenterOnlineInput,
        JobType, Location, ServerType, Source,
    },
    models::{
        delay::ProcessorSharingQueueDelayModel,
        energy_consumption::{
            LinearEnergyConsumptionModel, NonLinearEnergyConsumptionModel,
            SimplifiedLinearEnergyConsumptionModel,
        },
        energy_cost::{LinearEnergyCostModel, QuotasEnergyCostModel},
        revenue_loss::MinimalDetectableDelayRevenueLossModel,
        switching_cost::{SwitchingCost, SwitchingCostModel},
    },
};
use pyo3::prelude::*;

fn data_center_submodule(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<DataCenterModel>()?;

    m.add_class::<DataCenterOfflineInput>()?;
    m.add_class::<DataCenterOnlineInput>()?;

    m.add_class::<Location>()?;
    m.add_class::<ServerType>()?;
    m.add_class::<Source>()?;
    m.add_class::<JobType>()?;

    m.add_class::<ProcessorSharingQueueDelayModel>()?;

    m.add_class::<LinearEnergyConsumptionModel>()?;
    m.add_class::<SimplifiedLinearEnergyConsumptionModel>()?;
    m.add_class::<NonLinearEnergyConsumptionModel>()?;

    m.add_class::<LinearEnergyCostModel>()?;
    m.add_class::<QuotasEnergyCostModel>()?;

    m.add_class::<MinimalDetectableDelayRevenueLossModel>()?;

    m.add_class::<SwitchingCostModel>()?;
    m.add_class::<SwitchingCost>()?;

    Ok(())
}

pub fn submodule(py: Python, _m: &PyModule) -> PyResult<()> {
    let data_center = PyModule::new(py, "data_center")?;
    data_center_submodule(py, data_center)?;

    Ok(())
}
