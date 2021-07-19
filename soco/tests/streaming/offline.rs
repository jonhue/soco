use crate::utils::hash_map;
use soco::{
    algorithms::offline::multi_dimensional::optimal_graph_search::optimal_graph_search,
    model::data_center::{
        loads::LoadProfile,
        model::{
            DataCenterModel, DataCenterOfflineInput, JobType, Location,
            ServerType, Source, DEFAULT_KEY,
        },
        models::{
            energy_consumption::{
                EnergyConsumptionModel, SimplifiedLinearEnergyConsumptionModel,
            },
            energy_cost::{EnergyCostModel, LinearEnergyCostModel},
            revenue_loss::{
                MinimalDetectableDelayRevenueLossModel, RevenueLossModel,
            },
            switching_cost::{SwitchingCost, SwitchingCostModel},
        },
    },
    streaming::offline,
};
use std::sync::Arc;

#[test]
fn solve() {
    let t_end = 100;
    let delta = 1.;
    let m = 10;
    let model = DataCenterModel::new(
        delta,
        0.,
        vec![Location {
            key: DEFAULT_KEY.to_string(),
            m: hash_map(&[(DEFAULT_KEY.to_string(), m)]),
        }],
        vec![ServerType::default()],
        vec![Source::default()],
        vec![JobType::default()],
        EnergyConsumptionModel::SimplifiedLinear(hash_map(&[(
            DEFAULT_KEY.to_string(),
            SimplifiedLinearEnergyConsumptionModel { phi_max: 1. },
        )])),
        EnergyCostModel::Linear(hash_map(&[(
            DEFAULT_KEY.to_string(),
            LinearEnergyCostModel {
                cost: Arc::new(|_| 1.),
            },
        )])),
        RevenueLossModel::MinimalDetectableDelay(hash_map(&[(
            DEFAULT_KEY.to_string(),
            MinimalDetectableDelayRevenueLossModel::default(),
        )])),
        SwitchingCostModel::new(hash_map(&[(
            DEFAULT_KEY.to_string(),
            SwitchingCost {
                energy_cost: 1.,
                phi_min: 0.5,
                phi_max: 1.,
                epsilon: 1.,
                delta: 1.,
                tau: 5.,
                rho: 5.,
            },
        )])),
    );
    let input = DataCenterOfflineInput {
        loads: vec![LoadProfile::raw(vec![10.]); t_end as usize],
    };

    let result =
        offline::solve(&model, &optimal_graph_search, (), input, false)
            .unwrap();
    result.xs.verify(t_end, &vec![m]).unwrap();
}
