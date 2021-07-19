// use soco::{
//     algorithms::online::uni_dimensional::randomly_biased_greedy::{
//         rbg, Options,
//     },
//     hash_map,
//     model::data_center::{
//         loads::LoadProfile,
//         model::{
//             DataCenterModel, DataCenterOfflineInput, JobType, ServerType,
//             DEFAULT_KEY,
//         },
//         models::{
//             delay::{DelayModel, ProcessorSharingQueueDelayModel},
//             energy_consumption::{
//                 EnergyConsumptionModel, SimplifiedLinearEnergyConsumptionModel,
//             },
//             energy_cost::{EnergyCostModel, LinearEnergyCostModel},
//             revenue_loss::{
//                 MinimalDetectableDelayRevenueLossModel, RevenueLossModel,
//             },
//             switching_cost::{SwitchingCost, SwitchingCostModel},
//         },
//     },
//     streaming::offline,
// };
// use std::sync::Arc;

// #[ignore]
// #[test]
// fn integration() {
//     let addr = "127.0.0.1:5000".parse().unwrap();

//     let t_end = 2;
//     let delta = 1.;
//     let m = 10;
//     let model = DataCenterModel::single(
//         delta,
//         0.,
//         vec![ServerType::default()],
//         hash_map(&[(DEFAULT_KEY.to_string(), m)]),
//         vec![JobType::default()],
//         EnergyConsumptionModel::SimplifiedLinear(hash_map(&[(
//             DEFAULT_KEY.to_string(),
//             SimplifiedLinearEnergyConsumptionModel { phi_max: 1. },
//         )])),
//         EnergyCostModel::Linear(hash_map(&[(
//             DEFAULT_KEY.to_string(),
//             LinearEnergyCostModel {
//                 cost: Arc::new(|_| 1.),
//             },
//         )])),
//         RevenueLossModel::MinimalDetectableDelay(hash_map(&[(
//             DEFAULT_KEY.to_string(),
//             MinimalDetectableDelayRevenueLossModel::default(),
//         )])),
//         DelayModel::ProcessorSharingQueue(ProcessorSharingQueueDelayModel {
//             mu: delta,
//         }),
//         SwitchingCostModel::new(hash_map(&[(
//             DEFAULT_KEY.to_string(),
//             SwitchingCost {
//                 energy_cost: 1.,
//                 phi_min: 0.5,
//                 phi_max: 1.,
//                 epsilon: 1.,
//                 delta: 1.,
//                 tau: 5.,
//                 rho: 5.,
//             },
//         )])),
//     );
//     let input = DataCenterOfflineInput {
//         loads: vec![LoadProfile::new(vec![10.]); t_end as usize],
//     };

//     let result =
//         offline::start(addr, &model, &rbg, Options::default(), 0, input)
//             .unwrap();
//     result.0.verify(t_end, &vec![m as f64]).unwrap();

//     offline::stop(addr);
// }
