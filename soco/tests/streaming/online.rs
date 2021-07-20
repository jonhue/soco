use crate::utils::hash_map;
use soco::{
    algorithms::online::uni_dimensional::randomly_biased_greedy::{
        rbg, Memory, Options,
    },
    model::data_center::{
        loads::LoadProfile,
        model::{
            DataCenterModel, DataCenterOfflineInput, DataCenterOnlineInput,
            JobType, Location, ServerType, Source, DEFAULT_KEY,
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
    problem::FractionalSmoothedConvexOptimization,
    streaming::online,
};
use std::{
    sync::{mpsc::channel, Arc},
    thread,
};

#[test]
fn integration() {
    let addr = "127.0.0.1:5000";

    let t_end = 2;
    let m = 10;

    let (sender, receiver) = channel();
    let server = thread::spawn(move || {
        let delta = 1.;
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

        let options = Options::default();

        let (xs, _, _) = online::start(
            addr.parse().unwrap(),
            model,
            &rbg,
            options,
            0,
            input,
            Some(sender),
        )
        .unwrap();
        xs.verify(t_end, &vec![m as f64]).unwrap();
    });

    receiver.recv().unwrap();
    for t in 0..1 {
        let input = DataCenterOnlineInput {
            loads: vec![vec![LoadProfile::raw(vec![10.])]],
        };
        let (x, _, _) = online::next::<
            f64,
            FractionalSmoothedConvexOptimization,
            Memory,
            DataCenterOnlineInput,
        >(addr.parse().unwrap(), input);
        x.verify(t_end + t, &vec![m as f64]).unwrap();
    }
    online::stop(addr.parse().unwrap());

    server.join().unwrap();
}