use crate::{init, utils::hash_map};
use soco::{
    algorithms::online::uni_dimensional::randomly_biased_greedy::{
        rbg, Memory, Options,
    },
    model::data_center::{
        loads::{LoadProfile, PredictedLoadProfile},
        model::{
            DataCenterModel, DataCenterModelOutputFailure,
            DataCenterModelOutputSuccess, DataCenterOfflineInput,
            DataCenterOnlineInput, JobType, Location, ServerType, Source,
            DEFAULT_KEY,
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
    streaming::online::{self, OfflineResponse},
};
use std::{
    sync::{mpsc::channel, Arc},
    thread,
};

#[test]
fn integration() {
    init();

    let addr = "127.0.0.1:5000";

    let t_end = 2;
    let m = 10;

    let (sender, receiver) = channel();
    let server = thread::spawn(move || {
        let delta = 1.;
        let model = DataCenterModel::new(
            delta,
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

        let OfflineResponse {
            xs: (xs, _),
            int_xs: (int_xs, _),
            ..
        } = online::start(
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
        int_xs.verify(t_end, &vec![m]).unwrap();
    });

    receiver.recv().unwrap();
    for t in 0..1 {
        let input = DataCenterOnlineInput {
            loads: vec![PredictedLoadProfile::raw(vec![vec![10.]])],
        };
        let ((x, _), (int_x, _), _) = online::next::<
            f64,
            FractionalSmoothedConvexOptimization<
                DataCenterModelOutputSuccess,
                DataCenterModelOutputFailure,
            >,
            Memory,
            DataCenterOnlineInput,
            DataCenterModelOutputSuccess,
            DataCenterModelOutputFailure,
        >(addr.parse().unwrap(), input);
        x.verify(t_end + t, &vec![m as f64]).unwrap();
        int_x.verify(t_end + t, &vec![m]).unwrap();
    }
    online::stop(addr.parse().unwrap());

    server.join().unwrap();
}
