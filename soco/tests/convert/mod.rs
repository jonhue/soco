mod into_sco {
    use crate::factories::{penalize_zero, random};
    use rand::prelude::*;
    use rand_pcg::Pcg64;
    use soco::config::Config;
    use soco::objective::Objective;
    use soco::problem::SimplifiedSmoothedConvexOptimization;
    use soco::schedule::Schedule;
    use soco::verifiers::VerifiableProblem;

    #[test]
    fn _1() {
        let p = SimplifiedSmoothedConvexOptimization {
            d: 2,
            t_end: 2,
            bounds: vec![2, 1],
            switching_cost: vec![1.5, 1.],
            hitting_cost: penalize_zero(),
        };
        p.verify().unwrap();
        let p_sco = p.clone().into_sco();
        p_sco.verify().unwrap();

        let xs = Schedule::new(vec![
            Config::new(vec![0, 1]),
            Config::new(vec![0, 1]),
        ]);

        assert!(p.objective_function(&xs).unwrap().raw().is_finite());
        assert_abs_diff_eq!(
            p.objective_function(&xs).unwrap().raw(),
            p_sco.objective_function(&xs).unwrap().raw(),
        );
    }

    #[test]
    fn _2() {
        let d = 2;
        let t_end = 100;
        let p = SimplifiedSmoothedConvexOptimization {
            d,
            t_end,
            bounds: vec![8, 8],
            switching_cost: vec![1., 3.],
            hitting_cost: random(),
        };
        p.verify().unwrap();
        let p_sco = p.clone().into_sco();
        p_sco.verify().unwrap();

        let xs = Schedule::new(
            (0..t_end)
                .map(|t| {
                    Config::new(
                        (0..d)
                            .map(|k| {
                                Pcg64::seed_from_u64((t * k) as u64)
                                    .gen_range(0..8)
                            })
                            .collect(),
                    )
                })
                .collect(),
        );

        assert!(p.objective_function(&xs).unwrap().raw().is_finite());
        assert_abs_diff_eq!(
            p.objective_function(&xs).unwrap().raw(),
            p_sco.objective_function(&xs).unwrap().raw(),
        );
    }
}

mod into_ssco {
    use crate::factories::inv_e_sblo;
    use rand::prelude::*;
    use rand_pcg::Pcg64;
    use soco::config::Config;
    use soco::objective::Objective;
    use soco::problem::SmoothedBalancedLoadOptimization;
    use soco::schedule::Schedule;
    use soco::verifiers::VerifiableProblem;

    #[test]
    fn _1() {
        let d = 2;
        let p = SmoothedBalancedLoadOptimization {
            d,
            t_end: 2,
            bounds: vec![2, 1],
            switching_cost: vec![1.5, 1.],
            hitting_cost: vec![inv_e_sblo(); d as usize],
            load: vec![1, 1],
        };
        p.verify().unwrap();
        let p_ssco = p.clone().into_ssco();
        p_ssco.verify().unwrap();

        let xs = Schedule::new(vec![
            Config::new(vec![0, 1]),
            Config::new(vec![0, 1]),
        ]);

        assert!(p_ssco.objective_function(&xs).unwrap().raw().is_finite());
        p_ssco.objective_function(&xs).unwrap();
    }

    #[test]
    fn _2() {
        let d = 2;
        let t_end = 100;
        let p = SmoothedBalancedLoadOptimization {
            d,
            t_end,
            bounds: vec![8, 8],
            switching_cost: vec![1., 3.],
            hitting_cost: vec![inv_e_sblo(); d as usize],
            load: (0..t_end)
                .map(|t| Pcg64::seed_from_u64(t as u64).gen_range(1..8))
                .collect(),
        };
        p.verify().unwrap();
        let p_ssco = p.clone().into_ssco();
        p_ssco.verify().unwrap();

        let xs = Schedule::new(
            (0..t_end)
                .map(|t| {
                    Config::new(
                        (0..d)
                            .map(|k| {
                                Pcg64::seed_from_u64((t * k) as u64)
                                    .gen_range(0..8)
                            })
                            .collect(),
                    )
                })
                .collect(),
        );

        assert!(p_ssco.objective_function(&xs).unwrap().raw().is_finite());
        p_ssco.objective_function(&xs).unwrap();
    }
}

mod into_sblo {
    use rand::prelude::*;
    use rand_pcg::Pcg64;
    use soco::config::Config;
    use soco::objective::Objective;
    use soco::problem::SmoothedLoadOptimization;
    use soco::schedule::Schedule;
    use soco::verifiers::VerifiableProblem;

    #[test]
    fn _1() {
        let d = 2;
        let p = SmoothedLoadOptimization {
            d,
            t_end: 2,
            bounds: vec![2, 1],
            switching_cost: vec![1.5, 1.],
            hitting_cost: vec![1., 2.],
            load: vec![1, 1],
        };
        p.verify().unwrap();
        let p_ssco = p.clone().into_sblo().into_ssco();
        p_ssco.verify().unwrap();

        let xs = Schedule::new(vec![
            Config::new(vec![0, 1]),
            Config::new(vec![0, 1]),
        ]);

        assert!(p.objective_function(&xs).unwrap().raw().is_finite());
        assert_abs_diff_eq!(
            p.objective_function(&xs).unwrap().raw(),
            p_ssco.objective_function(&xs).unwrap().raw(),
        );
    }

    #[test]
    fn _2() {
        let d = 2;
        let t_end = 100;
        let p = SmoothedLoadOptimization {
            d,
            t_end,
            bounds: vec![8, 8],
            switching_cost: vec![3., 1.],
            hitting_cost: vec![1., 2.],
            load: (0..t_end)
                .map(|t| Pcg64::seed_from_u64(t as u64).gen_range(1..8))
                .collect(),
        };
        p.verify().unwrap();
        let p_ssco = p.clone().into_sblo().into_ssco();
        p_ssco.verify().unwrap();

        let xs = Schedule::new(
            (0..t_end)
                .map(|t| {
                    Config::new(
                        (0..d)
                            .map(|k| {
                                Pcg64::seed_from_u64((t * k) as u64)
                                    .gen_range(0..8)
                            })
                            .collect(),
                    )
                })
                .collect(),
        );

        assert!(p.objective_function(&xs).unwrap().raw().is_finite());
        assert_abs_diff_eq!(
            p.objective_function(&xs).unwrap().raw(),
            p_ssco.objective_function(&xs).unwrap().raw(),
        );
    }
}
