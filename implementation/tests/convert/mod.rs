#[cfg(test)]
mod into_sco {
    use crate::factories::{penalize_zero, random};
    use crate::init;
    use rand::prelude::*;
    use rand_pcg::Pcg64;
    use soco::config::Config;
    use soco::problem::{Problem, SimplifiedSmoothedConvexOptimization};
    use soco::schedule::Schedule;
    use soco::verifiers::VerifiableProblem;

    #[test]
    fn _1() {
        init();

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

        assert!(p.objective_function(&xs).unwrap().cost.raw().is_finite());
        assert_abs_diff_eq!(
            p.objective_function(&xs).unwrap().cost.raw(),
            p_sco.objective_function(&xs).unwrap().cost.raw(),
        );
    }

    #[test]
    fn _2() {
        init();

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

        assert!(p.objective_function(&xs).unwrap().cost.raw().is_finite());
        assert_abs_diff_eq!(
            p.objective_function(&xs).unwrap().cost.raw(),
            p_sco.objective_function(&xs).unwrap().cost.raw(),
        );
    }
}

#[cfg(test)]
mod into_ssco {
    use crate::factories::inv_e_sblo;
    use crate::init;
    use rand::prelude::*;
    use rand_pcg::Pcg64;
    use soco::config::Config;
    use soco::problem::{Problem, SmoothedBalancedLoadOptimization};
    use soco::schedule::Schedule;
    use soco::verifiers::VerifiableProblem;

    #[test]
    fn _1() {
        init();

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
            Config::new(vec![2, 1]),
            Config::new(vec![2, 1]),
        ]);

        assert!(p_ssco
            .objective_function(&xs)
            .unwrap()
            .cost
            .raw()
            .is_finite());
        p_ssco.objective_function(&xs).unwrap();
    }

    #[test]
    fn _2() {
        init();

        let d = 2;
        let t_end = 100;
        let p = SmoothedBalancedLoadOptimization {
            d,
            t_end,
            bounds: vec![8, 8],
            switching_cost: vec![1., 3.],
            hitting_cost: vec![inv_e_sblo(); d as usize],
            load: (0..t_end)
                .map(|t| Pcg64::seed_from_u64(t as u64).gen_range(0..8))
                .collect(),
        };
        p.verify().unwrap();
        let p_ssco = p.clone().into_ssco();
        p_ssco.verify().unwrap();

        let xs = Schedule::new(
            (0..t_end as usize)
                .map(|t| {
                    Config::new(
                        (0..d)
                            .map(|k| {
                                Pcg64::seed_from_u64((t as i32 * k) as u64)
                                    .gen_range(p.load[t]..8)
                            })
                            .collect(),
                    )
                })
                .collect(),
        );

        assert!(p_ssco
            .objective_function(&xs)
            .unwrap()
            .cost
            .raw()
            .is_finite());
        p_ssco.objective_function(&xs).unwrap();
    }
}

#[cfg(test)]
mod into_sblo {
    use rand::prelude::*;
    use rand_pcg::Pcg64;
    use soco::config::Config;
    use soco::problem::{Problem, SmoothedLoadOptimization};
    use soco::schedule::Schedule;
    use soco::verifiers::VerifiableProblem;

    use crate::init;

    #[test]
    fn _1() {
        init();

        let p = SmoothedLoadOptimization {
            d: 2,
            t_end: 2,
            bounds: vec![2, 1],
            switching_cost: vec![1., 1.5],
            hitting_cost: vec![2., 1.],
            load: vec![1, 1],
        };
        p.verify().unwrap();
        let p_ssco = p.clone().into_sblo().into_ssco();
        p_ssco.verify().unwrap();

        let xs = Schedule::new(vec![
            Config::new(vec![2, 1]),
            Config::new(vec![2, 1]),
        ]);

        assert!(p.objective_function(&xs).unwrap().cost.raw().is_finite());
        assert_abs_diff_eq!(
            p.total_movement(&xs, false).unwrap().raw(),
            p_ssco.total_movement(&xs, false).unwrap().raw(),
        );
        assert_abs_diff_eq!(
            p.objective_function(&xs).unwrap().cost.raw(),
            p_ssco.objective_function(&xs).unwrap().cost.raw(),
        );
    }

    #[test]
    fn _2() {
        init();

        let p = SmoothedLoadOptimization {
            d: 2,
            t_end: 2,
            bounds: vec![2, 1],
            switching_cost: vec![1., 2.],
            hitting_cost: vec![2., 1.],
            load: vec![1, 2],
        };
        p.verify().unwrap();
        let p_ssco = p.clone().into_sblo().into_ssco();
        p_ssco.verify().unwrap();

        let xs = Schedule::new(vec![
            Config::new(vec![0, 1]),
            Config::new(vec![0, 1]),
        ]);

        assert!(p.objective_function(&xs).unwrap().cost.raw().is_infinite());
        assert!(p_ssco
            .objective_function(&xs)
            .unwrap()
            .cost
            .raw()
            .is_infinite());
    }

    #[test]
    fn _3() {
        init();

        let d = 2;
        let t_end = 100;
        let p = SmoothedLoadOptimization {
            d,
            t_end,
            bounds: vec![8, 8],
            switching_cost: vec![1., 3.],
            hitting_cost: vec![2., 1.],
            load: (0..t_end)
                .map(|t| Pcg64::seed_from_u64(t as u64).gen_range(0..8))
                .collect(),
        };
        p.verify().unwrap();
        let p_ssco = p.clone().into_sblo().into_ssco();
        p_ssco.verify().unwrap();

        let xs = Schedule::new(
            (0..t_end as usize)
                .map(|t| {
                    Config::new(
                        (0..d)
                            .map(|k| {
                                Pcg64::seed_from_u64((t as i32 * k) as u64)
                                    .gen_range(p.load[t]..8)
                            })
                            .collect(),
                    )
                })
                .collect(),
        );

        assert!(p.objective_function(&xs).unwrap().cost.raw().is_finite());
        assert_abs_diff_eq!(
            p.total_movement(&xs, false).unwrap().raw(),
            p_ssco.total_movement(&xs, false).unwrap().raw(),
        );
        assert_abs_diff_eq!(
            p.objective_function(&xs).unwrap().cost.raw(),
            p_ssco.objective_function(&xs).unwrap().cost.raw(),
        );
    }
}

#[cfg(test)]
mod reset {
    use crate::factories::moving_parabola;
    use soco::config::Config;
    use soco::convert::Resettable;
    use soco::problem::{Problem, SimplifiedSmoothedConvexOptimization};
    use soco::schedule::Schedule;

    use crate::init;

    #[test]
    fn _1() {
        init();

        let p = SimplifiedSmoothedConvexOptimization {
            d: 1,
            t_end: 10,
            bounds: vec![10.],
            switching_cost: vec![1.],
            hitting_cost: moving_parabola(5),
        };

        let xs = Schedule::new(vec![
            Config::single(1.0),
            Config::single(2.0),
            Config::single(3.0),
            Config::single(3.5),
            Config::single(0.5),
            Config::single(1.0),
            Config::single(2.0),
            Config::single(3.0),
            Config::single(3.5),
            Config::single(0.0),
        ]);
        let cost = p.objective_function(&xs).unwrap().cost.raw();

        let p1 = p.reset(2);
        let cost1 = p1.objective_function(&xs.reset(2)).unwrap().cost.raw();

        let p2 = p1.reset(3);
        let cost2 = p2.objective_function(&xs.reset(5)).unwrap().cost.raw();

        assert_eq!(xs.get(3), xs.reset(2).get(1));

        assert_eq!(
            p.hit_cost(9, xs.get(9).unwrap().clone()).cost.raw(),
            p1.hit_cost(7, xs.get(9).unwrap().clone()).cost.raw()
        );
        assert_eq!(
            p.hit_cost(9, xs.get(9).unwrap().clone()).cost.raw(),
            p2.hit_cost(4, xs.get(9).unwrap().clone()).cost.raw()
        );

        assert!(cost.is_finite());
        assert_eq!(
            cost - p.hit_cost(1, xs.get(1).unwrap().clone()).cost.raw()
                - p.hit_cost(2, xs.get(2).unwrap().clone()).cost.raw()
                - p.movement(
                    Config::single(0.),
                    xs.get(1).unwrap().clone(),
                    false
                )
                .raw()
                - p.movement(
                    xs.get(1).unwrap().clone(),
                    xs.get(2).unwrap().clone(),
                    false
                )
                .raw()
                - p.movement(
                    xs.get(2).unwrap().clone(),
                    xs.get(3).unwrap().clone(),
                    false
                )
                .raw()
                + p.movement(
                    Config::single(0.),
                    xs.get(3).unwrap().clone(),
                    false
                )
                .raw(),
            cost1
        );
        assert_eq!(
            cost1
                - p.hit_cost(3, xs.get(3).unwrap().clone()).cost.raw()
                - p.hit_cost(4, xs.get(4).unwrap().clone()).cost.raw()
                - p.hit_cost(5, xs.get(5).unwrap().clone()).cost.raw()
                - p.movement(
                    Config::single(0.),
                    xs.get(3).unwrap().clone(),
                    false
                )
                .raw()
                - p.movement(
                    xs.get(3).unwrap().clone(),
                    xs.get(4).unwrap().clone(),
                    false
                )
                .raw()
                - p.movement(
                    xs.get(4).unwrap().clone(),
                    xs.get(5).unwrap().clone(),
                    false
                )
                .raw()
                - p.movement(
                    xs.get(5).unwrap().clone(),
                    xs.get(6).unwrap().clone(),
                    false
                )
                .raw()
                + p.movement(
                    Config::single(0.),
                    xs.get(6).unwrap().clone(),
                    false
                )
                .raw(),
            cost2
        );
    }
}
