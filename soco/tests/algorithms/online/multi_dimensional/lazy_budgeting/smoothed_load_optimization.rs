mod lb {
    // use crate::factories::inv_e;
    // use soco::algorithms::online::multi_dimensional::lazy_budgeting::smoothed_load_optimization::{lb, Options};
    // use soco::config::Config;
    // use soco::convert::DiscretizableSchedule;
    // use soco::cost::CostFn;
    // use soco::problem::{Online, SmoothedLoadOptimization};
    // use soco::schedule::Schedule;

    // #[test]
    // fn _1() {
    //     let p = SmoothedLoadOptimization {
    //         d: 2,
    //         t_end: 1,
    //         bounds: vec![2, 1],
    //         switching_cost: vec![2., 1.],
    //         hitting_cost: vec![1., 2.],
    //         load: vec![1, 2, 0, 2, 1],
    //     };
    //     let mut o = Online { p, w: 0 };
    //     o.verify().unwrap();

    //     let result = o
    //         .offline_stream(lb, 5, Options { randomized: false })
    //         .unwrap();
    //     result.0.verify(o.p.t_end, &o.p.bounds).unwrap();

    //     assert_eq!(result.0, Schedule::new(vec![Config::single(1)]));
    // }

    // #[test]
    // fn _2() {
    //     let p = SimplifiedSmoothedConvexOptimization {
    //         d: 1,
    //         t_end: 1,
    //         bounds: vec![2.],
    //         switching_cost: vec![1.],
    //         hitting_cost: CostFn::new(inv_e),
    //     };
    //     let mut o = Online { p, w: 0 };
    //     o.verify().unwrap();

    //     let t_end = 2;
    //     let result = o.offline_stream(memoryless, t_end, ()).unwrap();
    //     result.0.verify(t_end, &o.p.bounds).unwrap();

    //     assert_eq!(
    //         result.0.to_i(),
    //         Schedule::new(vec![Config::single(1), Config::single(1)])
    //     );
    // }
}
