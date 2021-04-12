struct Problem<T> {
    // number of servers (homogeneous)
    m: i32,
    // finite time horizon
    t_end: usize,
    // non-negative convex cost functions for 1<=t<=T; 0<=x_t<=m
    // at time t, f constains all cost functions up to time t
    fs: Vec<fn(T) -> f64>,
    // positive real constant resembling the switching cost
    beta: f64,
}

fn verify_problem<T>(p: &Problem<T>) {
    assert_eq!(
        p.fs.len(),
        p.t_end,
        "problem must include a cost function for each time step"
    );
    assert!(p.beta > 0., "beta must be positive");
}

fn verify_discrete_schedule(p: &Problem<i32>, xs: &Schedule<i32>) {
    let min = xs.iter().min();
    let max = xs.iter().max();

    assert_eq!(
        xs.len(),
        p.t_end,
        "schedule must have a value for each time step"
    );
    match min {
        Some(i) => assert!(i >= &0, "values in schedule must be non-negative"),
        _ => (),
    }
    match max {
        Some(i) => assert!(
            i <= &p.m,
            "values in schedule must not exceed number of servers"
        ),
        _ => (),
    }
}

// number of active servers from time 1 to time T
type Schedule<T> = Vec<T>;

fn discrete_max(value: i32) -> f64 {
    if value >= 0 {
        return value as f64;
    } else {
        return 0.;
    }
}

// fn continuous_max(value: f64) -> f64 {
//     if value >= 0. {
//         return value;
//     } else {
//         return 0.;
//     }
// }

fn discrete_objective_function(p: &Problem<i32>, xs: &Schedule<i32>) -> f64 {
    let mut cost = 0.;
    for t in 0..p.t_end {
        let prev_x = if t > 0 { xs[t - 1] } else { 0 };
        cost += p.fs[t](xs[t]) + p.beta * discrete_max(xs[t] - prev_x);
    }
    return cost;
}

// fn continuous_objective_function(P: &Problem<f64>, X: &Schedule<f64>) -> f64 {
//     let mut cost = 0.;
//     for t in 1..P.T {
//         cost += P.f[t](X[t]) + P.beta * continuous_max(X[t] - X[t - 1]);
//     }
//     return cost;
// }

fn f1(t: i32) -> f64 {
    return (t as f64) + 1.;
}

fn main() {
    println!("Hello, world!");

    let instance = Problem {
        m: 3,
        t_end: 3,
        fs: vec![f1, f1, f1],
        beta: 0.4,
    };
    verify_problem(&instance);
    let schedule: Schedule<i32> = vec![2, 3, 1];
    verify_discrete_schedule(&instance, &schedule);
    println!(
        "Cost: {:.1}",
        discrete_objective_function(&instance, &schedule)
    )
}
