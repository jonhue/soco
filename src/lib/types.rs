// Homogeneous Data-Center Right-Sizing problem.
pub struct HomProblem<'a, T> {
    // Number of servers.
    pub m: i32,
    // Finite time horizon.
    pub t_end: i32,
    // Positive real constant resembling the switching cost.
    pub beta: f64,
    // Non-negative convex cost functions.
    // Must be total on 1<=t<=T, 0<=x_t<=m; may return `None` otherwise.
    pub f: Box<dyn Fn(i32, T) -> Option<f64> + 'a>,
}
pub type DiscreteHomProblem<'a> = HomProblem<'a, i32>;
#[allow(dead_code)]
pub type ContinuousHomProblem<'a> = HomProblem<'a, f64>;

// Result of the Homogeneous Data-Center Right-Sizing problem.
// Number of active servers from time 1 to time T.
pub type Schedule<T> = Vec<T>;
pub type DiscreteSchedule = Schedule<i32>;
#[allow(dead_code)]
pub type ContinuousSchedule = Schedule<f64>;
