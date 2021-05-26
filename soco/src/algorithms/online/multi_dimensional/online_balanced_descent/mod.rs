//! Online Balanced Descent

/// Maximum number of bisection iterations.
static MAX_ITERATIONS: usize = 1_000;

/// Maximum `l` for bisection is calculated as `MAX_L_FACTOR * min_hitting_cost`.
static MAX_L_FACTOR: f64 = 1_000.;

pub mod dual_online_balanced_descent;
pub mod mirror_map;
pub mod online_balanced_descent;
pub mod primal_online_balanced_descent;
