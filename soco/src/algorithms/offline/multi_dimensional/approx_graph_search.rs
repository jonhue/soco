use crate::algorithms::graph_search::Path;
use crate::algorithms::offline::multi_dimensional::graph_search::{
    graph_search, Values,
};
use crate::algorithms::offline::OfflineOptions;
use crate::problem::IntegralSimplifiedSmoothedConvexOptimization;
use crate::result::Result;

static DEFAULT_GAMMA: f64 = 1.1;

pub struct Options {
    /// `gamma > 1`. Default is `1.1`.
    pub gamma: Option<f64>,
}

/// Graph-Based Approximation Algorithm
pub fn approx_graph_search<'a>(
    p: &'a IntegralSimplifiedSmoothedConvexOptimization<'a>,
    options: &Options,
    offline_options: &OfflineOptions,
) -> Result<Path> {
    let gamma = options.gamma.unwrap_or(DEFAULT_GAMMA);
    let all_values = build_all_values(p, gamma);
    graph_search(p, all_values, offline_options)
}

/// Computes all values allowed by the approximation algorithm.
fn build_all_values(
    p: &IntegralSimplifiedSmoothedConvexOptimization<'_>,
    gamma: f64,
) -> Values {
    let mut all_values = vec![];
    for k in 0..p.d {
        all_values.push(build_values(p.bounds[k as usize], gamma));
    }
    all_values
}

fn build_values(bound: i32, gamma: f64) -> Vec<i32> {
    if bound as f64 >= gamma.powi(bound) && gamma < 2. {
        build_values_via_log(bound, gamma)
    } else {
        build_values_via_exp(bound, gamma)
    }
}

fn build_values_via_exp(bound: i32, gamma: f64) -> Vec<i32> {
    let mut vs: Vec<i32> = vec![0, 1];

    let mut i = 1;
    loop {
        let l = gamma.powi(i).floor() as i32;
        if l > bound {
            break;
        }
        if !vs.contains(&l) {
            vs.push(l);
        }

        let u = gamma.powi(i).ceil() as i32;
        if u > bound {
            break;
        }
        if !vs.contains(&u) {
            vs.push(u);
        }

        i += 1;
    }
    if !vs.contains(&bound) {
        vs.push(bound);
    }

    vs
}

fn build_values_via_log(bound: i32, gamma: f64) -> Vec<i32> {
    let mut vs: Vec<i32> = vec![0, 1];

    for j in 2..=bound {
        let l = (j as f64 - 1.).log(gamma);
        let u = (j as f64 + 1.).log(gamma);
        if l.floor() as i32 != u.floor() as i32
            || l.ceil() as i32 != u.ceil() as i32
        {
            vs.push(j);
        }
    }
    if !vs.contains(&bound) {
        vs.push(bound);
    }

    vs
}

#[cfg(test)]
mod tests {
    use super::{build_values_via_exp, build_values_via_log};

    #[test]
    fn test_build_values() {
        let bounds = vec![1, 2, 5, 10, 100, 100_000];
        let gammas = vec![
            1.005, 1.1, 1.2, 1.3, 1.4, 1.45, 1.5, 1.6, 1.7, 1.8, 1.9, 1.99,
        ];

        for bound in bounds {
            for &gamma in gammas.iter() {
                assert_eq!(
                    build_values_via_exp(bound, gamma),
                    build_values_via_log(bound, gamma),
                    "bound={}, gamma={}",
                    bound,
                    gamma
                );
            }
        }
    }
}
