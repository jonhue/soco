//! Offline and Online Algorithms.
//!
//! # Algorithms
//!
//! ## Offline
//!
//! ### Backward-Recurrent Capacity Provisioning \[1\]
//!
//! SSCO - Fractional - Uni-Dimensional
//!
//! Stays within bounds on the optimal solution moving backwards in time.
//!
//! ### Convex Optimization
//!
//! SCO - Fractional
//!
//! Finds the minimizer of a convex objective.
//!
//! ### Graph-Based Optimal Algorithm (in one dimension) \[5\]
//!
//! SSCO - Integral - Uni-Dimensional
//!
//! Finds a shortest path in a graph using dynamic programming (in polynomial time).
//!
//! ### Graph-Based Optimal Algorithm \[9\]
//!
//! SSCO - Integral
//!
//! Finds a shortest path in a graph using dynamic programming (**not** in polynomial time).
//!
//! ### Graph-Based Polynomial-Time Approximation Scheme \[9\]
//!
//! SSCO - Integral
//!
//! Extends the graph-based optimal algorithm by only considering a subset of the decision space to achieve a better performance.
//!
//! ### Static Fractional Optimum
//!
//! SCO - Fractional
//!
//! Solves a smaller problem than _Convex Optimization_ to obtain an optimal static solution.
//!
//! ### Static Integral Optimum
//!
//! SCO - Integral
//!
//! Cycles through _all_ possible configurations to find the optimal static integral solution. Convexity is used to stop the search of the decision space quickly in practice, however, the worst-case runtime is exponential.
//!
//! ## Online
//!
//! ### Lazy Capacity Provisioning \[1\]
//!
//! SSCO - Fractional - Uni-Dimensional - $3$-competitive
//!
//! Lazily stays within fractional bounds on the decision space.
//!
//! ### Lazy Capacity Provisioning \[5\]
//!
//! SSCO - Integral - Uni-Dimensional - $3$-competitive (optimal for a deterministic algorithm)
//!
//! Lazily stays within integral bounds on the decision space.
//!
//! ### Memoryless algorithm \[3\]
//!
//! SSCO - Fractional - Uni-Dimensional - $3$-competitive (optimal for a memoryless algorithm)
//!
//! Moves towards the minimizer of the hitting cost balancing the paid movement cost. Special case of Primal Online Balanced Descent.
//!
//! ### Probabilistic Algorithm \[3\]
//!
//! SSCO - Fractional - Uni-Dimensional - $2$-competitive (optimal)
//!
//! Constructs a probability distribution of well-performing configurations over time.
//!
//! ### Randmoly Biased Greedy \[4\]
//!
//! SCO - Fractional - Uni-Dimensional - $2$-competitive (optimal)
//!
//! Uses a work function to balance hitting and movement costs.
//!
//! ### Randomized Integral Relaxation \[5\]
//!
//! SSCO - Integral - Uni-Dimensional - $2$-competitive (optimal)
//!
//! Randomly rounds solutions of any $2$-competitive fractional algorithm.
//!
//! ### Lazy Budgeting for SLO \[8\]
//!
//! SLO - Integral - $2d$-competitive (optimal for a deterministic algorithm)
//!
//! Keeps servers active for some minimum time delta even when they are not use to balance hitting costs and movement costs.
//!
//! ### Lazy Budgeting for SLO (randomized) \[8\]
//!
//! SLO - Integral - $(e / (e - 1))d$-competitive
//!
//! Keeps servers active for some minimum time delta even when they are not use to balance hitting costs and movement costs.
//!
//! ### Lazy Budgeting for SBLO \[9\]
//!
//! SBLO - Integral - $2d + 1 + \epsilon$-competitive
//!
//! Keeps servers active for some minimum time delta even when they are not use to balance hitting costs and movement costs.
//!
//! ### Online Gradient Descent \[4\]
//!
//! SCO - Fractional
//!
//! Achieves sublinear regret by _learning_ the static offline optimum.
//!
//! ### Primal Online Balanced Descent \[6\]
//!
//! SCO - Fractional - $3+\mathcal{O}(1/\alpha)$-competitive for the $\ell_2$ norm and locally $\alpha$-polyhedral hitting costs, $\mathcal{O}(\sqrt{d})$-competitive for the $\ell_1$ norm; mirror map must be $m$-strongly convex and $M$-Lipschitz smooth in the switching cost norm
//!
//! Takes a gradient step orthogonal to some landing level set balancing costs in the primal space.
//!
//! ### Dual Online Balanced Descent \[6\]
//!
//! SCO - Fractional - mirror map must be $m$-strongly convex and $M$-Lipschitz smooth in the switching cost norm
//!
//! Takes a gradient step orthogonal to some landing level set balancing costs in the dual space. Achieves sublinear regret.
//!
//! ### Greedy Online Balanced Descent \[7\]
//!
//! SCO - Fractional - $\mathcal{O}(1 / \sqrt{m})$-competitive for $m$-quasiconvex hitting costs and $\ell_2$-squared switching costs
//!
//! Takes a normal OBD-step and then an additional step directly towards the minimizer of the hitting cost depending on the convexity parameter $m$.
//!
//! ### Regularized Online Balanced Descent \[7\]
//!
//! SCO - Fractional - $\mathcal{O}(1 / \sqrt{m})$-competitive (optimal) for $m$-strongly convex and differentiable hitting costs and switching costs modeled as the Bregman divergence where the potential function is $\alpha$-strongly convex, $\beta$-strongly smooth, differentiable, and its Fenchel Conjugate is differentiable; $\Omega(1/m)$-competitive for $m$-quasiconvex hitting costs and $\ell_2$-squared switching costs
//!
//! Using a computationally simpler local view.
//!
//! ### Receding Horizon Control \[2\]
//!
//! SSCO - Fractional - $(1 + \Omega(\beta/e_0))$-competitive where $e_0$ is the idle cost and $\beta$ is the scaling of the Manhattan distance; when uni-dimensional the competitive ratio is $1 + \mathcal{O}(1/w)$
//!
//! ### Averaging Fixed Horizon Control \[2\]
//!
//! SSCO - Fractional - $1 + \mathcal{O}(1/w)$-competitive
//!
//! # References
//!
//! 1. Minghong Lin and Adam Wierman and Lachlan L. H. Andrew and Eno Thereska. _Dynamic right-sizing for power-proportional data centers_. 2011.
//! 2. Minghong Lin and Zhenhua Liu and Adam Wierman and Lachlan L. H. Andrew. _Online Algorithms for Geographical Load Balancing_. 2012.
//! 3. Nikhil Bansal and Anupam Gupta and Ravishankar Krishnaswamy and Kirk Pruhs and Kevin Schewior and Cliff Stein. _A 2-Competitive Algorithm For Online Convex Optimization With Switching Costs_. 2015.
//! 4. Lachlan L. H. Andrew and Siddharth Barman and Katrina Ligett and Minghong Lin and Adam Myerson and Alan Roytman and Adam Wierman. _A Tale of Two Metrics: Simultaneous Bounds on Competitiveness and Regret_. 2015.
//! 5. Susanne Albers and Jens Quedenfeld. _Optimal Algorithms for Right-Sizing Data Centers_. 2018.
//! 6. Niangjun Chen and Gautam Goel and Adam Wierman. _Smoothed Online Convex Optimization in High Dimensions via Online Balanced Descent_. 2018.
//! 7. Gautam Goel and Yiheng Lin and Haoyuan Sun and Adam Wierman. _Beyond Online Balanced Descent: An Optimal Algorithm for Smoothed Online Optimization_. 2019.
//! 8. Susanne Albers and Jens Quedenfeld. _Algorithms for Energy Conservation in Heterogeneous Data Centers_. 2021.
//! 9. Susanne Albers and Jens Quedenfeld. _Algorithms for Right-Sizing Heterogeneous Data Centers_. 2021.

use crate::{
    model::{ModelOutputFailure, ModelOutputSuccess},
    problem::{DefaultGivenProblem, Problem},
};

mod capacity_provisioning;

pub mod offline;
pub mod online;

/// Options of an algorithm.
pub trait Options<T, P, C, D>:
    Clone + DefaultGivenProblem<T, P, C, D> + Send
where
    P: Problem<T, C, D>,
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
}
impl<T, P, C, D, O> Options<T, P, C, D> for O
where
    O: Clone + DefaultGivenProblem<T, P, C, D> + Send,
    P: Problem<T, C, D>,
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
}
