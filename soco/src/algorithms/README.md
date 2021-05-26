# List of Algorithms

| Name                                                                                         | Variant | Multi-Dimensional? | Integral? | Online? | Approximation/Competitiveness*   | Complexity** |
| -------------------------------------------------------------------------------------------- | ------- | ------------------ | --------- | ------- | -------------------------------- | ------------ |
| `offline::uni_dimensional::capacity_provisioning::bcp` [1]                                   | 2       | ❌                 | ❌        | ❌      | optimal                          |              |
| `offline::uni_dimensional::capacity_provisioning::fcp` [*]                                   | 2       | ❌                 | ✅        | ❌      | optimal                          | O(T log m)   |
| `offline::multi_dimensional::optimal_graph_search::optimal_graph_search` [7]                 | 2       | ✅                 | ✅        | ❌      | optimal                          |              |
| `offline::multi_dimensional::approx_graph_search::approx_graph_search` [7]                   | 2       | ✅                 | ✅        | ❌      | (2𝛾 - 1)-approximation           |              |
| `online::uni_dimensional::lazy_capacity_provisioning::fractional::lcp` [1]                   | 2       | ❌                 | ❌        | ✅      | 3-competitive                    |              |
| `online::uni_dimensional::lazy_capacity_provisioning::integral::lcp` [3]                     | 2       | ❌                 | ✅        | ✅      | 3-competitive                    |              |
| `online::uni_dimensional::memoryless::memoryless` [2]                                        | 2       | ❌                 | ❌        | ✅      | 3-competitive                    |              |
| `online::uni_dimensional::probabilistic::probabilistic` [2]                                  | 2       | ❌                 | ❌        | ✅      | 2-competitive                    |              |
| `online::uni_dimensional::randomized::randomized` [3]                                        | 2       | ❌                 | ✅        | ✅      | 2-competitive                    |              |
| `online::multi_dimensional::lazy_budgeting::smoothed_load_optimization::lb` [6]              | 3       | ✅                 | ✅        | ✅      | 2d-competitive                   |              |
| `online::multi_dimensional::lazy_budgeting::smoothed_load_optimization::lb` (randomized) [6] | 3       | ✅                 | ✅        | ✅      | (e / (e - 1))d-competitive       |              |
| `online::multi_dimensional::lazy_budgeting::smoothed_balanced_load_optimization::lb` [7]     | 4       | ✅                 | ✅        | ✅      | (2d + 1 + ε)-competitive         |              |
| `online::multi_dimensional::online_balanced_descent::meta::obd` (meta algorithm) [4]         | 1       | ✅                 | ❌        | ✅      |                                  |              |
| `online::multi_dimensional::online_balanced_descent::primal::pobd` [4]                       | 1       | ✅                 | ❌        | ✅      | O(sqrt(d))-competitive***        |              |
| `online::multi_dimensional::online_balanced_descent::dual::dobd` [4]                         | 1       | ✅                 | ❌        | ✅      |                                  |              |
| `online::multi_dimensional::online_balanced_descent::greedy::gobd` [5]                       | 1       | ✅                 | ❌        | ✅      | O(1/sqrt(m))-competitive\*\*\*\* |              |
| `online::multi_dimensional::online_balanced_descent::regularized::robd` [5]                  | 1       | ✅                 | ❌        | ✅      | O(1/sqrt(m))-competitive\*\*\*\* |              |

\* If online, the competitive ratio describes how much worse the algorithm performs compared to an optimal offline algorithm in the worst case.

\*\* If online, complexity is with respect to one iteration of the algorithm.

\*\*\* For the l1-norm, constant competitiveness for the l2-norm.

\*\*\*\* hitting cost is `m`-strongly convex

### Problem Variants

1. Smoothed Convex Optimization
2. Simplified Smoothed Convex Optimization
3. Smoothed Load Optimization
4. Smoothed Balanced-Load Optimization

Above order is from most general to most specific.

### Optimal Competitiveness

| Variant | Multi-Dimensional? | Integral? | Deterministic? | Memoryless? | Optimal Competitiveness                         |
| ------- | ------------------ | --------- | -------------- | ----------- | ----------------------------------------------- |
| 1       | ❌                 | ❌        | ✅             | ✅          | 3-competitive                                   |
| 1       | ❌                 | ❌        | ✅             | ❌          | 2-competitive                                   |
| 1       | ❌                 | ✅        | ✅             | ❌          | 3-competitive                                   |
| 1       | ❌                 | ✅        | ❌             | ❌          | 2-competitive                                   |
| 2, 3    | ✅                 | ✅        | ✅             | ❌          | 2d-competitive                                  |
| 1       | ✅                 | ❌        | ❌             | ❌          | O(1/sqrt(m))-competitive, for m to 0^+ \*\*\*\* |

### References

1. Minghong Lin and Adam Wierman and Lachlan L. H. Andrew and Eno Thereska. _Dynamic right-sizing for power-proportional data centers_. 2011.
2. Nikhil Bansal and Anupam Gupta and Ravishankar Krishnaswamy and Kirk Pruhs and Kevin Schewior and Cliff Stein. _A 2-Competitive Algorithm For Online Convex Optimization With Switching Costs_. 2015.
3. Susanne Albers and Jens Quedenfeld. _Optimal Algorithms for Right-Sizing Data Centers_. 2018.
4. Niangjun Chen and Gautam Goel and Adam Wierman. _Smoothed Online Convex Optimization in High Dimensions via Online Balanced Descent_. 2018.
5. Gautam Goel and Yiheng Lin and Haoyuan Sun and Adam Wierman. _Beyond Online Balanced Descent: An Optimal Algorithm for Smoothed Online Optimization_. 2019.
6. Susanne Albers and Jens Quedenfeld. _Algorithms for Energy Conservation in Heterogeneous Data Centers_. 2021.
7. Susanne Albers and Jens Quedenfeld. _Algorithms for Right-Sizing Heterogeneous Data Centers_. 2021.
