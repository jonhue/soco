# List of Algorithms

| Name                                                                                         | Variant | Multi-Dimensional? | Integral? | Online? | Approximation/Competitiveness* | Complexity** |
| -------------------------------------------------------------------------------------------- | ------- | ------------------ | --------- | ------- | ------------------------------ | ------------ |
| `offline::uni_dimensional::capacity_provisioning::bcp` [1]                                   | 1       | ❌                 | ❌        | ❌      | optimal                        |              |
| `offline::uni_dimensional::capacity_provisioning::fcp` [*]                                   | 1       | ❌                 | ❌        | ❌      | optimal                        |              |
| `offline::uni_dimensional::optimal_graph_search::optimal_graph_search` [3]                   | 1       | ❌                 | ✅        | ❌      | optimal                        | O(T log m)   |
| `offline::multi_dimensional::optimal_graph_search::optimal_graph_search` [5]                 | 1       | ✅                 | ✅        | ❌      | optimal                        |              |
| `offline::multi_dimensional::approx_graph_search::approx_graph_search` [5]                   | 1       | ✅                 | ✅        | ❌      | (2𝛾 - 1)-approximation         |              |
| `online::uni_dimensional::lazy_capacity_provisioning::fractional::lcp` [1]                   | 1       | ❌                 | ❌        | ✅      | 3-competitive                  |              |
| `online::uni_dimensional::lazy_capacity_provisioning::integral::lcp` [3]                     | 1       | ❌                 | ✅        | ✅      | 3-competitive                  |              |
| `online::uni_dimensional::memoryless::memoryless` [2]                                        | 1       | ❌                 | ❌        | ✅      | 3-competitive                  |              |
| `online::uni_dimensional::probabilistic::probabilistic` [2]                                  | 1       | ❌                 | ❌        | ✅      | 2-competitive                  |              |
| `online::uni_dimensional::randomized::randomized` [3]                                        | 1       | ❌                 | ✅        | ✅      | 2-competitive                  |              |
| `online::multi_dimensional::lazy_budgeting::smoothed_load_optimization::lb` [4]              | 2       | ✅                 | ✅        | ✅      | 2d-competitive                 |              |
| `online::multi_dimensional::lazy_budgeting::smoothed_load_optimization::lb` (randomized) [4] | 2       | ✅                 | ✅        | ✅      | (e / (e - 1))d-competitive     |              |
| `online::multi_dimensional::lazy_budgeting::smoothed_balanced_load_optimization::lb` [5]     | 3       | ✅                 | ✅        | ✅      | (2d + 1 + ε)-competitive       |              |

\* If online, the competitive ratio describes how much worse the algorithm performs compared to an optimal offline algorithm in the worst case.

\*\* If online, complexity is with respect to one iteration of the algorithm.

### Problem Variants

1. Simplified Smoothed Convex Optimization
2. Smoothed Load Optimization
3. Smoothed Balanced-Load Optimization

### Optimal Competitiveness

| Variant | Multi-Dimensional? | Integral? | Deterministic? | Memoryless? | Optimal Competitiveness |
| ------- | ------------------ | --------- | -------------- | ----------- | ----------------------- |
| 1       | ❌                 | ❌        | ✅             | ✅          | 3-competitive           |
| 1       | ❌                 | ❌        | ✅             | ❌          | 2-competitive           |
| 1       | ❌                 | ✅        | ✅             | ❌          | 3-competitive           |
| 1       | ❌                 | ✅        | ❌             | ❌          | 2-competitive           |
| 2, 3    | ✅                 | ✅        | ✅             | ❌          | 2d-competitive          |

### References

1. Minghong Lin and Adam Wierman and Lachlan L. H. Andrew and Eno Thereska. _Dynamic right-sizing for power-proportional data centers_. 2011.
2. Nikhil Bansal and Anupam Gupta and Ravishankar Krishnaswamy and Kirk Pruhs and Kevin Schewior and Cliff Stein. _A 2-Competitive Algorithm For Online Convex Optimization With Switching Costs_. 2015.
3. Susanne Albers and Jens Quedenfeld. _Optimal Algorithms for Right-Sizing Data Centers_. 2018.
4. Susanne Albers and Jens Quedenfeld. _Algorithms for Energy Conservation in Heterogeneous Data Centers_. 2021.
5. Susanne Albers and Jens Quedenfeld. _Algorithms for Right-Sizing Heterogeneous Data Centers_. 2021.
