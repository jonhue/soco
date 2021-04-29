# List of Algorithms

| Name                                             | Variant | Heterogeneous? | Discrete? | Online? | Competitiveness* | Complexity** |
| ------------------------------------------------ | ------- | -------------- | --------- | ------- | ---------------- | ------------ |
| [`offline::iopt`](offline/iopt.rs) [3]           | 1       | ❌             | ✅        | ❌      |                  | O(T log m)   |
| [`lcp::ilcp`](lcp/ilcp.rs) [3]                   | 1       | ❌             | ✅        | ✅      | 3-competitive    | ?            |
| [`lcp::elcp`](lcp/elcp.rs) [*]                   | 1       | ❌             | ❌        | ✅      | 3-competitive    | ?            |
| [`bansal::memoryless`](bansal/memoryless.rs) [2] | 1       | ❌             | ❌        | ✅      | 3-competitive    | ?            |
| [`bansal::det`](bansal/det.rs) [2]               | 1       | ❌             | ❌        | ✅      | 2-competitive    | ?            |
| [`bansal::irand`](bansal/irand.rs) [3]           | 1       | ❌             | ✅        | ✅      | 2-competitive    | ?            |

\* If online, the competitive ratio describes how much worse the algorithm performs compared to an optimal offline algorithm in the worst case.

\*\* If online, complexity is with respect to one iteration of the algorithm.

### Problem Variants

1. The most relaxed version of the problem as used by [2] and [3]. Here, the convex cost functions `f_t` arrive over time.
2. A more restricted version of the problem used by [1] where workloads arrive over time and the convex cost function `f` remains fixed.
3. Adding a prediction window of length `w` where `w` is constant in independent of `T`. Described by [1] and [3].

### Optimal Competitiveness

| Variant | Heterogeneous? | Discrete? | Deterministic? | Memoryless? | Optimal Competitiveness |
| ------- | -------------- | --------- | -------------- | ----------- | ----------------------- |
| 1       | ❌             | ❌        | ✅             | ✅          | 3-competitive           |
| 1, 2, 3 | ❌             | ❌        | ✅             | ❌          | 2-competitive           |
| 1, 2, 3 | ❌             | ✅        | ✅             | ❌          | 3-competitive           |

### References

1. Minghong Lin and Adam Wierman and Lachlan L. H. Andrew and Eno Thereska. _Dynamic right-sizing for power-proportional data centers_. 2011.
2. Nikhil Bansal and Anupam Gupta and Ravishankar Krishnaswamy and Kirk Pruhs and Kevin Schewior and Cliff Stein. _A 2-Competitive Algorithm For Online Convex Optimization With Switching Costs_. 2015.
3. Susanne Albers and Jens Quedenfeld. _Optimal Algorithms for Right-Sizing Data Centers_. 2018.
