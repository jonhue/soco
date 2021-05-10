# List of Algorithms

| Name                                             | Variant | Multi-Dimensional? | Discrete? | Online? | Competitiveness* | Complexity** |
| -----------------------------------------------  | ------- | ------------------ | --------- | ------- | ---------------- | ------------ |
| [`offline::opt_backward`](offline/opt.rs) [1]    | 1       | ❌                 | ❌        | ❌      |                  | ?            |
| [`offline::opt_forward`](offline/opt.rs) [*]     | 1       | ❌                 | ❌        | ❌      |                  | ?            |
| [`offline::iopt`](offline/iopt.rs) [3]           | 1       | ❌                 | ✅        | ❌      |                  | O(T log m)   |
| [`lcp::ilcp`](lcp/ilcp.rs) [3]                   | 1       | ❌                 | ✅        | ✅      | 3-competitive    | ?            |
| [`lcp::lcp`](lcp/lcp.rs) [1]                     | 1       | ❌                 | ❌        | ✅      | 3-competitive    | ?            |
| [`bansal::memoryless`](bansal/memoryless.rs) [2] | 1       | ❌                 | ❌        | ✅      | 3-competitive    | ?            |
| [`bansal::det`](bansal/det.rs) [2]               | 1       | ❌                 | ❌        | ✅      | 2-competitive    | ?            |
| [`bansal::irand`](bansal/irand.rs) [3]           | 1       | ❌                 | ✅        | ✅      | 2-competitive    | ?            |
| [`quedenfeld::idet`](quedenfeld/idet.rs) [4]     | 2       | ✅                 | ✅        | ✅      | 2d-competitive   | ?            |

\* If online, the competitive ratio describes how much worse the algorithm performs compared to an optimal offline algorithm in the worst case.

\*\* If online, complexity is with respect to one iteration of the algorithm.

### Problem Variants

1. Smoothed optimization with convex cost functions.
2. Smoothed optimization with dimension-dependent constant cost functions.

### Optimal Competitiveness

| Variant | Multi-Dimensional? | Discrete? | Deterministic? | Memoryless? | Optimal Competitiveness |
| ------- | ------------------ | --------- | -------------- | ----------- | ----------------------- |
| 1       | ❌                 | ❌        | ✅             | ✅          | 3-competitive           |
| 1       | ❌                 | ❌        | ✅             | ❌          | 2-competitive           |
| 1       | ❌                 | ✅        | ✅             | ❌          | 3-competitive           |
| 2       | ✅                 | ✅        | ✅             | ❌          | 2d-competitive          |

### References

1. Minghong Lin and Adam Wierman and Lachlan L. H. Andrew and Eno Thereska. _Dynamic right-sizing for power-proportional data centers_. 2011.
2. Nikhil Bansal and Anupam Gupta and Ravishankar Krishnaswamy and Kirk Pruhs and Kevin Schewior and Cliff Stein. _A 2-Competitive Algorithm For Online Convex Optimization With Switching Costs_. 2015.
3. Susanne Albers and Jens Quedenfeld. _Optimal Algorithms for Right-Sizing Data Centers_. 2018.
4. Susanne Albers and Jens Quedenfeld. _Algorithms for Energy Conservation in Heterogeneous Data Centers_. 2019.
