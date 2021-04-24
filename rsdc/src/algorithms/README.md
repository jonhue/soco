# List of Algorithms

| Name                                                                           | Abbrev. | Discrete? | Online? | Competitiveness | Complexity* |
| ------------------------------------------------------------------------------ | -------- | --------- | ------- | --------------- | ----------- |
| [Optimal Discrete Deterministic Polynomial-Time Offline Algorithm](opt.rs) [1] | `iopt`   | ✅        | ❌      | optimal         | O(T log m)  |
| [Discrete Lazy Capacity Provisioning](lcp.rs) [1]                              | `ilcp`   | ✅        | ✅      | 3-competitive   | ?           |
| [Lazy Capacity Provisioning](lcp.rs) [2]                                       | `lcp`    | ❌        | ✅      | 3-competitive   | ?           |

\* If online, complexity is with respect to one iteration of the algorithm.

### References

1. Susanne Albers and Jens Quedenfeld. _Optimal Algorithms for Right-Sizing Data Centers_. 2018.
2. Minghong Lin and Adam Wierman and Lachlan L. H. Andrew and Eno Thereska. _Dynamic right-sizing for power-proportional data centers_. 2011.
