# List of Algorithms

| No. | Name   | Discrete? | Online? | Competitiveness | Complexity (per iteration) |
| --- | ------ | --------- | ------- | --------------- | -------------------------- |
| 1   | `iopt` | ✅        | ❌      | optimal         | O(T log m)                 |
| 2   | `ilcp` | ✅        | ✅      | 3-competitive   | ?                          |
| 3   | `lcp`  | ❌        | ✅      | 3-competitive   | ?                          |

1. [Optimal Discrete Deterministic Polynomial-Time Offline Algorithm](opt.rs) [1]
2. [Discrete Lazy Capacity Provisioning](lcp.rs) [1]
3. [Lazy Capacity Provisioning](lcp.rs) [2]

References:

1. Susanne Albers and Jens Quedenfeld. Optimal Algorithms for Right-Sizing Data Centers. 2018.
2. Minghong Lin and Adam Wierman and Lachlan L. H. Andrew and Eno Thereska. Dynamic right-sizing for power-proportional data centers. 2011.
