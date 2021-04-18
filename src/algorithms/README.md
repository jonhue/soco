# List of Algorithms

1. [Optimal Polynomial-Time Offline Algorithm computing an Integral Solution](alg1.rs) [1]
2. [Discrete Lazy Capacity Provisioning](alg2.rs) [1]

Overview:

| No. | Name   | Discrete? | Online? | Competitiveness | Complexity (per iteration) |
| --- | ------ | --------- | ------- | --------------- | -------------------------- |
| 1   | `dopt` | ✅        | ❌      | optimal         | O(T log m)                 |
| 2   | `lcp`  | ✅        | ✅      | 3-competitive   | ?                          |

References:

1. Susanne Albers and Jens Quedenfeld. Optimal Algorithms for Right-Sizing Data Centers. 2018.
