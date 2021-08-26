---
layout: primer_without_heading
title: Algorithms for Smoothed Online Convex Optimization
---

# Algorithms for Smoothed Online Convex Optimization

Smoothed Online Convex Optimization (SOCO) is the problem of choosing a sequence of points in some decision space minimizing a hitting cost which is paid for choosing a point and which changes in-between rounds as well as a movement cost that is paid for movement in the decision space.

Thus, SOCO can be understood as online convex optimization with an additional smoothing element.

A special focus of this work is the application to the dynamic right-sizing of data centers.

[**Thesis**](https://jonhue.github.io/soco/main.pdf), [Presentation](https://jonhue.github.io/soco/handout.pdf) [(with animations)](https://jonhue.github.io/soco/slides.pdf), [Documentation](https://jonhue.github.io/soco/doc/soco)

## Acknowledgement

The following is a result of my undergraduate thesis work at [TUM](https://www.tum.de/en/) under the supervision of [Prof. Dr. Susanne Albers](https://www.professoren.tum.de/en/albers-susanne) and advised by [Jens Quedenfeld](http://www14.in.tum.de/personen/quedenfeld/index.html.en).

## Organization

The top-level folders are described as follows:

| folder           | description |
| ---------------- | ----------- |
| `analysis`       | empirical evaluation of the implemented algorithms in the application of dynamically right-sizing data centers |
| `implementation` | implementation of the algorithms |
| `thesis`         | source files of the thesis and the presentation |

### Overview

The implementation can mainly be broken down into three separate parts.

* **Algorithms** - The implementation of various offline and online algorithms for SOCO andrelated problems. [Here is a complete list of the implemented algorithms](https://jonhue.github.io/soco/algorithms).
* **Streaming** - Utilities for streaming the online algorithms in practice. This includes a TCP server that can be queried to run iterations of the online algorithms sequentially.
* **Data Center Model** - For the application of dynamically right-sizing data centers, this implementation includes a comprehensive cost model of data centers.

To achieve optimal performance, everything is implemented in Rust and heavily parallelized. Python bindings are included to interface with the _streaming_ and _data center model_ components.

## Usage

TBD

## Development

### Crate

The Rust crate is contained in the `soco` directory. See the relevant [development](soco#development) section there.

### Analysis

The `analysis` directory contains the case studies with real-world data. It requires the [Python bindings](soco#python-bindings) for the `soco` crate.
See the relevant [prerequisites](analysis#prerequisites) section for more information.
