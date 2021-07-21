from typing import List
from dataclasses import dataclass
from scipy.stats import median_abs_deviation, mode
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import tikzplotlib
import pandas as pd


@dataclass
class TimeDelta:
    d: float
    h: float
    m: float
    s: float


def timedelta(seconds: int) -> TimeDelta:
    d, r = divmod(seconds, 24 * 60 * 60)
    h, r = divmod(r, 60 * 60)
    m, r = divmod(r, 60)
    return TimeDelta(d, h, m, r)


@dataclass
class DistributionSummary:
    mean: float
    median: float
    modes: List[float]
    std: float
    mad: float
    min_: float
    p25: float
    p75: float
    max_: float


def summarize_distribution(series: np.array) -> DistributionSummary:
    return DistributionSummary(
        series.mean(),
        np.quantile(series, 0.5),
        mode(series).mode,
        series.std(),
        median_abs_deviation(series),
        series.min(),
        np.quantile(series, 0.25),
        np.quantile(series, 0.75),
        series.max(),
    )


def distance_distribution(series: np.array) -> np.array:
    result = series - np.roll(series, 1)
    result[0] = series[0]
    return result


def plot_cdf(samples: np.array, label: str, name: str):
    sb.ecdfplot(samples)
    plt.xlabel(label)
    plt.ylabel("proportion")
    tikzplotlib.save(f"out/figures/{name}.tex")


def plot(x: np.array, y: np.array, xlabel: str, ylabel: str, name: str):
    df = pd.DataFrame({"x": x, "y": y})
    sb.lineplot(x="x", y="y", data=df)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    tikzplotlib.save(f"out/figures/{name}.tex")
