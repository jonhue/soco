from typing import List
import pandas as pd
import numpy as np
import json


SECONDS_PER_DAY = 60 * 60 * 24


def parse_loads(trace: str) -> List[List[int]]:
    """
    Parses prepared loads for a trace.
    Returns: Time > Job Type > Load
    """
    path = f"out/loads/{trace}.csv"
    df = pd.read_csv(path)
    return df.to_numpy().tolist()


def convert_offline_to_online_input(
    loads: List[List[int]],
) -> List[List[List[List[int]]]]:
    """
    Converts some offline input to online input.
    Returns: Arrival Time > Prediction Time > Job Type > List of Samples
    """
    return [[[load_profile]] for load_profile in loads]


def select_load_from_nth_last_day(
    loads: List[List[int]], time_slot_length: int, n: int
) -> List[List[int]]:
    """
    Selects all load profiles for time slots belonging to the nth last day.
    """
    time_slots_per_day = int(SECONDS_PER_DAY / time_slot_length)
    a = -(time_slots_per_day * n)
    b = -(time_slots_per_day * (n - 1)) - 1
    print(a, b)
    result = loads[a:b]
    result.append(loads[len(loads) + b])
    return result


def perfect_load_prediction(
    loads: List[List[int]],
) -> List[List[List[List[int]]]]:
    """
    Converts some offline input to (perfect knowledge) online input.
    Returns: Arrival Time > Prediction Time > Job Type > List of Samples
    """
    return [
        [[load_profile] for load_profile in loads[i:] + [[0] * len(loads[0])] * 24]
        for i in range(len(loads))
    ]


def predict_loads(
    trace: str,
    loads: List[List[int]],
) -> List[List[List[List[int]]]]:
    """
    Predict loads for a trace with the training data up to some time.
    Returns: Arrival Time > Prediction Time > Job Type > List of Samples
    """
    path = f"out/loads/{trace}.json"
    with open(path, "r") as f:
        values = np.array(list(json.load(f).values()))
        predictions = np.array(values)
    tmp = np.mean(np.swapaxes(predictions, 0, 1), axis=2)
    return [
        [[loads[i]]]
        + [
            [load_profile]
            for load_profile in list(tmp[i + 1 :]) + [[0] * len(loads[0])] * 24
        ]
        for i in range(len(loads))
    ]
