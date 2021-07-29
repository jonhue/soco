from typing import List
import pandas as pd
import json


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
    Converts some offline input to (perfect knowledge) online input.
    Returns: Arrival Time > Prediction Time > Job Type > List of Samples
    """
    return [[[load_profile] for load_profile in loads] for _ in range(len(loads))]


def predict_loads(trace: str, t: int) -> List[List[List[int]]]:
    """
    Predict loads for a trace with the data up to some time `t`.
    Returns: Time > Job Type > List of Samples
    """
    path = f"out/loads/{trace}.json"
    with open(path, "r") as f:
        predictions = json.load(f).values()
        print(predictions[0])
