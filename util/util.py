import pandas as pd
import numpy as np
from typing import Tuple


def data_clean_records(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocessing data

    :param data: data to be cleaned
    :type data: pd.DataFrame
    :return: cleaned ddata
    :rtype: pd.DataFrame
    """

    data = data.dropna()

    # From the paper. This makes sense as the deep in/out-of-money option delta
    # are much less relevant to our model (movements in price do not meaningfully change
    # the delta)
    data = data[(data["delta"] >= 0.05) & (data["delta"] <= 0.95)]

    return data


def split_train_test(
    data: pd.DataFrame, train_pct: float, seed: int = 1
) -> Tuple(pd.DataFrame):
    """Split data by optionid into training and testing sets

    :param data: full data set
    :type data: pd.DataFrame
    :param train_pct: percentage of data to be using in training set
    :type train_pct: float
    :param seed: seed for random sampling, defaults to 1
    :type seed: int, optional
    :return: Tuple(pd.DataFrame)
    :rtype: The traiing and testing dataset, stored in a tuple
    """

    # Choose the traninig set
    n = len(data["optionid"].unique())
    np.random.seed(seed)
    n_train = int(train_pct * n)
    train_id = np.random.choice(data["optionid"].unique(), n_train, replace=False)

    # Create output
    train_mask = data.optionid.isin(train_id)
    train = data[train_mask].reset_index(drop=True)
    test_mask = [mask is False for mask in train_mask]
    test = data[test_mask].reset_index(drop=True)

    return train, test
