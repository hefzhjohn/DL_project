import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf


def get_spx(sd, ed) -> pd.DataFrame:

    # Retrieve data
    sp500 = yf.download("^GSPC", sd, ed)

    # Create columns
    sp500["date"] = sp500.index
    sp500 = sp500[["date", "Adj Close"]]
    sp500.columns = ["date", "sp500"]
    sp500["sp500_return"] = sp500["sp500"] / sp500["sp500"].shift(1) - 1
    sp500["sp500_log_return"] = np.log(sp500["sp500"] / sp500["sp500"].shift(1))

    # Remove NA and create final columns
    sp500 = sp500.iloc[1:]
    sp500 = sp500[["date", "sp500", "sp500_log_return"]]
    sp500.columns = ["date", "spot", "spot_return"]

    return sp500


def get_vix(sd, ed) -> pd.DataFrame:

    vix = yf.download("^VIX", sd, ed)

    # create columns
    vix["date"] = vix.index
    vix = vix[["date", "Adj Close"]]
    vix.columns = ["date", "VIX"]

    return vix


def process_raw_data(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocessing raw data to experiment data"""

    # select useful data from raw data
    data = data[
        [
            "optionid",
            "date",
            "exdate",
            "cp_flag",
            "strike_price",
            "best_bid",
            "best_offer",
            "volume",
            "open_interest",
            "delta",
            "gamma",
            "vega",
            "theta",
            "impl_volatility",
        ]
    ]

    # construct additional information from existing columns in the dataset
    data.date = pd.to_datetime(data["date"], format="%Y%m%d")
    data.exdate = pd.to_datetime(data["exdate"], format="%Y%m%d")
    data = data.sort_values(by=["optionid", "date"])
    data["time_to_maturity"] = (data.exdate - data.date).dt.days
    data["ttm_yr"] = data.time_to_maturity / 252
    data["strike_price"] = data.strike_price / 1000
    data["impvol_chg"] = data.groupby(["optionid"])["impl_volatility"].diff()

    # get market reference data
    sd = data.date.iloc[0] - dt.timedelta(days=1)
    ed = data.date.iloc[-1] + dt.timedelta(days=1)
    sp500 = get_spx(sd, ed)
    vix = get_vix(sd, ed)
    data = data.merge(sp500, on="date")
    data = data.merge(vix, on="date")

    # drop incomplete data
    data = data.dropna().reset_index(drop=True)

    return data
