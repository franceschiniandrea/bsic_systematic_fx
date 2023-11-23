import os

import numpy as np
import pandas as pd


def process_dates(df: pd.DataFrame, hour: int):
    df.index = pd.to_datetime(df.index, utc=True)
    df.index = df.index.map(lambda x: x.replace(hour=hour))


def load_data(data_dir: str = "data"):
    # FX Fixes
    fx_lon_fix = pd.read_excel(
        data_dir + "/fx_fixes.xlsx", sheet_name="LON_Fix", index_col=0, parse_dates=True
    )
    fx_ny_fix = pd.read_excel(
        data_dir + "/fx_fixes.xlsx", sheet_name="NY_Fix", index_col=0, parse_dates=True
    )

    fx_lon_fix.columns = [col.split()[0] for col in fx_lon_fix.columns]
    fx_ny_fix.columns = [col.split()[0] for col in fx_ny_fix.columns]
    process_dates(fx_lon_fix, 16)
    process_dates(fx_ny_fix, 22)

    fx_fixes = [fx_lon_fix, fx_ny_fix]
    fx_fixes = pd.concat(fx_fixes)

    def convert_fx(df: pd.DataFrame):
        new_df = df.copy()
        for col in df.columns:
            if col[:3] == "USD":
                new_df[col[3:] + "USD"] = 1 / df[col]
                new_df.drop(col, axis=1, inplace=True)

        return new_df

    fx_fixes = convert_fx(fx_fixes)
    fx_fixes.sort_index(inplace=True)

    # Swaps Fixes
    countries_to_fx = {
        "AD": "AUDUSD",
        "CD": "CADUSD",
        "SF": "CHFUSD",
        "DK": "DKKUSD",
        "EU": "EURUSD",
        "BP": "GBPUSD",
        "JY": "JPYUSD",
        "NK": "NOKUSD",
        "ND": "NZDUSD",
        "SK": "SEKUSD",
        "US": "USD",
    }
    swaps_lon_fix = pd.read_excel(
        data_dir + "/swaps_fixes.xlsx", sheet_name="LON_Fix", index_col=0
    )
    swaps_ny_fix = pd.read_excel(
        data_dir + "/swaps_fixes.xlsx", sheet_name="NY_Fix", index_col=0
    )
    swaps_lon_fix.columns = [
        countries_to_fx[col.split()[0][:2]] for col in swaps_lon_fix.columns
    ]
    swaps_ny_fix.columns = [
        countries_to_fx[col.split()[0][:2]] for col in swaps_ny_fix.columns
    ]

    process_dates(swaps_lon_fix, 16)
    process_dates(swaps_ny_fix, 22)
    swaps_fixes = [swaps_lon_fix, swaps_ny_fix]
    swaps_fixes = pd.concat(swaps_fixes)
    swaps_fixes.sort_index(inplace=True)
    us_swaps = swaps_fixes["USD"]

    swaps_fixes = swaps_fixes.reindex(fx_fixes.columns, axis=1)
    swaps_fixes["USD"] = us_swaps

    cpi_data = pd.read_excel(data_dir + "/fmt_cpi_data.xlsx", index_col=0)
    cpi_data.ffill(inplace=True)
    cpi_est_data = cpi_data.rolling(12).mean()
    process_dates(cpi_est_data, 16)

    def map_cols(col: str):
        if col == "USD":
            return col

        return col + "USD"

    cpi_est_data.columns = map(map_cols, list(cpi_est_data.columns))  # type: ignore
    cpi_est_data = cpi_est_data.reindex(swaps_fixes.columns, axis=1)
    cpi_est_data = cpi_est_data.reindex(swaps_fixes.index, axis=0, method="ffill")
    cpi_est_data.sort_index(inplace=True)

    real_swaps_rates = swaps_fixes - cpi_est_data

    return fx_fixes, swaps_fixes, cpi_data, real_swaps_rates


def load_em_data(data_path="data"):
    fx_fixes = pd.read_excel(
        "../data/em_data.xlsx", sheet_name="fx_fixes", index_col=0, parse_dates=True
    )
    swaps_fixes = pd.read_excel(
        "../data/em_data.xlsx", sheet_name="swaps_fixes", index_col=0, parse_dates=True
    )

    _, swapsg10, _, _ = load_data(data_path)
    usswaps = swapsg10["USD"]
    usswaps = swaps_fixes[swaps_fixes.index.hour == 16]["USD"]
    usswaps.index = usswaps.index.map(lambda x: x.replace(hour=0))
    usswaps.index = usswaps.index.map(lambda x: x.tz_localize(None))

    fx_fixes.sort_index(inplace=True)
    swaps_fixes.sort_index(inplace=True)

    fx_fixes = fx_fixes.merge(usswaps, left_index=True, right_index=True)

    return fx_fixes, swaps_fixes


# _, swaps_fixes, cpi_data, real_swaps_rates = load_data()

# print("swaps")
# print(swaps_fixes)
# print("CPI")
# print(cpi_data)
# print(real_swaps_rates)
