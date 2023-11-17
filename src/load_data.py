import os

import numpy as np
import pandas as pd


def load_data():
    print(os.getcwd())
    # FX Fixes
    fx_lon_fix = pd.read_excel(
        "data/fx_fixes.xlsx", sheet_name="LON_Fix", index_col=0, parse_dates=True
    )
    fx_ny_fix = pd.read_excel(
        "data/fx_fixes.xlsx", sheet_name="NY_Fix", index_col=0, parse_dates=True
    )
    fx_lon_fix.columns = [col.split()[0] for col in fx_lon_fix.columns]
    fx_ny_fix.columns = [col.split()[0] for col in fx_ny_fix.columns]

    def convert_fx(df: pd.DataFrame):
        new_df = df.copy()
        for col in df.columns:
            if col[:3] == "USD":
                new_df[col[3:] + "USD"] = 1 / df[col]
                new_df.drop(col, axis=1, inplace=True)

        return new_df

    fx_lon_fix = convert_fx(fx_lon_fix)
    fx_ny_fix = convert_fx(fx_ny_fix)
    fx_lon_fix.head()
    fx_ny_fix.head()

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
        "data/swaps_fixes.xlsx", sheet_name="LON_Fix", index_col=0, parse_dates=True
    )
    swaps_ny_fix = pd.read_excel(
        "data/swaps_fixes.xlsx", sheet_name="NY_Fix", index_col=0, parse_dates=True
    )
    swaps_lon_fix.columns = [
        countries_to_fx[col.split()[0][:2]] for col in swaps_lon_fix.columns
    ]
    swaps_ny_fix.columns = [
        countries_to_fx[col.split()[0][:2]] for col in swaps_ny_fix.columns
    ]

    return fx_lon_fix, fx_ny_fix, swaps_lon_fix, swaps_ny_fix
