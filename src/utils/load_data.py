import numpy as np
import pandas as pd


def _process_dates(df: pd.DataFrame, hour: int):
    df.index = pd.to_datetime(df.index, utc=True)
    df.index = df.index.map(lambda x: x.replace(hour=hour))


def _convert_fx(df: pd.DataFrame):
    new_df = df.copy()
    for col in df.columns:
        if col[:3] == "USD":
            new_df[col[3:] + "USD"] = 1 / df[col]
            new_df.drop(col, axis=1, inplace=True)

    return new_df


COUNTRIES_TO_FX = {
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


def load_data(data_dir: str = "data"):
    # FX Fixes
    fx_fixes_path = data_dir + "/fixes/fx_fixes.xlsx"
    fx_lon_fix = pd.read_excel(fx_fixes_path, sheet_name="LON_Fix", index_col=0)
    fx_ny_fix = pd.read_excel(fx_fixes_path, sheet_name="NY_Fix", index_col=0)

    fx_lon_fix.columns = [col.split()[0] for col in fx_lon_fix.columns]
    fx_ny_fix.columns = [col.split()[0] for col in fx_ny_fix.columns]
    _process_dates(fx_lon_fix, 16)
    _process_dates(fx_ny_fix, 22)

    fx_fixes = pd.concat([fx_lon_fix, fx_ny_fix])
    fx_fixes = _convert_fx(fx_fixes)
    fx_fixes.sort_index(inplace=True)

    # Swaps Fixes
    swaps_fixes_path = data_dir + "/fixes/swaps_fixes.xlsx"
    swaps_lon_fix = pd.read_excel(swaps_fixes_path, sheet_name="LON_Fix", index_col=0)
    swaps_ny_fix = pd.read_excel(swaps_fixes_path, sheet_name="NY_Fix", index_col=0)

    swaps_lon_fix.columns = [
        COUNTRIES_TO_FX[col.split()[0][:2]] for col in swaps_lon_fix.columns
    ]
    swaps_ny_fix.columns = [
        COUNTRIES_TO_FX[col.split()[0][:2]] for col in swaps_ny_fix.columns
    ]

    _process_dates(swaps_lon_fix, 16)
    _process_dates(swaps_ny_fix, 22)
    swaps_fixes = pd.concat([swaps_lon_fix, swaps_ny_fix])
    swaps_fixes.sort_index(inplace=True)

    # reindex swaps_fixes to match fx_fixes and append USD swaps data
    us_swaps = swaps_fixes["USD"]

    swaps_fixes = swaps_fixes.reindex(fx_fixes.columns, axis=1)
    swaps_fixes["USD"] = us_swaps

    fx_fixes.ffill(inplace=True)
    swaps_fixes.ffill(inplace=True)

    return fx_fixes, swaps_fixes


def load_em_data(data_path="data"):
    fx_fixes = pd.read_excel(
        data_path + "/em_data.xlsx",
        sheet_name="fx_fixes",
        index_col=0,
        parse_dates=True,
    )
    swaps_fixes = pd.read_excel(
        data_path + "/em_data.xlsx",
        sheet_name="swaps_fixes",
        index_col=0,
        parse_dates=True,
    )

    _, swapsg10 = load_data(data_path)

    usswaps = swapsg10["USD"]
    usswaps = usswaps[usswaps.index.hour == 16]  # type: ignore
    # usswaps.index = usswaps.index.map(lambda x: x.replace(hour=0))
    # usswaps.index = usswaps.index.map(lambda x: x.tz_localize(None))

    fx_fixes.sort_index(inplace=True)
    swaps_fixes.sort_index(inplace=True)

    _process_dates(fx_fixes, 16)
    _process_dates(swaps_fixes, 16)

    swaps_fixes = swaps_fixes.merge(usswaps, left_index=True, right_index=True)

    fx_fixes.dropna(how="all", inplace=True)
    swaps_fixes = swaps_fixes.reindex(fx_fixes.index, method="ffill")
    swaps_fixes.ffill(inplace=True)

    return fx_fixes, swaps_fixes


def load_real_data(data_dir: str):
    fx_fixes, swaps_fixes = load_data(data_dir)

    # load cpi data
    cpi_data = pd.read_excel(data_dir + "/fmt_cpi_data.xlsx", index_col=0)
    cpi_data.ffill(inplace=True)
    cpi_est_data = cpi_data.rolling(12).mean()
    _process_dates(cpi_est_data, 16)

    def map_cols(col: str):
        if col == "USD":
            return col

        return col + "USD"

    cpi_est_data.columns = map(map_cols, list(cpi_est_data.columns))  # type: ignore
    cpi_est_data = cpi_est_data.reindex(swaps_fixes.columns, axis=1)
    cpi_est_data = cpi_est_data.reindex(swaps_fixes.index, axis=0, method="ffill")
    cpi_est_data.sort_index(inplace=True)

    real_swaps_fixes = swaps_fixes - cpi_est_data

    return fx_fixes, real_swaps_fixes


def load_factors(data_path="data/"):
    # pull mkt data
    fx_data = pd.read_excel(
        data_path + "fx_fixes.xlsx", sheet_name="LON_Fix", index_col=0
    )
    fx_data = fx_data.pct_change()
    mkt_returns = fx_data.mean(axis=1).sort_index()
    mkt_returns.name = "mkt"

    # pull factors
    factors_list = ["momentum", "value", "carry"]
    factorsdf = []

    for factor in factors_list:
        df = pd.read_excel(
            data_path + "factors_data.xlsx", sheet_name=factor, index_col=0
        )
        df.index = pd.to_datetime(df.index)
        factorsdf.append(df)

    factors = pd.concat(factorsdf, axis=1)
    factors.sort_index(inplace=True)
    factors.ffill(inplace=True)
    factors = factors.pct_change()
    factors.columns = factors_list

    # merge everything together
    data = pd.concat([mkt_returns, factors], axis=1)
    data.replace(0, np.nan, inplace=True)
    data.dropna(inplace=True)
    # data[data.abs() < 1e-4] = np.nan
    return data
