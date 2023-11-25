import logging
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_bsic import apply_bsic_logo, apply_bsic_style

logging.basicConfig(stream=sys.stdout)
log = logging.getLogger("backtester")

spreads_dict = {
    "spread": {
        "AUDUSD": 0.006 / 100,
        "CADUSD": 0.010 / 100,
        "CHFUSD": 0.011 / 100,
        "DKKUSD": 0.005 / 100,
        "EURUSD": 0.0036 / 100,
        "GBPUSD": 0.005 / 100,
        "JPYUSD": 0.006 / 100,
        "NOKUSD": 0.035 / 100,
        "NZDUSD": 0.014 / 100,
        "SEKUSD": 0.032 / 100,
    }
}
spreads = pd.DataFrame.from_dict(spreads_dict, orient="index")


class Backtester:
    def __init__(
        self,
        fx_fixes: pd.DataFrame,
        swaps_fixes: pd.DataFrame,
        ma_window: int,
        rebalancing_threshold: int,
        rebalancing_freq=None,
    ) -> None:
        self.fx_fixes = fx_fixes
        self.swaps_fixes = swaps_fixes
        self.spreads = spreads
        self.rebalancing_threshold = rebalancing_threshold
        self.rebalancing_freq = rebalancing_freq

        self.ma_window = ma_window
        self.currencies = fx_fixes.columns
        self.countries = [cur[:3] for cur in self.currencies] + ["USD"]

        # initialize positions
        self.positions = pd.DataFrame(
            0, index=fx_fixes.index, columns=self.currencies, dtype=float
        )
        # initialize signals for LON and NY fixes (only LON if no NY is present)
        self.signal = pd.DataFrame(
            0, index=fx_fixes.index, columns=self.currencies, dtype=int
        )

    def compute_signal(self):
        self.not_rebalance = pd.DataFrame(
            0, index=self.fx_fixes.index, columns=self.currencies
        )

    def compute_positions(
        self,
        target_gross_exposure: float = 1_000_000,
    ):
        log.debug("-" * 20 + "COMPUTE POSITIONS" + "-" * 20)

        signal = self.signal.iloc[self.ma_window * 2 :]

        log.debug(f"Signal: \n{signal.iloc[:30]}")

        base_amt = target_gross_exposure / signal.abs().sum(axis=1)
        nominal_exposures = signal * base_amt.to_numpy().reshape(-1, 1)
        nominal_exposures = nominal_exposures.reindex(
            self.fx_fixes.index, method="ffill"
        )

        # do not rebalance when rebalance == False
        nominal_exposures = pd.DataFrame(
            np.where(self.not_rebalance, np.nan, nominal_exposures),
            index=nominal_exposures.index,
            columns=nominal_exposures.columns,
        )
        log.debug(f"Nominal Exposures:\n{nominal_exposures}")

        nominal_exposures.ffill(inplace=True)
        self.positions[:] = nominal_exposures

        if self.rebalancing_freq is not None:
            if self.rebalancing_freq == "W-MON":
                log.debug("Rebalancing weekly on Monday")
                positions = self.positions
                monthly_pos: pd.DataFrame = (
                    positions[positions.index.hour == 16].resample("W-MON").last()  # type: ignore
                )
                monthly_pos.index = monthly_pos.index.map(lambda x: x.replace(hour=16))
                new_positions = monthly_pos.reindex(positions.index, method="ffill")
                self.positions[:] = new_positions

    def compute_transaction_costs(self):
        log.debug("-" * 20 + "COMPUTE TC" + "-" * 20)
        positions = self.positions
        spreads = self.spreads.reindex(columns=positions.columns)

        position_chg = positions.diff().abs()
        tc = position_chg * spreads.to_numpy()
        return tc

    def compute_pnl(self):
        log.debug("-" * 20 + "COMPUTE PNL" + "-" * 20)
        positions = self.positions
        tc = self.compute_transaction_costs().reindex(columns=positions.columns)
        returns = self.fx_fixes.pct_change()

        # spreads not available for EMs, so getting a df of nans
        pnl = returns * positions.shift(1) - tc
        pnl["total"] = pnl.sum(axis=1)
        pnl["total_pct"] = pnl["total"] / 1_000_000

        self.pnl = pnl

    def compute_stats(self):
        log.debug("-" * 20 + "COMPUTE STATS" + "-" * 20)

        pnl = self.pnl[["total", "total_pct"]].copy()

        def compute_return(col: pd.Series):
            return col.mean() * len(col)

        def compute_vol(col):
            return col.std() * np.sqrt(len(col))

        pnl_pct = pnl["total_pct"]
        y_resample = pnl_pct.resample("Y")

        y_return = y_resample.apply(compute_return)
        y_vol = y_resample.apply(compute_vol)
        sharpe = y_return / y_vol
        hit_ratio = y_resample.apply(lambda x: (x > 0).sum() / len(x))
        kurt = y_resample.apply(lambda x: x.kurtosis())
        skew = y_resample.apply(lambda x: x.skew())

        df = pd.DataFrame(
            {
                "return": y_return,
                "vol": y_vol,
                "skew": skew,
                "kurt": kurt,
                "hit_ratio": hit_ratio,
                "sharpe": sharpe,
            }
        )
        df.index = pd.to_datetime(df.index).year
        df.loc["average"] = df.mean(axis=0)
        df.loc["average", "kurt"] = pnl_pct.kurt()
        df.loc["average", "skew"] = pnl_pct.skew()

        if 2000 in list(df.index):
            df.loc["2000-2010"] = df.loc[2000:2011].mean(axis=0)
            df.loc["2000-2010", "kurt"] = pnl_pct.loc["2000":"2011"].kurt()
            df.loc["2000-2010", "skew"] = pnl_pct.loc["2000":"2011"].skew()

        df.loc["2010-2020"] = df.loc[2010:2020].mean(axis=0)
        df.loc["2010-2020", "kurt"] = pnl_pct.loc["2010":"2021"].kurt()
        df.loc["2010-2020", "skew"] = pnl_pct.loc["2010":"2021"].skew()

        return df

    def plot(self):
        pnl = self.pnl
        cumul_pnl = pnl["total"].cumsum()

        fig, ax = plt.subplots(1, 1)
        ax.set_title("Cumulative PnL")
        apply_bsic_logo(fig, ax)
        apply_bsic_style(fig, ax)

        ax.plot(cumul_pnl.index, cumul_pnl)

        plt.show()

    def run(
        self,
    ):
        log.debug(f"Running Strategy for currencies:\n{self.currencies}")
        self.compute_signal()
        self.compute_positions()
        self.compute_pnl()
