import logging
import sys
from enum import Enum
from itertools import combinations
from locale import currency
from typing import Literal

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_bsic import apply_bsic_logo, apply_bsic_style

# TODO implement slippage and transaction costs
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


class Backtest:
    def __init__(
        self,
        fx_fixes: pd.DataFrame,
        swaps_fixes: pd.DataFrame,
        ma_window: int = 15,
    ) -> None:
        self.fx_fixes = fx_fixes
        self.swaps_fixes = swaps_fixes
        self.spreads = spreads

        self.ma_window = ma_window
        self.currencies = fx_fixes.columns
        self.countries = [cur[:3] for cur in self.currencies] + ["USD"]

        # initialize positions
        self.positions = pd.DataFrame(
            0, index=fx_fixes.index, columns=self.currencies, dtype=float
        )
        # initialize signals for LON and NY fixes
        self.signal = pd.DataFrame(
            0, index=fx_fixes.index, columns=self.currencies, dtype=int
        )

    def compute_signals(self, rebalancing_threshold: int):
        countries, ma_window = self.countries, self.ma_window
        swaps_data, signal = self.swaps_fixes, self.signal

        pairs = [
            "".join(pair) for pair in combinations(countries, 2)
        ]  # create all possible pairs of countries
        subsignals = pd.DataFrame(columns=pairs, dtype=float, index=swaps_data.index)
        for i, country1 in enumerate(countries):
            country1_cur = country1 + "USD"
            for country2 in countries[i + 1 :]:
                country2_cur = country2 + "USD" if country2 != "USD" else country2
                diff = swaps_data[country1_cur] - swaps_data[country2_cur]

                # to make sure that
                # - avg for london is calculated using lon data
                # - avg for ny is calculated using ny data
                avg = diff.ewm(span=ma_window * 2).mean()

                subsignals_col = (diff - avg) / np.abs(avg)
                subsignals[country1 + country2] = subsignals_col

        # we now have the subsignals for all dates for all combinations of countries

        # compute the threshold for each country
        subsignals["threshold"] = subsignals.abs().quantile(
            axis=1, q=0.5, numeric_only=True
        )
        log.debug(f"subsignals:\n{subsignals}")

        # compute the composite signals for each country
        log.debug("-" * 20 + "COMPOSITE SIGNALS" + "-" * 20)
        for col in subsignals.drop("threshold", axis=1).columns:
            country1, country2 = col[:3], col[3:]

            thresholds = subsignals["threshold"]
            col_data = subsignals[col].copy()

            # find dates where the subsignal is above the threshold
            # how does this handle NaNs?

            col_data[(col_data.abs() >= thresholds) & (col_data >= 0)] = 1
            col_data[(col_data.abs() >= thresholds) & (col_data < 0)] = -1
            col_data[col_data.abs() != 1] = 0

            if country1 != "USD":
                signal[country1 + "USD"] += col_data
            if country2 != "USD":
                signal[country2 + "USD"] -= col_data

        # compute if the strategy should rebalance on that day
        not_rebalance = pd.DataFrame(
            index=signal.index, columns=signal.columns, dtype=bool
        )
        should_not_rebalance = np.where(
            signal.diff().loc[signal.index.hour == 22].abs() > rebalancing_threshold,  # type: ignore
            False,
            True,
        )

        not_rebalance.loc[not_rebalance.index.hour == 22] = should_not_rebalance  # type: ignore
        not_rebalance.loc[not_rebalance.index.hour == 16] = False  # type: ignore
        log.debug(f"REBALANCE:\n{ not_rebalance}")
        not_rebalance_times = not_rebalance.sum().sum()
        total_times = not_rebalance.shape[0] * not_rebalance.shape[1]
        log.debug(
            f"Rebalancing {total_times - not_rebalance_times} times out of {total_times} ({(total_times - not_rebalance_times) / total_times * 100:.2f}%)"
        )
        self.not_rebalance = not_rebalance

    def compute_positions(
        self, target_gross_exposure: float = 1_000_000, rebalancing: str | None = None
    ):
        log.debug("-" * 20 + "COMPUTE POSITIONS" + "-" * 20)

        signal = self.signal.iloc[self.ma_window * 2 :]

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

        if rebalancing is not None:
            if rebalancing == "W-MON":
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

        y_return = pnl["total_pct"].resample("Y").apply(compute_return)
        y_vol = pnl["total_pct"].resample("Y").apply(compute_vol)
        sharpe = y_return / y_vol

        df = pd.DataFrame({"return": y_return, "vol": y_vol, "sharpe": sharpe})
        df.index = pd.to_datetime(df.index).year
        df.loc["average"] = df.mean(axis=0)
        if 2000 in list(df.index):
            df.loc["average2000s"] = df.loc[2000:2011].mean(axis=0)
        df.loc["average2010s"] = df.loc[2010:2020].mean(axis=0)

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
        rebalancing_threshold: int,
        rebalancing_freq=None,
    ):
        self.compute_signals(rebalancing_threshold)
        self.compute_positions(rebalancing=rebalancing_freq)
        self.compute_pnl()
