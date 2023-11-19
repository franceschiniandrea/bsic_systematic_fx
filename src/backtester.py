import logging
import sys
from enum import Enum
from itertools import combinations
from locale import currency
from typing import Literal

import numpy as np
import pandas as pd

# TODO implement slippage and transaction costs
logging.basicConfig(stream=sys.stdout)
log = logging.getLogger()

Fixes = Enum("Fixes", ["LON", "NY"])


class Backtest:
    def __init__(
        self,
        fx_lon_fixes: pd.DataFrame,
        fx_ny_fixes: pd.DataFrame,
        swaps_lon_fixes: pd.DataFrame,
        swaps_ny_fixes: pd.DataFrame,
        ma_window: int = 15,
        logging_level: int = 0,
    ) -> None:
        log.setLevel(logging_level)

        self.ma_window = ma_window
        self.currencies = fx_lon_fixes.columns
        self.countries = [cur[:3] for cur in self.currencies]
        self.positions = np.zeros(len(self.currencies))

        self.ma_lon = self.swaps_lon_fixes.rolling(ma_window).mean()  # use EWMA?
        self.ma_ny = self.swaps_ny_fixes.rolling(ma_window).mean()

    def compute_signals(self, fix: Fixes):
        countries, ma_window = self.countries, self.ma_window

        if fix == Fixes.LON:
            swaps_data = self.swaps_lon_fixes
            signals = self.signals_lon
        else:
            swaps_data = self.swaps_ny_fixes
            signals = self.signals_ny

        pairs = ["".join(pair) for pair in combinations(countries, 2)]
        subsignals = pd.DataFrame(columns=pairs, dtype=float, index=swaps_data.index)
        for i, country1 in enumerate(countries):
            country1_cur = country1 + "USD"
            for country2 in countries[i + 1 :]:
                country2_cur = country2 + "USD"
                # print("-" * 50)
                # print(swaps_data[country1_cur], swaps_data[country2_cur])
                # print("-" * 50)
                diff = swaps_data[country1_cur] - swaps_data[country2_cur]
                avg = diff.rolling(
                    ma_window,
                ).mean()

                subsignals_col = (diff - avg) / np.abs(avg)
                log.debug(f"diff for {country1}-{country2}:\n{diff}")
                log.debug(f"avg for {country1}-{country2}:\n{avg}")
                log.debug(f"subsignals for {country1}-{country2}:\n{subsignals_col}")

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

            col_data[(col_data.abs() > thresholds) & (col_data >= 0)] = 1
            col_data[(col_data.abs() > thresholds) & (col_data < 0)] = -1
            col_data[col_data.abs() != 1] = 0

            signals[country1 + "USD"] += col_data
            signals[country2 + "USD"] -= col_data

    def compute_positions(self, fix: Fixes, target_gross_exposure: float = 1_000_000):
        log.debug("-" * 20 + "COMPUTE POSITIONS" + "-" * 20)

        signal = self.signals_lon if fix == Fixes.LON else self.signals_ny

        base_amt = target_gross_exposure / signal.abs().sum(axis=1)
        nominal_exposures = signal * base_amt.to_numpy().reshape(-1, 1)
        self.positions[:] = nominal_exposures
        print("POS COLS", self.positions.columns)

    def compute_pnl(self):
        log.debug("-" * 20 + "COMPUTE PNL" + "-" * 20)
        positions = self.positions

        returns = self.fx_lon_fixes.pct_change()
        log.debug(f"returns:\n{returns}")
        log.debug(f"positions (shifted):\n{positions.shift(1)}")
        positions, returns = positions.align(returns, join="outer", axis=1)
        print(positions.columns, returns.columns)
        log.debug(f"prod:\n{returns * positions.shift(1)}")
        log.debug(f"prod2:\n{returns * positions}")
        pnl = returns * positions.shift(1)
        pnl["total"] = pnl.sum(axis=1)
        pnl["total_pct"] = pnl["total"] / 1_000_000
        log.debug(f"pnl:\n{pnl}")

        self.pnl = pnl


    def run(self):
        self.compute_signals(Fixes.LON)
        self.compute_positions(Fixes.LON)
        self.compute_pnl()
