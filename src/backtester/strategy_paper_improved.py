import logging
from itertools import combinations

import numpy as np
import pandas as pd

from backtester.backtester import Backtester

log = logging.getLogger("backtester")


class ImprovedStrategy(Backtester):
    def __init__(
        self,
        fx_fixes: pd.DataFrame,
        swaps_fixes: pd.DataFrame,
        st_window: int,
        lt_window: int,
        rebalancing_threshold: int,
        rebalancing_freq=None,
    ):
        self.st_window = st_window
        self.lt_window = lt_window

        super().__init__(
            fx_fixes, swaps_fixes, st_window, rebalancing_threshold, rebalancing_freq
        )

    def compute_signal(self):
        rebalancing_threshold = self.rebalancing_threshold
        st_window, lt_window = self.st_window, self.lt_window
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

                avg = (
                    diff.ewm(span=st_window * 2).mean()
                    + diff.ewm(span=lt_window * 2).mean()
                ) / 2

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
