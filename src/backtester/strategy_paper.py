import logging
from itertools import combinations

import numpy as np
import pandas as pd

from backtester.backtester import Backtester

log = logging.getLogger("backtester")


class PaperStrategy(Backtester):
    def compute_signal(self):
        countries, ma_window, rebalancing_threshold = (
            self.countries,
            self.ma_window,
            self.rebalancing_threshold,
        )
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
                avg = pd.Series(index=diff.index, dtype=float)
                avg.loc[avg.index.hour == 16] = (  # type: ignore
                    diff[diff.index.hour == 16].rolling(ma_window).mean()  # type: ignore
                )
                avg.loc[avg.index.hour == 22] = (  # type: ignore
                    diff[diff.index.hour == 22].rolling(ma_window).mean()  # type: ignore
                )

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
            signal.diff().loc[signal.index.hour == 22].abs() >= rebalancing_threshold,  # type: ignore
            False,
            True,
        )

        not_rebalance.loc[not_rebalance.index.hour == 22] = should_not_rebalance  # type: ignore
        not_rebalance.loc[not_rebalance.index.hour == 16] = False  # type: ignore
        log.debug(f"REBALANCE:\n{ not_rebalance}")

        not_rebalance_times = not_rebalance.sum().sum()
        total_times = not_rebalance.shape[0] * not_rebalance.shape[1]
        rebalancing_count = total_times - not_rebalance_times - total_times / 2
        rebalancing_pct = rebalancing_count / (total_times) * 100
        log.info(
            f"Rebalancing {rebalancing_count} times out of {total_times / 2} ({rebalancing_pct:.2f}%)"
        )
        self.not_rebalance = not_rebalance
