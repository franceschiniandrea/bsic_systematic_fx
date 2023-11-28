import logging

import numpy as np
import pandas as pd

from backtester.backtester import Backtester

logging.basicConfig(stream=sys.stdout)
log = logging.getLogger("backtester")


def process_for_excel(df, name: str):
    df = df.copy()
    df.index = df.index.map(lambda x: x.tz_localize(None))

    df.to_excel("output/" + name + ".xlsx")


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
        countries = np.array(self.countries)
        swaps_data, signal = self.swaps_fixes, self.signal

        # zscores = pd.DataFrame(index=swaps_data.index)

        for i, country1 in enumerate(countries[countries != "USD"]):
            country1_cur = country1 + "USD"
            subsignals = pd.DataFrame(
                0,
                index=swaps_data.index,
                columns=[c2 + country1 for c2 in countries[countries != country1]],
                dtype=float,
            )

            for country2 in countries[i + 1 :]:
                country2_cur = country2 + "USD" if country2 != "USD" else country2
                diff = swaps_data[country1_cur] - swaps_data[country2_cur]

                zscore_st = (diff - diff.rolling(st_window).mean()) / diff.rolling(
                    st_window
                ).std()
                zscore_lt = (diff - diff.rolling(lt_window).mean()) / diff.rolling(
                    lt_window
                ).std()

                subsignals[country2 + country1] = (zscore_st + zscore_lt) / 2

                # signal_val = (expit(zscore_st) + expit(zscore_lt)) / 2

                # if country1 != "USD":
                #     signal[country1 + "USD"] += signal_val
                # if country2 != "USD":
                #     signal[country2 + "USD"] -= signal_val

            signal[country1 + "USD"] = subsignals.mean(axis=1)

        # process_for_excel(zscores, "zscoresdata")
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
