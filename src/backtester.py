import logging
from locale import currency
from typing import Literal

import numpy as np
import pandas as pd

# TODO implement slippage and transaction costs
log = logging.getLogger()


class Backtest:
    def __init__(
        self,
        fx_lon_fixes: pd.DataFrame,
        fx_ny_fixes: pd.DataFrame,
        swaps_lon_fixes: pd.DataFrame,
        swaps_ny_fixes: pd.DataFrame,
        ma_window: int = 15,
    ) -> None:
        self.fx_lon_fixes = fx_lon_fixes.astype(float)
        self.fx_ny_fixes = fx_ny_fixes.astype(float)
        self.swaps_lon_fixes = swaps_lon_fixes.astype(float)
        self.swaps_ny_fixes = swaps_ny_fixes.astype(float)

        self.ma_window = ma_window
        self.currencies = fx_lon_fixes.columns
        self.countries = [cur[:3] for cur in self.currencies]
        self.positions = np.zeros(len(self.currencies))

        self.ma_lon = self.swaps_lon_fixes.rolling(ma_window).mean()  # use EWMA?
        self.ma_ny = self.swaps_ny_fixes.rolling(ma_window).mean()

    def signal(self, date: str, fix: Literal["NY", "LON"]):
        countries, ma_window = self.countries, self.ma_window
        n_countries = len(countries)

        if fix == "LON":
            swaps_data = self.swaps_lon_fixes
            fx_data = self.fx_lon_fixes
        elif fix == "NY":
            swaps_data = self.swaps_ny_fixes
            fx_data = self.fx_ny_fixes
        else:
            raise Exception("Invalid fix")

        swaps_data = swaps_data[swaps_data.index <= date]
        fx_data = fx_data[fx_data.index <= date]

        # check that the columns are aligned between ny and lon
        signals = pd.DataFrame(
            data=np.zeros(n_countries), index=countries, columns=["signal"]
        )
        subsignalsdf = pd.DataFrame(
            index=countries,
            columns=countries,
        )

        for i, country1 in enumerate(countries):
            subsignals = []

            country1_cur = country1 + "USD"
            for country2 in countries[i + 1 :]:
                country2_cur = country2 + "USD"

                diff = pd.to_numeric(
                    swaps_data.loc[date, country1_cur]
                ) - pd.to_numeric(swaps_data.loc[date, country2_cur])

                ma = swaps_data[country1_cur] - swaps_data[country2_cur]
                average = ma[
                    ma.index >= pd.to_datetime(date) - pd.Timedelta(days=ma_window)
                ].mean()

                sub_signal = (diff - average) / abs(average)
                subsignals.append(sub_signal)
                subsignalsdf.loc[country1, country2] = sub_signal

        thresholds = subsignalsdf.quantile(axis=0)
        subsignalsdf["0.5_quantile"] = subsignalsdf.quantile(
            axis=1,
        )

        all_subsignals = subsignalsdf.values.flatten()
        all_subsignals = np.abs(all_subsignals[~np.isnan(all_subsignals)])
        quantile_all = np.quantile(all_subsignals, 0.5)
        print(subsignalsdf)
        print(all_subsignals)
        print(quantile_all)

        for country, threshold in zip(countries, thresholds):
            row = subsignalsdf.loc[country]
            for subsignal in row:
                if abs(subsignal) > threshold:
                    signals.loc[country] += 1 if subsignal > 0 else -1
                else:
                    signals.loc[country] -= 1 if subsignal > 0 else -1

        return signals

    def rebalance(self, signal, positions):
        nominal_per_instrument = 100_000  # target a gross exposure in dollars
        new_positions = signal * nominal_per_instrument

        rebalance = new_positions - self.positions
        return rebalance

    def run(self):
        pass

    def plot(self):
        pass

    def test(self, date):
        print(self.fx_lon_fixes)
        print(self.fx_ny_fixes)
        print(self.swaps_lon_fixes)
        print(self.swaps_ny_fixes)
        signal = self.signal(date, "LON")
        print(signal)
        rebalancing_amt = self.rebalance(signal, self.positions)
        print(rebalancing_amt)
