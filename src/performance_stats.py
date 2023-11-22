import numpy as np
import pandas as pd
from scipy.stats import norm


class PerformanceStats:
    def __init__(self, pnl) -> None:
        self.pnl = pnl

    def probabilistic_sharpe(self):
        pnl = self.pnl

        def compute_return(col: pd.Series):
            return col.mean() * len(col)

        def compute_vol(col):
            return col.std() * np.sqrt(len(col))

        yearly_return = pnl["total_pct"].resample("Y").apply(compute_return)
        yearly_vol = pnl["total_pct"].resample("Y").apply(compute_vol)
        sr = yearly_return / yearly_vol
        sr.index = pd.to_datetime(sr.index).year

        sr = sr.loc[2003:2014].mean()

        skew = pnl["total_pct"].skew()
        kurt = pnl["total_pct"].kurtosis()
        n = len(pnl["total_pct"])

        sr_stderr = np.sqrt((1 - skew * sr + (kurt - 1) * (sr**2) / 4) / n - 1)

        confidence_level = 0.95
        alpha = 1 - confidence_level
        Z_alpha_over_2 = norm.ppf(1 - alpha / 2)  # 95% CI

        lower_bound = sr - Z_alpha_over_2 * sr_stderr
        upper_bound = sr + Z_alpha_over_2 * sr_stderr

        return sr, lower_bound, upper_bound

    def plot(self):
        pass

    def summary(self):
        pass
