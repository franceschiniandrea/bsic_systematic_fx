import logging
import re

import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib import pyplot as plt

from backtester import Backtest
from backtester_2 import Backtest as Backtest2
from load_data import load_data
from performance_stats import PerformanceStats

log = logging.getLogger("backtester")
log.setLevel(10)

fx_fixes, swaps_fixes, cpi_data, real_swaps_fixes = load_data()

real_fx_fixes: pd.DataFrame = fx_fixes[
    (fx_fixes.index.year >= 2010) & (fx_fixes.index.year <= 2020)  # type: ignore
]
real_swaps_fixes: pd.DataFrame = real_swaps_fixes[
    (real_swaps_fixes.index.year >= 2010) & (real_swaps_fixes.index.year <= 2020)  # type: ignore
]

MA_WINDOW = 10
REBAL_THRESHOLD = 3

BT_FX = real_fx_fixes
BT_SWAPS = real_swaps_fixes

# real_swaps_fixes.index = real_swaps_fixes.index.map(lambda x: x.tz_localize(None))
# real_swaps_fixes.to_excel("real_swaps_fixes.xlsx")

backtest = Backtest(BT_FX, BT_SWAPS, ma_window=MA_WINDOW)
backtest.run(rebalancing_threshold=REBAL_THRESHOLD, rebalancing_freq=None)
perf1 = backtest.compute_stats()

bt2 = Backtest2(BT_FX, BT_SWAPS, ma_window=MA_WINDOW)
bt2.run(rebalancing_freq=None, rebalancing_threshold=REBAL_THRESHOLD)
perf2 = bt2.compute_stats()

pnl = bt2.pnl
stats = PerformanceStats(pnl)


def plot_pnl_currencies(pnl):
    pnl = pnl[pnl.index.year > 2010]
    fig, axs = plt.subplots(4, 3)
    fig.tight_layout()

    for i, ax in enumerate(axs.flatten()):
        currency = pnl.columns[i]
        pnl[currency].cumsum().plot(ax=ax, title=currency)

    fig.suptitle("PnL for each currency")

    plt.show()


plot_pnl_currencies(pnl)
# real_swaps_fixes.index = real_swaps_fixes.index.map(lambda x: x.tz_localize(None))
# try:
#     real_swaps_fixes.to_excel("real_swaps_fixes.xlsx")
# except:
#     print("error occured")
# plot_pnl_currencies(pnl)

print(perf1)
print(perf2)
# windows = range(2, 30, 2)
# perfsdf = pd.DataFrame(index=windows, columns=["avg", "avg2000s", "avg2010s"])
# for ma_window in windows:
#     backtest = Backtest(fx_fixes, swaps_fixes, ma_window=ma_window)
#     backtest.run(rebalancing_freq=None)
#     perfs = backtest.compute_stats()

#     perfsdf.loc[ma_window, :] = (
#         perfs["sharpe"].loc[["average", "average2000s", "average2010s"]].to_numpy()
#     )

# print(perfsdf)
# backtest.run(rebalancing_freq=None)

# backtest.compute_stats()
# pos = backtest.positions
# pos.index = pos.index.map(lambda x: x.tz_localize(None))

# pos.to_excel("positions.xlsx")
# backtest.plot()

# pnl = backtest.pnl
# pct_pnl = pnl["total_pct"]
# avg = pct_pnl.mean()
# pnl_stdev = pct_pnl.std()
# kurt = pct_pnl.kurtosis()
# skew = pct_pnl.skew()
# print(f"mean: {avg}, kurtosis: {kurt}, skew: {skew}, stdev: {pnl_stdev}")
# x = np.linspace(-6 * pnl_stdev, 6 * pnl_stdev, 100)
# plt.plot(x, stats.norm.pdf(x, 0, pnl_stdev))

# plt.hist(pct_pnl, bins=100, density=True)
# plt.show()
