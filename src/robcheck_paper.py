import logging

import pandas as pd

from backtester.strategy_paper import PaperStrategy
from load_data import load_data

log = logging.getLogger("backtester")
log.setLevel(30)

fx_fixes, swaps_fixes, cpi_data, real_swaps_fixes = load_data()
swaps_fixes.ffill(inplace=True)
fx_fixes.ffill(inplace=True)

BT_FX = fx_fixes
BT_SWAPS = swaps_fixes

MA_WINDOWS = range(2, 31)
REBAL_THRESHOLD = 3

sharpes = pd.DataFrame(index=MA_WINDOWS, columns=["sharpe"])

for window in MA_WINDOWS:
    print(f"{window}")

    bt = PaperStrategy(fx_fixes, swaps_fixes, window, REBAL_THRESHOLD, "W-MON")

    bt.run()

    perf = bt.compute_stats()

    sharpes.loc[window, "sharpe"] = perf.loc["2010-2020", "sharpe"]

print(sharpes)

sharpes.to_excel("results/paper_weeklyrebal_2010s.xlsx")
