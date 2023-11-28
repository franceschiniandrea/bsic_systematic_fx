import logging
from itertools import product

import pandas as pd
from matplotlib import pyplot as plt

from backtester.strategy_paper_improved import ImprovedStrategy
from utils.load_data import load_data

log = logging.getLogger("backtester")
log.setLevel(30)

fx_fixes, swaps_fixes = load_data()
swaps_fixes.ffill(inplace=True)
fx_fixes.ffill(inplace=True)

BT_FX = fx_fixes
BT_SWAPS = swaps_fixes

MA_WINDOW_LONGS = range(10, 31, 1)
MA_WINDOW_SHORTS = range(2, 11)
REBAL_THRESHOLD = 3

sharpes = pd.DataFrame(index=MA_WINDOW_SHORTS, columns=MA_WINDOW_LONGS)

for short, long in product(MA_WINDOW_SHORTS, MA_WINDOW_LONGS):
    print(f"{short}, {long}")
    MA_WINDOW = (short, long)

    bt = ImprovedStrategy(fx_fixes, swaps_fixes, short, long, REBAL_THRESHOLD, None)

    bt.run()

    perf = bt.compute_stats()

    sharpes.loc[short, long] = perf.loc["average", "sharpe"]

print(sharpes)

# sharpes.to_excel("results/improved_tot.xlsx")
