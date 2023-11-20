import pandas as pd

from backtester import Backtest
from load_data import load_data

fx_fixes, swaps_fixes = load_data()
print(fx_fixes)
backtest = Backtest(fx_fixes, swaps_fixes, ma_window=2, logging_level=0)

print(backtest.signal)
backtest.run(rebalancing_freq="W-MON")

backtest.compute_stats()
pos = backtest.positions

backtest.plot()
