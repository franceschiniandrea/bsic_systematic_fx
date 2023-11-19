import logging
from typing import Literal

import pandas as pd

from backtester import Backtest
from load_data import load_data

fx_fixes, swaps_fixes = load_data()
backtest = Backtest(fx_fixes, swaps_fixes, ma_window=3, logging_level=1)

print(fx_fixes, fx_fixes.index)

backtest.run()
# backtest.run(fx_lon_fix.index[-2], fx_lon_fix.index[0])
# backtest.run(fx_lon_fix.index[-2], fx_lon_fix.index[0], rebalancing_freq="W-MON")
# backtest.compute_signals(fix=Fixes.NY)
# backtest.compute_positions(fix=Fixes.LON)
# backtest.compute_positions(fix=Fixes.NY)
# backtest.compute_pnl()
backtest.compute_stats()


s = backtest.signal
