import logging
from typing import Literal

import pandas as pd

from backtester import Backtest, Fixes
from load_data import load_data

fx_lon_fix, fx_ny_fix, swaps_lon_fix, swaps_ny_fix = load_data()
backtest = Backtest(
    fx_lon_fix, fx_ny_fix, swaps_lon_fix, swaps_ny_fix, ma_window=15, logging_level=1
)

# backtest.run(fx_lon_fix.index[-2], fx_lon_fix.index[0])
# backtest.run(fx_lon_fix.index[-2], fx_lon_fix.index[0], rebalancing_freq="W-MON")

backtest.compute_signals(fix=Fixes.LON)
backtest.compute_positions(fix=Fixes.LON)
backtest.compute_pnl()
backtest.compute_stats()
