import logging

from backtester.strategy_paper import PaperStrategy
from utils.load_data import load_data

log = logging.getLogger("backtester")
log.setLevel(20)

fx_fixes, swaps_fixes = load_data()


MA_WINDOW = 15
REBAL_THRESHOLD = 1

BT_FX = fx_fixes
BT_SWAPS = swaps_fixes

backtest = PaperStrategy(
    BT_FX, BT_SWAPS, ma_window=MA_WINDOW, rebalancing_threshold=REBAL_THRESHOLD
)
backtest.run()
perf1 = backtest.compute_stats()

print(perf1)
