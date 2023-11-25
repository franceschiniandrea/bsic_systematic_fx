import logging

from matplotlib import pyplot as plt

from backtester.strategy_improved import ImprovedStrategy as ImprStrat
from backtester.strategy_improved_2 import ImprovedStrategy as ImprStratSigm
from load_data import load_data

log = logging.getLogger("backtester")
log.setLevel(10)

fx_fixes, swaps_fixes, cpi_data, real_swaps_fixes = load_data()

MA_WINDOW = (5, 15)
REBAL_THRESHOLD = 3

swaps_fixes.ffill(inplace=True)
fx_fixes.ffill(inplace=True)

BT_FX = fx_fixes
BT_SWAPS = swaps_fixes

bt = ImprStratSigm(
    fx_fixes, swaps_fixes, MA_WINDOW[0], MA_WINDOW[1], REBAL_THRESHOLD, None
)

bt.run()

perf = bt.compute_stats()

print(perf)


def process_for_excel(df, name: str):
    df.index = df.index.map(lambda x: x.tz_localize(None))

    df.to_excel(name + ".xlsx")


signal = bt.signal
pos = bt.positions

process_for_excel(signal, "signal")
process_for_excel(pos, "positions")
