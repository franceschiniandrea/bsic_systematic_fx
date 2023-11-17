from backtester import Backtest
from load_data import load_data

fx_lon_fix, fx_ny_fix, swaps_lon_fix, swaps_ny_fix = load_data()
backtest = Backtest(fx_lon_fix, fx_ny_fix, swaps_lon_fix, swaps_ny_fix, ma_window=15)

print(swaps_lon_fix)

backtest.test("10-10-2017")
