import pandas as pd

# TODO implement slippage and transaction costs


class Backtest:
    def __init__(
        self,
        fx_lon_fixes: pd.DataFrame,
        fx_ny_fixes: pd.DataFrame,
        swaps_lon_fixes: pd.DataFrame,
        swaps_ny_fixes: pd.DataFrame,
        ma_window: int = 15,
    ) -> None:
        self.fx_lon_fixes = fx_lon_fixes
        self.fx_ny_fixes = fx_ny_fixes
        self.swaps_lon_fixes = swaps_lon_fixes
        self.swaps_ny_fixes = swaps_ny_fixes

        self.ma_window = ma_window

    def signal(self):
        pass

    def run(self):
        pass

    def plot(self):
        pass
