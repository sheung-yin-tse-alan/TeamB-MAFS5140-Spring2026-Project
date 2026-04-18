import pandas as pd
import numpy as np

class Strategy:
    def __init__(self):
        self.price_history = []
        self.lookback_period_zscore = 100
        self.thres_zscore = -1
        self.lookback_period_reg = 70
        self.thres_reg = -0.3

    def step(self, current_market_data: pd.DataFrame) -> pd.Series:
        if "close" not in current_market_data.columns:
            raise ValueError("Input market data must contain a 'close' column.")

        current_prices = current_market_data["close"]

        # Update history
        self.price_history.append(current_prices)

        if len(self.price_history) > self.lookback_period_zscore:
            self.price_history.pop(0)

        # Not enough history yet
        if len(self.price_history) < self.lookback_period_zscore:
            return pd.Series(0.0, index=current_prices.index)

        # Build history dataframe
        history_df = pd.DataFrame(self.price_history)

        # Compute returns
        pct_change_df = history_df.pct_change()

        # ---------- Z-SCORE SIGNAL ----------
        mean_ret = pct_change_df.mean()
        std_ret = pct_change_df.std()

        latest_ret = pct_change_df.iloc[-1]

        # Avoid division by zero
        zscore = pd.Series(np.nan, index=pct_change_df.columns)
        valid_std = std_ret > 0
        zscore[valid_std] = (latest_ret[valid_std] - mean_ret[valid_std]) / std_ret[valid_std]

        z_signal = zscore <= self.thres_zscore

        # ---------- REGRESSION SIGNAL ----------
        reg = pd.Series(np.nan, index=pct_change_df.columns)

        for col in pct_change_df.columns:
            x = pct_change_df[col].shift(1).iloc[-self.lookback_period_reg:]
            y = pct_change_df[col].iloc[-self.lookback_period_reg:]

            # Keep only finite paired observations
            valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
            x_valid = x[valid]
            y_valid = y[valid]

            # Need at least 2 points
            if len(x_valid) < 2:
                continue

            # Skip near-constant x
            if np.isclose(x_valid.std(), 0):
                continue

            try:
                coeffs = np.polyfit(x_valid, y_valid, 1)
                reg[col] = coeffs[0]
            except Exception:
                continue

        reg_signal = reg <= self.thres_reg

        # ---------- COMBINED SIGNAL ----------
        combined_signal = z_signal & reg_signal.fillna(False)

        bullish_assets = combined_signal[combined_signal].index.tolist()

        # ---------- PORTFOLIO WEIGHTS ----------
        weights = pd.Series(0.0, index=current_prices.index)

        num_bullish = len(bullish_assets)
        if num_bullish > 0:
            weights[bullish_assets] = 1.0 / num_bullish

        return weights
