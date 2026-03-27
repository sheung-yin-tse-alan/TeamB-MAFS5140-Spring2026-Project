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

        # 1. Update internal state with the new data
        self.price_history.append(current_prices)

        # Keep only the required lookback period to save memory (Best Practice!)
        if (len(self.price_history) > self.lookback_period_zscore):
            self.price_history.pop(0)

        # 2. Strategy Logic
        # If we don't have enough data yet, stay 100% in cash (return all zeros)
        if (len(self.price_history) < self.lookback_period_zscore):
            return pd.Series(0.0, index=current_prices.index)

        # Convert our history list into a DataFrame to easily calculate the signals
        history_df = pd.DataFrame(self.price_history)
        pct_change_df = history_df / history_df.shift(1) - 1
        z_signal = ((pct_change_df.tail(1) - pct_change_df.mean())/pct_change_df.std()) <= self.thres_zscore #check z-score to ensure previous return is low enough
        z_signal = z_signal.rename(index = {'close':'z'})
        reg = pd.DataFrame(index=['reg'], columns = pct_change_df.columns)
        for col in pct_change_df.columns:
          coeffs = np.polyfit(pct_change_df.shift(1)[-self.lookback_period_reg:][col],pct_change_df[-self.lookback_period_reg:][col],1)
          reg[col] = coeffs[0]
        reg_signal = reg <= self.thres_reg #check if the stock has mean reversion tendency by LM
        signals = pd.concat([z_signal,reg_signal],axis=0)
        #print(signals.sum(axis=1))


        # Identify assets where both signals are on
        bullish_assets = signals.columns[signals.all()].tolist()

        # 3. Portfolio Allocation
        # Initialize all weights to 0.0
        weights = pd.Series(0.0, index=current_prices.index)

        # Allocate equally among bullish assets
        num_bullish = len(bullish_assets)

        #to avoid bearish timestamp (when the whole market is bearish)
        #if num_bullish > 0.05*len(history_df.columns): #avoid bearish market
        #  return pd.Series(0.0, index=current_prices.index)

        if num_bullish > 0:
            weight_per_asset = 1.0 / num_bullish
            weights[bullish_assets] = weight_per_asset

        # Return the weights.
        # The engine will verify that weights >= 0 and weights.sum() <= 1.0
        return weights
