#best so far
import pandas as pd
import numpy as np

class KalmanFilter:
    """Simple Kalman filter implementation for state estimation"""
    def __init__(self, process_variance=0.005, measurement_variance=0.05):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = 0.0
        self.error_covariance = 1.0

    def update(self, measurement, lag_return):
        # Prediction step
        predicted_estimate = self.estimate
        predicted_error = self.error_covariance + self.process_variance

        # Observation matrix
        H = lag_return

        # Kalman gain
        kalman_gain = predicted_error * H / (H * predicted_error * H + self.measurement_variance)

        # Update step
        self.estimate = predicted_estimate + kalman_gain * (measurement - H * predicted_estimate)
        self.error_covariance = (1 - kalman_gain * H) * predicted_error

        return self.estimate

class Strategy:
    def __init__(self):
        # ---------- PARAMETERS TO FINE-TUNE ----------

        self.lookback_period_zscore = 75      # Tune range: 30-100
        self.base_thres_zscore = -1.2         # Tune range: -0.5 to -1.5
        self.lookback_period_reg = 40         # Tune range: 30-70
        self.base_thres_reg = -0.25           # Tune range: -0.05 to -0.3
        self.volatility_lookback = 20         # Tune range: 15-30
        self.volatility_scaling_factor = 0.5  # Tune range: 0.3-1.5
        self.kalman_process_variance = 0.008  # Tune range: 0.002-0.02
        self.kalman_measurement_variance = 0.03  # Tune range: 0.01-0.2

        # Internal state
        self.price_history = []
        self.kalman_filters = {}
        self.kalman_beta_history = {}

    def compute_dynamic_threshold(self, current_volatility, historical_volatility):
        if historical_volatility == 0:
            return self.base_thres_zscore

        vol_ratio = current_volatility / historical_volatility

        if vol_ratio > 1:
            # High volatility: easier entry
            scaled_threshold = self.base_thres_zscore / (1 + 0.3 * (vol_ratio - 1))
        else:
            # Low volatility: stricter entry
            scaled_threshold = self.base_thres_zscore * (1 - 0.2 * (1 - vol_ratio))

        return max(min(scaled_threshold, -0.3), -1.5)

    def step(self, current_market_data: pd.DataFrame) -> pd.Series:
        if "close" not in current_market_data.columns:
            raise ValueError("Input market data must contain a 'close' column.")

        current_prices = current_market_data["close"]

        # Update history
        self.price_history.append(current_prices)

        if len(self.price_history) > self.lookback_period_zscore:
            self.price_history.pop(0)

        if len(self.price_history) < max(self.lookback_period_zscore, 20):
            return pd.Series(0.0, index=current_prices.index)

        # Build history dataframe
        history_df = pd.DataFrame(self.price_history)
        pct_change_df = history_df.pct_change()

        # ---------- Z-SCORE SIGNAL ----------
        mean_ret = pct_change_df.mean()
        std_ret = pct_change_df.std()

        recent_volatility = pct_change_df.iloc[-self.volatility_lookback:].std()
        historical_volatility = pct_change_df.std()

        latest_ret = pct_change_df.iloc[-1]

        zscore = pd.Series(np.nan, index=pct_change_df.columns)
        valid_std = std_ret > 0
        zscore[valid_std] = (latest_ret[valid_std] - mean_ret[valid_std]) / std_ret[valid_std]

        # Calculate dynamic threshold
        z_signal = pd.Series(False, index=pct_change_df.columns)
        for col in pct_change_df.columns:
            if col in recent_volatility.index and col in historical_volatility.index:
                dynamic_threshold = self.compute_dynamic_threshold(
                    recent_volatility[col],
                    historical_volatility[col]
                )
                if col in zscore.index and not pd.isna(zscore[col]):
                    z_signal[col] = zscore[col] <= dynamic_threshold

        # ---------- KALMAN FILTER SIGNAL ----------
        for col in pct_change_df.columns:
            if col not in self.kalman_filters:
                self.kalman_filters[col] = KalmanFilter(
                    process_variance=self.kalman_process_variance,
                    measurement_variance=self.kalman_measurement_variance
                )
                self.kalman_beta_history[col] = []

        reg_signal = pd.Series(False, index=pct_change_df.columns)

        for col in pct_change_df.columns:
            returns_series = pct_change_df[col].dropna()
            if len(returns_series) < self.lookback_period_reg // 2:
                continue

            kalman_betas = []
            kf = self.kalman_filters[col]

            # Run Kalman filter
            for i in range(1, min(len(returns_series), self.lookback_period_reg)):
                current_ret = returns_series.iloc[i]
                lag_ret = returns_series.iloc[i-1]

                if pd.isna(current_ret) or pd.isna(lag_ret):
                    continue

                beta = kf.update(current_ret, lag_ret)
                kalman_betas.append(beta)

            if kalman_betas:
                self.kalman_beta_history[col].extend(kalman_betas[-5:])

            if len(self.kalman_beta_history[col]) > self.lookback_period_reg:
                self.kalman_beta_history[col] = self.kalman_beta_history[col][-self.lookback_period_reg:]

            if len(self.kalman_beta_history[col]) > 10:
                kalman_beta_series = pd.Series(self.kalman_beta_history[col])
                rolling_mean = kalman_beta_series.rolling(window=10, min_periods=3).mean()

                if not pd.isna(rolling_mean.iloc[-1]):
                    reg_signal[col] = rolling_mean.iloc[-1] <= self.base_thres_reg

        # ---------- COMBINED SIGNAL ----------
        combined_signal = z_signal & reg_signal

        # Skip only in extremely quiet markets
        market_volatility = recent_volatility.mean()
        avg_signal_strength = zscore[combined_signal].abs().mean() if combined_signal.any() else 0

        if market_volatility < 0.0002 and avg_signal_strength < 0.5:
            combined_signal[:] = False

        bullish_assets = combined_signal[combined_signal].index.tolist()

        # ---------- PORTFOLIO WEIGHTS ----------
        weights = pd.Series(0.0, index=current_prices.index)

        num_bullish = len(bullish_assets)
        if num_bullish > 0:
            if num_bullish == 1:
                weights[bullish_assets] = 1.0
            else:
                z_scores_selected = zscore[bullish_assets].abs()
                weights_values = z_scores_selected + 0.1
                if weights_values.sum() > 0:
                    weights[bullish_assets] = weights_values / weights_values.sum()
                else:
                    weights[bullish_assets] = 1.0 / num_bullish

        return weights
