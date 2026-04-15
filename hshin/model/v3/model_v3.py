"""V3 Model — Q-Learning DCA Strategy.

Uses tabular Q-Learning (with Dyna-Q) to learn optimal DCA multipliers
from discretized market states.

State: (mvrv_zone, ma_regime, flow_state) → 5 × 2 × 3 = 30 states
Action: DCA multiplier level → 5 actions [0.25x, 0.5x, 1.0x, 1.5x, 2.5x]
Reward: Sats-per-dollar improvement vs uniform DCA on next day
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from template.model_development_template import (
    allocate_sequential_stable,
    _clean_array,
)
from hshin.model.v1.model_v1 import load_poly_indices, _poly_change_signal

W_POLY = 0.15

PRICE_COL = "PriceUSD_coinmetrics"
MVRV_COL = "CapMVRVCur"
FLOW_IN_COL = "FlowInExNtv"
FLOW_OUT_COL = "FlowOutExNtv"
MA_WINDOW = 200

# State discretization
MVRV_BINS = [-np.inf, 0.8, 1.0, 2.0, 3.5, np.inf]  # 5 zones
MVRV_LABELS = [0, 1, 2, 3, 4]  # deep_value, value, neutral, caution, danger

MA_BINS = 2   # 0=below MA200, 1=above MA200

FLOW_BINS = [-np.inf, -0.3, 0.3, np.inf]  # 3 zones
FLOW_LABELS = [0, 1, 2]  # outflow, neutral, inflow

NUM_MVRV = 5
NUM_MA = 2
NUM_FLOW = 3
NUM_STATES = NUM_MVRV * NUM_MA * NUM_FLOW  # 30

# Actions: DCA multiplier levels
ACTIONS = np.array([0.25, 0.5, 1.0, 1.5, 2.5])
NUM_ACTIONS = len(ACTIONS)


class DCAQLearner:
    """Q-Learning agent for DCA multiplier selection."""

    def __init__(
        self,
        alpha: float = 0.15,
        gamma: float = 0.9,
        rar: float = 0.3,
        radr: float = 0.995,
        dyna: int = 100,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.Q = np.zeros((NUM_STATES, NUM_ACTIONS))
        self.experience = []
        self.s = 0
        self.a = 2  # default: 1.0x

    def _choose_action(self, s: int) -> int:
        if np.random.random() < self.rar:
            return np.random.randint(0, NUM_ACTIONS)
        return int(np.argmax(self.Q[s, :]))

    def querysetstate(self, s: int) -> int:
        self.s = s
        self.a = self._choose_action(s)
        return self.a

    def query(self, s_prime: int, r: float) -> int:
        old_q = self.Q[self.s, self.a]
        self.Q[self.s, self.a] = (
            (1 - self.alpha) * old_q
            + self.alpha * (r + self.gamma * np.max(self.Q[s_prime, :]))
        )

        if self.dyna > 0:
            self.experience.append((self.s, self.a, s_prime, r))
            indices = np.random.randint(0, len(self.experience), size=self.dyna)
            for idx in indices:
                s, a, sp, rr = self.experience[idx]
                self.Q[s, a] = (
                    (1 - self.alpha) * self.Q[s, a]
                    + self.alpha * (rr + self.gamma * np.max(self.Q[sp, :]))
                )

        self.rar *= self.radr
        self.s = s_prime
        self.a = self._choose_action(s_prime)
        return self.a

    def get_multiplier(self, action: int) -> float:
        return ACTIONS[action]

    def get_policy_multiplier(self, s: int) -> float:
        """Get greedy (no exploration) multiplier for a state."""
        return ACTIONS[int(np.argmax(self.Q[s, :]))]

    def get_inverse_multiplier(self, s: int) -> float:
        """Get inverse of greedy multiplier: 1/original.
        If Q learned to buy 2.5x in state s (bad idea), inverse is 0.4x.
        """
        original = self.get_policy_multiplier(s)
        return 1.0 / original if original > 0 else 1.0


# =============================================================================
# State Discretization
# =============================================================================

def discretize_state(mvrv_raw: float, price_vs_ma: float, flow_signal: float) -> int:
    """Convert continuous features to discrete state index."""
    mvrv_zone = np.searchsorted(MVRV_BINS[1:], mvrv_raw)
    mvrv_zone = min(mvrv_zone, NUM_MVRV - 1)

    ma_regime = 1 if price_vs_ma >= 0 else 0

    flow_zone = np.searchsorted(FLOW_BINS[1:], flow_signal)
    flow_zone = min(flow_zone, NUM_FLOW - 1)

    return int(mvrv_zone * NUM_MA * NUM_FLOW + ma_regime * NUM_FLOW + flow_zone)


# =============================================================================
# Feature Engineering
# =============================================================================

def precompute_features(df: pd.DataFrame, use_poly: bool = False) -> pd.DataFrame:
    """Compute features for state discretization."""
    price = df[PRICE_COL].loc["2010-07-18":].copy()
    ma200 = price.rolling(MA_WINDOW, min_periods=MA_WINDOW // 2).mean()

    with np.errstate(divide="ignore", invalid="ignore"):
        price_vs_ma = ((price / ma200) - 1).clip(-1, 1).fillna(0)

    if MVRV_COL in df.columns:
        mvrv_raw = df[MVRV_COL].loc[price.index].fillna(1.5)
    else:
        mvrv_raw = pd.Series(1.5, index=price.index)

    if FLOW_IN_COL in df.columns and FLOW_OUT_COL in df.columns:
        flow_in = df[FLOW_IN_COL].loc[price.index].fillna(0)
        flow_out = df[FLOW_OUT_COL].loc[price.index].fillna(0)
        net_flow = flow_in - flow_out
        flow_z = (net_flow - net_flow.rolling(90, min_periods=30).mean()) / \
                 net_flow.rolling(90, min_periods=30).std().replace(0, 1)
        flow_signal = flow_z.clip(-3, 3).fillna(0)
    else:
        flow_signal = pd.Series(0.0, index=price.index)

    features = pd.DataFrame({
        PRICE_COL: price,
        "mvrv_raw": mvrv_raw,
        "price_vs_ma": price_vs_ma.shift(1).fillna(0),
        "mvrv_raw_lagged": mvrv_raw.shift(1).fillna(1.5),
        "flow_signal": flow_signal.shift(1).fillna(0),
    }, index=price.index)

    if use_poly:
        poly_indices = load_poly_indices()
        crypto_z = _poly_change_signal(poly_indices["crypto"])
        features["poly_crypto"] = crypto_z.reindex(price.index, fill_value=0).shift(1).fillna(0)
        us_z = _poly_change_signal(poly_indices["us_affairs"])
        features["poly_us_affairs"] = us_z.reindex(price.index, fill_value=0).shift(1).fillna(0)

    return features.fillna(0)


# =============================================================================
# Training
# =============================================================================

def train_qlearner(
    features_df: pd.DataFrame,
    train_start: str,
    train_end: str,
    n_epochs: int = 50,
    alpha: float = 0.15,
    gamma: float = 0.9,
    dyna: int = 100,
) -> DCAQLearner:
    """Train Q-Learner on historical data."""
    train_data = features_df.loc[train_start:train_end].copy()
    if len(train_data) < 200:
        raise ValueError(f"Not enough training data: {len(train_data)} rows")

    prices = train_data[PRICE_COL].values
    mvrv = train_data["mvrv_raw_lagged"].values
    ma = train_data["price_vs_ma"].values
    flow = train_data["flow_signal"].values

    agent = DCAQLearner(alpha=alpha, gamma=gamma, rar=0.5, radr=0.995, dyna=dyna)

    for epoch in range(n_epochs):
        s = discretize_state(mvrv[0], ma[0], flow[0])
        agent.querysetstate(s)

        for t in range(1, len(prices)):
            # Reward: did the multiplier help?
            # If we bought more (high multiplier) and price went up → bad (paid more)
            # If we bought more and price went down → good (got more sats)
            price_change = (prices[t] - prices[t-1]) / prices[t-1]
            current_action = agent.a
            multiplier = ACTIONS[current_action]

            # Reward = extra sats gained by using this multiplier vs 1.0x
            uniform_sats = 1.0 / prices[t]
            actual_sats = multiplier / prices[t]
            # Normalize reward: how much better than uniform
            reward = (actual_sats - uniform_sats) / uniform_sats if uniform_sats > 0 else 0

            # Penalize buying high (price went up after we bought more)
            if multiplier > 1.0 and price_change > 0.02:
                reward -= 0.1 * (multiplier - 1.0) * price_change
            # Reward buying low (price relative to recent)
            if multiplier > 1.0 and price_change < -0.02:
                reward += 0.1 * (multiplier - 1.0) * abs(price_change)

            s_prime = discretize_state(mvrv[t], ma[t], flow[t])
            agent.query(s_prime, reward)

        if (epoch + 1) % 10 == 0:
            logging.info(f"  Epoch {epoch+1}/{n_epochs}, rar={agent.rar:.4f}")

    return agent


# =============================================================================
# Weight Computation using trained agent
# =============================================================================

def compute_weights_with_agent(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    agent: DCAQLearner,
) -> pd.Series:
    """Compute DCA weights using trained Q-Learning agent's greedy policy."""
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")

    # Extend features if needed
    missing = full_range.difference(features_df.index)
    if len(missing) > 0:
        placeholder = pd.DataFrame({col: 0.0 for col in features_df.columns}, index=missing)
        if "mvrv_raw" in placeholder.columns:
            placeholder["mvrv_raw"] = 1.5
        if "mvrv_raw_lagged" in placeholder.columns:
            placeholder["mvrv_raw_lagged"] = 1.5
        features_df = pd.concat([features_df, placeholder]).sort_index()

    df = features_df.loc[start_date:end_date]
    if df.empty:
        return pd.Series(dtype=float)

    n = len(df)
    base = np.ones(n) / n

    # Get multiplier for each day from agent's learned policy
    multipliers = np.ones(n)
    mvrv = df["mvrv_raw_lagged"].values if "mvrv_raw_lagged" in df.columns else np.full(n, 1.5)
    ma = df["price_vs_ma"].values if "price_vs_ma" in df.columns else np.zeros(n)
    flow = df["flow_signal"].values if "flow_signal" in df.columns else np.zeros(n)

    for i in range(n):
        s = discretize_state(mvrv[i], ma[i], flow[i])
        multipliers[i] = agent.get_policy_multiplier(s)

    # Polymarket adjustment
    has_poly = "poly_crypto" in df.columns or "poly_us_affairs" in df.columns
    if has_poly:
        poly_combined = np.zeros(n)
        poly_count = 0
        if "poly_crypto" in df.columns:
            poly_combined += _clean_array(df["poly_crypto"].values)
            poly_count += 1
        if "poly_us_affairs" in df.columns:
            poly_combined += _clean_array(df["poly_us_affairs"].values)
            poly_count += 1
        if poly_count > 0:
            poly_combined /= poly_count
        poly_adj = np.clip(1.0 + poly_combined * W_POLY, 0.7, 1.3)
        multipliers *= poly_adj

    raw = base * multipliers
    weights = allocate_sequential_stable(raw, n)
    return pd.Series(weights, index=df.index).reindex(full_range, fill_value=0.0)


def compute_weights_with_agent_inverse(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    agent: DCAQLearner,
) -> pd.Series:
    """Compute DCA weights using INVERSE of agent's learned policy."""
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")

    missing = full_range.difference(features_df.index)
    if len(missing) > 0:
        placeholder = pd.DataFrame({col: 0.0 for col in features_df.columns}, index=missing)
        if "mvrv_raw" in placeholder.columns:
            placeholder["mvrv_raw"] = 1.5
        if "mvrv_raw_lagged" in placeholder.columns:
            placeholder["mvrv_raw_lagged"] = 1.5
        features_df = pd.concat([features_df, placeholder]).sort_index()

    df = features_df.loc[start_date:end_date]
    if df.empty:
        return pd.Series(dtype=float)

    n = len(df)
    base = np.ones(n) / n

    multipliers = np.ones(n)
    mvrv = df["mvrv_raw_lagged"].values if "mvrv_raw_lagged" in df.columns else np.full(n, 1.5)
    ma = df["price_vs_ma"].values if "price_vs_ma" in df.columns else np.zeros(n)
    flow = df["flow_signal"].values if "flow_signal" in df.columns else np.zeros(n)

    for i in range(n):
        s = discretize_state(mvrv[i], ma[i], flow[i])
        multipliers[i] = agent.get_inverse_multiplier(s)

    # Polymarket adjustment
    has_poly = "poly_crypto" in df.columns or "poly_us_affairs" in df.columns
    if has_poly:
        poly_combined = np.zeros(n)
        poly_count = 0
        if "poly_crypto" in df.columns:
            poly_combined += _clean_array(df["poly_crypto"].values)
            poly_count += 1
        if "poly_us_affairs" in df.columns:
            poly_combined += _clean_array(df["poly_us_affairs"].values)
            poly_count += 1
        if poly_count > 0:
            poly_combined /= poly_count
        poly_adj = np.clip(1.0 + poly_combined * W_POLY, 0.7, 1.3)
        multipliers *= poly_adj

    raw = base * multipliers
    weights = allocate_sequential_stable(raw, n)
    return pd.Series(weights, index=df.index).reindex(full_range, fill_value=0.0)
