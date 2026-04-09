# Step 1: Baseline Foundation

**Score**: 54.36% | **Win Rate**: 68.13% | **Exp**: 40.59%

---

##  Research Question

Can EDA-validated on-chain signals alone beat uniform Dollar-Cost Averaging (DCA)?

---

##  Approach

### 4 Core Signals (EDA-Justified)

1. **MVRV Z-score** (W=0.35) ★★★
   - Market Cap / Realized Cap normalized z-score (365-day rolling)
   - Measures valuation regime: low MVRV = undervalued = buy signal
   - EDA: MVRV < 1.0 periods show 1,048 undervaluation days

2. **MA200 Log Deviation** (W=0.35) ★★★
   - `log(price / MA200)` clipped to [-1, 1]
   - Trend regime: price below MA200 = buy signal
   - EDA: Price correlation r=0.99 with MA200

3. **Halving Proximity** (W=0.15) ★★
   - `exp(-|days_to_nearest_halving| / 180)` 
   - Cycle timing: closer to halving = higher weight
   - Known halving dates: 2012, 2016, 2020, 2024

4. **Net Exchange Flow Z-score** (W=0.15) ★★
   - `(FlowInExNtv - FlowOutExNtv)` 30-day MA, inverted
   - Supply pressure: outflow (negative net) = accumulation = buy signal
   - EDA: Flow correlation r=-0.42 with price

### Architecture

```python
# Each signal → sigmoid [0, 1]
sig_mvrv = sigmoid(-mvrv_z)      # Inverted: low MVRV → high signal
sig_ma200 = sigmoid(-ma200_dev)  # Inverted: below MA → high signal
sig_halving = halving_proximity  # Already [0, 1]
sig_flow = sigmoid(-flow_z)      # Inverted: outflow → high signal

# Weighted composite
composite = (
    W_MVRV * sig_mvrv +
    W_MA200 * sig_ma200 +
    W_HALVING * sig_halving +
    W_FLOW * sig_flow
)

# Exponential amplification
raw_weights = exp(composite × EXP_STRENGTH)  # EXP_STRENGTH = 2.0
daily_weights = raw_weights / sum(raw_weights)  # Normalize to 1.0
```

**Key Parameter**: `EXP_STRENGTH = 2.0`
- Higher values concentrate investment on high-signal days
- Tested in Model 1 strong: 2.0 → 3.5 yields +1.67pp

---

##  Results

| Metric | Value |
|--------|------:|
| Score | 54.36% |
| Win Rate | 68.13% |
| Exp Decay % | 40.59% |
| Mean Excess | +0.58 sats/$ |
| Median Excess | +1.07 sats/$ |
| Wins | 1,742 / 2,557 windows |
| Losses | 815 / 2,557 windows |

**Interpretation**: 68% of rolling 1-year windows beat uniform DCA.

---

##  Key Insights

1. **Yes, it works**: Simple signal combination beats uniform DCA 68% of the time
2. **Signal synergy**: 4 independent signals (valuation + trend + cycle + supply) complement each other
3. **No ML needed**: Rule-based approach with clear interpretability
4. **Foundation established**: This becomes the baseline for all future experiments

---

##  Why This Baseline Matters

- **Interpretable**: Every signal has clear on-chain/economic meaning
- **EDA-grounded**: All signals validated in exploratory data analysis
- **Simple**: No overfitting, no hyperparameter tuning (yet)
- **Replicable**: Static parameters, public halving dates, no look-ahead bias

---


##  Run This Model

```bash
python -m jpark.models_final.step1_baseline_foundation.run_backtest
```

---

*Model 1 baseline | EDA-validated 4-signal strategy*
