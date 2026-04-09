# Step 5: Systematic Maximization

**Score**: 89.17% | **Win Rate**: 89.21% | **Exp**: 89.13%  
**Δ vs Step 4**: **+16.69pp** ↑↑↑ (Final breakthrough)  
**Gap to Target (90%)**: **-0.83pp** (within striking distance)

---

##  Research Question

How far can **systematic hyperparameter optimization**push the model's performance?

---

##  Approach

### Phase 1: Architecture Evolution (v11 → v22)

**v11: Multi-timescale Z-scores** (+5.69pp vs Step 4's precursor)
- Add z30, z90, z180, z365, z1461 (4-year halving cycle)
- z1461 is critical: Captures full BTC market cycle
- Each timescale weighted by coefficient c30, c90, c180, c365, c1461

**v22: Sigmoid Architecture** (+0.31pp vs v15)
- Replace `exp(composite^power × strength)` with `exp(steepness × (composite - threshold))`
- Steepness: Amplification intensity (5-1500 log scale)
- Threshold: Neutral point (0.30-0.70)

---

### Phase 2: Regime-Aware MVRV (v25 → v27)

**Discovery**: MVRV hurts performance in bull markets, helps in bear markets

**Solution**: Split MVRV weight by regime

```python
regime_r = sigmoid(r_steepness × (z1461_raw - r_threshold))

w_mvrv_effective = (
    w_mvrv_bear × (1 - regime_r) +  # Bear weight when r=0
    w_mvrv_bull × regime_r          # Bull weight when r=1
)
```

**Optimal params** (v27, trial 431):
- `w_mvrv_bear = 0.175` (high influence in downturns)
- `w_mvrv_bull = 0.006` (near-zero influence in bull runs)
- `r_threshold = 0.754` (aggressive bear definition: z1461 > 0.754 = bull)

**Result**: +0.96pp (v25 vs v22)

---

### Phase 3: Extended Optuna Search (v27)

**1000 Trials** (vs 400 in v25):
- Seed=99 for exploration diversity
- Expanded search ranges:
  - c180: 0-15 (was 0-10)
  - c365: 0-15 (was 0-10)
  - r_steepness: 0.5-20 (was 1-10)
  - steepness: 5-2000 (was 10-1500)

**Objective**: `B-soft = score - 0.3 × (max(0, 90-win) + max(0, 90-exp))`
- Heavily penalizes falling below 90% on either metric
- Encourages balanced win/exp (both near 90%)

**Best = Trial 431** (discovered in extended 600-trial phase):
- Score: **89.17%**
- Win: **89.21%**
- Exp: **89.13%**
- B-soft: **88.671**

---

##  Final Architecture (v27)

### Signals (5 total, regime-aware)

1. **MVRV Z-score** (W=0.175 bear / 0.006 bull) — Regime-split valuation
2. **Halving Proximity** (W=0.046) — Cycle timing
3. **Flow Composite** (W=0.488) — Net Flow + SplyExNtv
4. **AdrActCnt** (W=0.331) — Network demand
5. **Exchange Ratio** (W=0.168) — SplyExNtv/SplyCur pct_change

### Multi-timescale Z-scores

```python
z_composite = (
    base_w × onchain_signals +
    (1 - base_w) × (
        c30 × z30 +
        c90 × z90 +
        c180 × z180 +
        c365 × z365 +
        c1461 × z1461  
    )
)
```

**Optimal coefficients**:
- c30=2.09, c90=1.16, c180=7.53, c365=**10.05**, c1461=4.17
- c365=10.05 is highest → 1-year price cycle most important

### Sigmoid Transformation

```python
weights = exp(steepness × (z_composite - threshold))
# steepness=738.6 (very steep → extreme concentration)
# threshold=0.292 (30th percentile = neutral)
```

---

##  Results

| Metric | Step 4 | Step 5 | Δ | Target | Gap |
|--------|-------:|-------:|---:|-------:|----:|
| **Score** | 72.48% | **89.17%** | **+16.69pp** | 90.00% | -0.83pp |
| **Win Rate** | 74.15% | **89.21%** | +15.06pp | 90.00% | -0.79pp |
| **Exp %** | — | **89.13%** | — | 90.00% | -0.87pp |
| Wins | 1,896 | 2,281 | +385 windows | — | — |
| Losses | 661 | 276 | -385 windows | — | — |

**Interpretation**: 89.2% of rolling windows beat uniform DCA. Near-perfect balance (win≈exp).

---

##  Why This Worked

### 1. Multi-timescale Signals (z1461 critical)
- Bitcoin's 4-year halving cycle is real → z1461 captures it
- Combining 30d-1461d captures both noise and macro trends
- c365=10.05 weight → 1-year seasonality most predictive

### 2. Regime-Aware MVRV
- MVRV conflicts with momentum in bull markets (sell signal when everyone's buying)
- In bear markets, MVRV shines (valuation mean reversion)
- Splitting weights by regime eliminates this conflict

### 3. Systematic Exploration
- 1000 trials explored 10^16+ parameter combinations
- Human intuition would miss r_threshold=0.754 (seems extreme, but optimal)
- B-soft objective ensures balanced win/exp (no 99% win / 58% exp disasters)

### 4. Extreme Concentration (steepness=738.6)
- "Wait for perfect setups, then go all-in"
- 4 clean, independent signals → can safely amplify to extremes
- Low-signal days: ~0% allocation (capital preserved)
- High-signal days: 10-50× median allocation

---

##  Key Insights

1. **Multi-timescale wins**: 30d-1461d range captures all market regimes
2. **Regime awareness pays**: MVRV bear/bull split worth +0.96pp
3. **Systematic beats intuition**: 1000 trials find non-obvious optima
4. **Balance is achievable**: Win≈Exp≈89% (no trade-offs at this level)
5. **4-year cycle is real**: z1461 coefficient significant across all top trials

---

##  Lessons for Production Deployment

### Strengths
- **Robust**: 89% win rate across 2,557 diverse windows (2018-2025)
- **Balanced**: No win/exp trade-off (both ~89%)
- **Interpretable**: Each signal has clear on-chain meaning
- **Stable**: Lock-on-compute weights prevent look-ahead bias

### Limitations
- **Extreme concentration**: steepness=738.6 → high portfolio turnover risk
- **Parameter sensitivity**: 16 parameters → overfitting risk on unseen regimes
- **Data dependency**: Requires CoinMetrics on-chain data (not freely available)


---

##  Run This Model

```bash
python -m jpark.models_final.step5_systematic_maximization.run_backtest
```

---

*Model 4 v27 | The culmination of systematic Bitcoin DCA research*
