# Step 2: Signal Independence Lesson

**Score**: 53.66% | **Win Rate**: 65.86% | **Exp**: 39.58%  
**Δ vs Step 1**: **-0.70pp** ↓

---

##  Research Question

Does adding NVT (Network Value to Transactions) improve performance beyond the baseline 4 signals?

---

##  Hypothesis

**NVT Proxy** = Market Cap / On-chain Transaction Volume

- Measures network utility: high NVT = overvalued relative to usage
- Should be **independent** of MVRV (network utility vs realized value)
- Low NVT + volume expansion = durable bottom signal

**EDA Evidence**:
- NVT proxy median: 102.97
- NVT < 50 periods: 1,048 undervaluation days
- Volume expansion during low NVT → sustainable rallies

---

##  Implementation

### 5 Signals (4 baseline + 1 new)

1. **MVRV Z-score** (W=0.30, ↓ from 0.35)
2. **MA200 Log Deviation** (W=0.30, ↓ from 0.35)
3. **Halving Proximity** (W=0.15, unchanged)
4. **Net Exchange Flow** (W=0.15, unchanged)
5. **NVT Proxy Z-score** (W=0.10) ★ NEW

**Trade-off**: Reduced MVRV and MA200 weights to make room for NVT.

### NVT Signal Construction

```python
# NVT Proxy = CapMrktCurUSD / TxTfrValAdjUSD
nvt_raw = df["CapMrktCurUSD"] / df["TxTfrValAdjUSD"]
nvt_smooth = nvt_raw.rolling(30, min_periods=5).mean()
nvt_z = rolling_zscore(nvt_smooth, window=365, clip=3.0)

# Inverted: low NVT → high signal
sig_nvt = sigmoid(-nvt_z)
```

---

##  Results

| Metric | Step 1 (Baseline) | Step 2 (+ NVT) | Δ |
|--------|------------------:|---------------:|---:|
| Score | 54.36% | 53.66% | **-0.70pp** |
| Win Rate | 68.13% | 65.86% | **-2.27pp** |
| Exp Decay % | 40.59% | 39.58% | -1.01pp |
| Wins | 1,742 | 1,684 | **-58 windows** |
| Losses | 815 | 873 | +58 windows |

**Interpretation**: Adding NVT **degraded** performance across all metrics.

---

##  Analysis: Why Did NVT Fail?

### Hypothesis 1: Signal Redundancy
Despite theoretical independence, NVT may overlap with MVRV in practice:
- Both measure "value vs fundamentals"
- MVRV: price vs realized value
- NVT: price vs transaction utility
- Correlation may exist during extreme regimes

### Hypothesis 2: Weight Dilution
Reducing MVRV/MA200 from 0.35 → 0.30 weakened the strongest signals:
- MVRV and MA200 had r=0.99 price correlation (very powerful)
- NVT's 0.10 weight insufficient to compensate for their dilution

### Hypothesis 3: Signal Conflict
NVT and MVRV may provide **contradictory** signals:
- MVRV low (buy) + NVT high (sell) → cancellation
- Mixed signals reduce conviction → more losses

---

##  Key Insights

1. **More signals ≠ better performance**
2. **Signal independence must be verified empirically**, not just theoretically
3. **Diluting strong signals is costly**: MVRV/MA200 are core, don't weaken
4. **Addition requires subtraction elsewhere**: Total weight = 1.0 constraint

---

##  Lesson Learned

### The "NVT Experiment" Teaches:

**Before adding a new signal, ask:**
1. Is it truly independent? (Check correlation matrix)
2. Does it add unique information? (Ablation study)
3. Is the weight allocation optimal? (Grid search)

**Takeaway**: This "failure" is valuable—it validates that the baseline's 4 signals were already well-chosen. Simplicity won.

---


##  Run This Model

```bash
python -m jpark.models_final.step2_signal_independence_lesson.run_backtest
```

---

*Model 2 ntv | The importance of signal independence*
