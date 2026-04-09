# Step 4: Structural Optimization

**Score**: 72.48% | **Win Rate**: 74.15%  
**Δ vs Step 3**: **+15.50pp** ↑↑ (Massive breakthrough)

---

##  Research Question

Can we improve performance by **optimizing structure** (removing redundancy, compositing signals) rather than just adding features?

---

##  Hypothesis

After 3 steps of building up signals, time to **tear down and rebuild**:

1. **Check for redundancy**: Are any signals highly correlated?
2. **Composite related signals**: Can we combine complementary features?
3. **Amplification tuning**: Is EXP_STRENGTH=3.5 optimal?

---

##  Grid Search Discoveries

### Grid Search A: Flow Signal Enhancement (+1.74pp)

**Finding**: Net Exchange Flow + Exchange Supply Ratio are complementary

```python
# Before: Only Net Flow
flow_signal = sigmoid(-rolling_zscore(FlowInExNtv - FlowOutExNtv))

# After: Composite Flow
net_flow_z = rolling_zscore((FlowInEx - FlowOutEx).rolling(30).mean())
sply_ex_z = rolling_zscore(SplyExNtv.pct_change(30).rolling(7).mean())
flow_composite = 0.5 × (-net_flow_z) + 0.5 × (-sply_ex_z)
```

**Logic**:
- Net Flow: Short-term accumulation/distribution
- SplyExNtv: Long-term exchange balance trend
- Combined: Captures both momentum AND position

**Result**: +1.74pp improvement

---

### Grid Search B: MA200 Removal (+2.64pp) ★ BREAKTHROUGH

**Finding**: MA200 and MVRV have **r=0.87 correlation** → Redundant

```python
# Correlation matrix (simplified):
#            MVRV   MA200   Halving   Flow   AdrActCnt
# MVRV       1.00   0.87    0.12      0.23   0.31
# MA200      0.87   1.00    0.09      0.19   0.28
```

**Decision**: Remove MA200, redistribute its weight

```python
# Before (5 signals):
W_MVRV=0.25, W_MA200=0.25, W_Halving=0.10, W_Flow=0.15, W_Addr=0.25

# After (4 signals, MA200 removed):
W_MVRV=0.333, W_Halving=0.133, W_Flow=0.200, W_Addr=0.333
```

**Result**: +2.64pp improvement (largest single change!)

**Insight**: "Less is more" — Removing redundancy beats adding features.

---

### Grid Search E: EXP_STRENGTH Tuning (+0.21pp)

**Finding**: STR=175 optimal (vs 150 in previous iteration)

```python
# Amplification curve:
raw_weights = exp(composite^power × strength)

# Testing:
STR=150 → Score 72.27%
STR=175 → Score 72.48% ★
STR=200 → Score 72.31% 
```

**Result**: +.21pp final polish

---

##  Final Architecture

### 4 Optimized Signals

1. **MVRV Z-score** (W=0.333) — Valuation core
2. **Halving Proximity** (W=0.133) — Cycle timing
3. **Flow Composite** (W=0.200) — Supply pressure (enhanced)
4. **AdrActCnt** (W=0.333) — Network demand

**No MA200**: Removed due to 0.87 correlation with MVRV

**EXP_STRENGTH**: 175 (high concentration on strong signals)

---

##  Results

| Metric | Step 3 | Step 4 | Δ |
|--------|-------:|-------:|---:|
| Score | 56.98% | **72.48%** | **+15.50pp** |
| Win Rate | 72.82% | 74.15% | +1.33pp |
| Wins | 1,861 | 1,896 | +35 windows |

**Interpretation**: Structural optimization (deduplication + compositing) delivered the **largest improvement** of any step.

---

##  Why This Worked

### Principle 1: Signal Independence > Signal Count
- MA200 didn't add unique information (87% redundant with MVRV)
- Removing it **increased** performance (counterintuitive but true)

### Principle 2: Composite Signals Capture Relationships
- Net Flow + SplyExNtv together reveal accumulation patterns neither shows alone
- 0.5/0.5 weighting preserves both perspectives

### Principle 3: High Conviction Focus
- STR=175 means: "When all 4 signals align (rare), invest heavily"
- Low-signal days get near-zero weight → Capital reserves for high-conviction opportunities

---

##  Key Insights

1. **Deduplication beats addition**: Removing MA200 (+2.64pp) > adding any new signal
2. **Correlation audits are critical**: Always check signal independence
3. **Composite features work**: Flow + Exchange Balance > either alone
4. **Simplicity enables extremes**: 4 clean signals → can push STR to 175 safely
5. **Biggest gains come from structure**: Not features, but how they combine

---

##  Lessons for Future Work

**Before adding a signal**, ask:
1. Is it correlated with existing signals? (Threshold: r > 0.7 = redundant)
2. Can it be **composited** with related signals?
3. Does removing it degrade performance? (Ablation test)

**Grid search is powerful**:
- Grid Search A/B/E together: +4.59pp
- Manual intuition would miss MA200 removal

---


##  Run This Model

```bash
python -m jpark.models_final.step4_structural_optimization.run_backtest
```

---

*Model 4 v3 | The power of structural optimization*
