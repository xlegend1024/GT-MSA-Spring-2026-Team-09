# Overview

## Problem

We ask a practical capital allocation question: can a data-driven Bitcoin accumulation strategy deploy a fixed budget more efficiently than uniform DCA while remaining systematic, interpretable, and long-only? The goal is not to predict every price move. The goal is to improve accumulation timing so that more capital is concentrated in higher-conviction buying conditions without abandoning the discipline that makes DCA useful in the first place.

## Baseline

Uniform DCA is the benchmark because it is realistic, transparent, and already strong. The first baseline-foundation strategy showed that even a simple interpretable on-chain rule could improve on that benchmark, so any later dynamic strategy had to justify its extra complexity by improving on a rule that was already competitive.

All results were evaluated under the same long-only, fixed-budget rolling backtest against uniform DCA using score, win rate, and exp-decay percentile.

## Final Result

Across the six-step strategy progression, the final strategy reached a 98.08% score, a 96.25% win rate, and a 99.92% exp-decay percentile in rolling one-year evaluations against uniform DCA. The path to that result was not a straight line. We began with an interpretable on-chain baseline, learned in Step 2 that simply adding NVT did not help, improved in Step 3 by introducing network-demand information, gained much more in Step 4 by removing redundant structure and strengthening the flow signal, and then pushed the cleaned architecture further in Step 5 through systematic search, regime-aware weighting, and multi-timescale price structure.

The final jump came in Step 6, where the same strong signal set was expressed through direct softmax normalization instead of sequential allocation. That allocation change produced the largest remaining improvement and brought the strategy close to the practical ceiling observed in this evaluation setting.


## Conclusion

The results show that a dynamic, interpretable, long-only Bitcoin accumulation rule can improve meaningfully on uniform DCA under the same rolling-budget constraints.

The main conclusion is that the final edge did not come from endlessly expanding the feature set. It came from keeping the signals that added independent information, removing redundant components such as MA200 when they no longer helped, strengthening the architecture around flow and demand, and finally correcting how conviction was translated into portfolio weights.
