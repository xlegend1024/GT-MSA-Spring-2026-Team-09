# Conclusion

We built an interpretable Bitcoin accumulation strategy that outperformed uniform DCA under the same long-only, fixed-budget constraints. Across the six-step progression, the final strategy reached a 98.08% score, a 96.25% win rate, and a 99.92% exp-decay percentile, showing that dynamic allocation can improve accumulation quality without relying on black-box complexity.

This result did not come from endlessly adding more features. It came from identifying which signals added independent information, removing redundant structure when it stopped helping, and improving how conviction was translated into portfolio weights. The strongest late-stage gains came first from simplifying the architecture around flow and demand, and then from correcting the allocation mechanism itself.

## Final Takeaway

The main lesson of the analysis is that Bitcoin accumulation can be improved as a decision-system problem rather than as a pure forecasting problem. Clean data, interpretable signals, and disciplined allocation logic mattered more than feature count. By the end of the study, the limiting factor was no longer signal discovery. It was the mechanism used to convert signal strength into capital deployment.