"""Checks for the rules that decide the savings. Run: python src/test_sizing.py"""
import numpy as np

from right_size import LADDER, band_margins, conformal_margin, risk_reasons, size_for

# Never recommends a larger size, even when demand is above the current size's target.
for cores in LADDER:
    for percent in (0, 1, 10, 35, 50, 70, 71, 90, 100):
        assert size_for(percent, cores) <= cores

# Never picks a size below what the demand needs.
for cores in LADDER:
    for percent in (0, 1, 10, 35, 50, 70):
        new = size_for(percent, cores)
        assert new >= percent / 100 * cores / 0.70

# A VM already at the smallest size gets no change.
assert size_for(0, 2) == 2

# Worked examples: 32.5% of 8 cores is 2.6 cores, which needs 3.7, so 4. Above 70% stays put.
assert size_for(32.5, 8) == 4
assert size_for(90, 8) == 8

# Risk rules.
assert risk_reasons("Delay-insensitive", 8, 4, 30) == []
assert len(risk_reasons("Interactive", 8, 4, 30)) == 1
assert len(risk_reasons("Unknown", 8, 4, 30)) == 1
assert len(risk_reasons("Delay-insensitive", 8, 2, 20)) == 1      # two sizes down
assert len(risk_reasons("Delay-insensitive", 8, 4, 80)) == 1      # peak of 6.4 cores is above 4

# Conformal margin: 90% of 19 errors, with the finite-sample correction, is the 18th smallest.
assert conformal_margin(list(range(1, 20)), [0] * 19) == 18

# Each band gets its own margin, and a margin is never negative.
predicted = np.array([10.0] * 19 + [50.0] * 19)
actual = np.concatenate([predicted[:19] + np.arange(19), predicted[19:] - 5])
assert band_margins(actual, predicted) == {0: 17.0, 2: 0.0}

print("all sizing checks passed")
