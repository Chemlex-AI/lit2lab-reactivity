"""Static checks of teacher likelihoods and reliability estimates."""

import numpy as np

from .model import noisy_teacher_prob
from .prior import estimate_teacher_noise_posterior


def run_sanity_tests(verbose: bool = True) -> None:
    """Check teacher likelihood and noise estimates; main asserts data isolation."""
    p = np.linspace(0.0, 1.0, 101)

    # Perfect teacher: pseudo and GT likelihoods match.
    assert np.allclose(noisy_teacher_prob(p, 1.0, 1.0), p, atol=1e-12), "Test 1 (perfect teacher) failed."

    # Random teacher: pseudo probability is always 0.5.
    assert np.allclose(noisy_teacher_prob(p, 0.5, 0.5), 0.5, atol=1e-12), "Test 2 (random teacher) failed."

    # Asymmetric teacher noise.
    assert abs(float(noisy_teacher_prob(1.0, 0.9, 0.6)) - 0.9) < 1e-12, "Test 3 (p=1) failed."
    assert abs(float(noisy_teacher_prob(0.0, 0.9, 0.6)) - 0.4) < 1e-12, "Test 3 (p=0) failed."

    # Beta(2,2) smoothing with TP=8, FN=2, TN=30, FP=10.
    y_gt = np.array([1] * 10 + [0] * 40)
    y_pseudo = np.array([1] * 8 + [0] * 2 + [1] * 10 + [0] * 30)
    cal = estimate_teacher_noise_posterior(y_gt, y_pseudo, alpha0=2.0, beta0=2.0)
    assert (cal["tp"], cal["fn"], cal["fp"], cal["tn"]) == (8, 2, 10, 30), "Teacher reliability confusion counts wrong."
    assert abs(cal["sensitivity_mean"] - (2 + 8) / (2 + 2 + 8 + 2)) < 1e-12
    assert abs(cal["specificity_mean"] - (2 + 30) / (2 + 2 + 30 + 10)) < 1e-12
    assert abs(cal["youden_j"] - (cal["sensitivity_mean"] + cal["specificity_mean"] - 1.0)) < 1e-12

    # Noise estimates stay finite with single-class GT.
    degenerate = estimate_teacher_noise_posterior(np.zeros(5), np.zeros(5), alpha0=2.0, beta0=2.0)
    assert np.isfinite(degenerate["sensitivity_mean"]) and np.isfinite(degenerate["specificity_mean"])

    # Skip missing pseudo labels.
    with_nan = estimate_teacher_noise_posterior(
        np.array([1, 0, 1]), np.array([1.0, np.nan, 0.0]), alpha0=2.0, beta0=2.0
    )
    assert with_nan["n_calibration"] == 2 and with_nan["n_dropped_missing_pseudo"] == 1

    if verbose:
        print("[SANITY] reliability adjust warm prior checks passed.")

