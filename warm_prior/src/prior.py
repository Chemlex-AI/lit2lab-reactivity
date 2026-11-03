"""Reliability adjust warm prior."""

from typing import Any, Dict, Optional, Sequence

import jax.numpy as jnp
import numpy as np
import pandas as pd

from .model import model, run_inference, samples_to_posterior_prior
from .utils import save_named_posterior_prior


def estimate_teacher_noise_posterior(
    y_gt: Sequence[int],
    y_pseudo: Sequence[float],
    alpha0: float = 2.0,
    beta0: float = 2.0,
) -> Dict[str, Any]:
    """Estimate teacher noise using only initial random GT; skip missing pseudo labels.
    Return means of Beta(alpha0+TP, beta0+FN) and Beta(alpha0+TN, beta0+FP)
    for sensitivity and specificity, respectively.
    """
    y_gt = np.asarray(y_gt).astype(float).reshape(-1)
    y_pseudo = np.asarray(y_pseudo).astype(float).reshape(-1)
    if y_gt.shape != y_pseudo.shape:
        raise ValueError("y_gt and y_pseudo must have the same shape for reliability estimation.")

    valid = ~np.isnan(y_pseudo)
    n_dropped = int((~valid).sum())
    g = y_gt[valid].astype(int)
    p = y_pseudo[valid].astype(int)

    tp = int(((p == 1) & (g == 1)).sum())
    fp = int(((p == 1) & (g == 0)).sum())
    tn = int(((p == 0) & (g == 0)).sum())
    fn = int(((p == 0) & (g == 1)).sum())

    sens_alpha = alpha0 + tp
    sens_beta = beta0 + fn
    spec_alpha = alpha0 + tn
    spec_beta = beta0 + fp

    sensitivity_mean = sens_alpha / (sens_alpha + sens_beta)
    specificity_mean = spec_alpha / (spec_alpha + spec_beta)

    n_pos = tp + fn
    n_neg = tn + fp
    weak_positive = n_pos < 5
    weak_negative = n_neg < 5
    if weak_positive:
        print(
            f"[WARN] weak reliability evidence: only {n_pos} initial GT-positive reactions; "
            f"sensitivity is dominated by the Beta({alpha0}, {beta0}) smoothing prior."
        )
    if weak_negative:
        print(
            f"[WARN] weak reliability evidence: only {n_neg} initial GT-negative reactions; "
            f"specificity is dominated by the Beta({alpha0}, {beta0}) smoothing prior."
        )

    out = {
        "n_calibration": int(valid.sum()),
        "n_dropped_missing_pseudo": n_dropped,
        "n_gt_positive": n_pos,
        "n_gt_negative": n_neg,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "alpha0": float(alpha0),
        "beta0": float(beta0),
        "sens_alpha": float(sens_alpha),
        "sens_beta": float(sens_beta),
        "spec_alpha": float(spec_alpha),
        "spec_beta": float(spec_beta),
        "sensitivity_mean": float(sensitivity_mean),
        "specificity_mean": float(specificity_mean),
        "youden_j": float(sensitivity_mean + specificity_mean - 1.0),
        "weak_positive_evidence": bool(weak_positive),
        "weak_negative_evidence": bool(weak_negative),
    }
    return out



def build_pseudo_prior(
    arm: str,
    rng_key,
    X_pool_all,
    df_pool_all: pd.DataFrame,
    pseudo_train_mask: np.ndarray,
    pseudo_label_col: str,
    args,
    teacher_sensitivity: Optional[float],
    teacher_specificity: Optional[float],
    ckpt_dir: str,
) -> Optional[Dict[str, jnp.ndarray]]:
    """Fit pseudo labels once and moment-match a fixed warm prior.
    Naive uses a clean-label likelihood; RA uses a noise-aware likelihood.
    """
    if arm == "vanilla":
        return None

    train_indices = np.where(pseudo_train_mask)[0]
    if len(train_indices) == 0:
        raise RuntimeError(
            "No pseudo-label rows available for prior construction after excluding the "
            "initial random GT set."
        )

    train_x = X_pool_all[train_indices, ...]
    train_y_np = (
        df_pool_all.iloc[train_indices][pseudo_label_col]
        .to_numpy(dtype=np.float32)
        .astype(np.int16)
    )
    train_y = jnp.asarray(train_y_np, dtype=jnp.int16)

    if arm == "naive_warm":
        label_mode = "gt"
        sens_used: Optional[float] = 1.0
        spec_used: Optional[float] = 1.0
        model_sens: Optional[float] = None
        model_spec: Optional[float] = None
    elif arm == "reliability_adjust_warm":
        if teacher_sensitivity is None or teacher_specificity is None:
            raise RuntimeError("reliability adjust warm prior requires teacher reliability estimates before prior construction.")
        label_mode = "pseudo"
        sens_used = float(teacher_sensitivity)
        spec_used = float(teacher_specificity)
        model_sens = sens_used
        model_spec = spec_used
    else:
        raise ValueError(f"Unknown arm `{arm}`.")

    print("\n" + "=" * 80)
    print(
        f"[PSEUDO PRIOR] arm={arm} label_mode={label_mode} N={len(train_y_np)} "
        f"(pos={(train_y_np == 1).sum()}, neg={(train_y_np == 0).sum()}) "
        f"sensitivity={sens_used:.4f} specificity={spec_used:.4f}"
    )

    samples = run_inference(
        model,
        rng_key,
        train_x,
        train_y,
        args.hidden_dim,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        posterior_prior=None,
        posterior_prior_scale_floor=args.posterior_prior_scale_floor,
        label_mode=label_mode,
        teacher_sensitivity=model_sens,
        teacher_specificity=model_spec,
    )

    pseudo_prior = samples_to_posterior_prior(
        samples,
        scale_floor=args.posterior_prior_scale_floor,
        scale_mult=args.posterior_prior_scale_mult,
    )
    assert not any(k.startswith("prob__") for k in pseudo_prior), "`prob` leaked into the warm prior."

    prior_path = save_named_posterior_prior(pseudo_prior, ckpt_dir, name="pseudo_prior")
    print(f"[PSEUDO PRIOR] saved fixed warm prior to {prior_path}")

    return pseudo_prior

