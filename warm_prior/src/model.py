"""Bayesian neural network, NumPyro inference and posterior summaries."""

import os
import time
from typing import Any, Dict, Optional, Tuple

import jax.nn as nn
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

PSEUDO_PROB_EPS = 1e-6


def _get_posterior_stat(
    posterior_prior: Optional[Dict[str, Any]],
    site_name: str,
) -> Tuple[Optional[jnp.ndarray], Optional[jnp.ndarray]]:
    """Return a latent site's prior mean and std."""
    if posterior_prior is None:
        return None, None
    mean_key = f"{site_name}__mean"
    std_key = f"{site_name}__std"
    if mean_key not in posterior_prior or std_key not in posterior_prior:
        return None, None
    return jnp.asarray(posterior_prior[mean_key]), jnp.asarray(posterior_prior[std_key])



def _beta_from_mean_std(mean, std, eps: float = 1e-4):
    """Compute numerically safe Beta parameters from mean and std."""
    mean = jnp.clip(mean, eps, 1.0 - eps)
    var = jnp.maximum(std ** 2, eps ** 2)
    max_var = jnp.maximum(mean * (1.0 - mean) - eps, eps ** 2)
    var = jnp.minimum(var, max_var)
    concentration = mean * (1.0 - mean) / var - 1.0
    alpha = mean * concentration
    beta = (1.0 - mean) * concentration
    return jnp.maximum(alpha, eps), jnp.maximum(beta, eps)



def _sample_with_optional_posterior_prior(
    site_name: str,
    default_dist,
    posterior_prior: Optional[Dict[str, Any]] = None,
    kind: str = "normal",
    scale_floor: float = 1e-3,
    beta_eps: float = 1e-4,
):
    """Sample from the default or pseudo-posterior prior."""
    post_mean, post_std = _get_posterior_stat(posterior_prior, site_name)

    if post_mean is None:
        return numpyro.sample(site_name, default_dist)

    if kind == "beta":
        mean = jnp.clip(post_mean, beta_eps, 1.0 - beta_eps)
        beta_std = jnp.maximum(post_std, scale_floor)
        alpha, beta = _beta_from_mean_std(mean, beta_std)
        return numpyro.sample(site_name, dist.Beta(alpha, beta))

    prior_std = jnp.maximum(post_std, scale_floor)
    if kind == "laplace":
        return numpyro.sample(site_name, dist.Laplace(post_mean, prior_std / jnp.sqrt(2.0)))

    return numpyro.sample(site_name, dist.Normal(post_mean, prior_std))



def mix_prior(
    shape1: int,
    shape2: Optional[int] = None,
    param_name: Optional[str] = None,
    posterior_prior: Optional[Dict[str, Any]] = None,
    posterior_prior_scale_floor: float = 1e-3,
):
    """Build a Normal/Laplace mixture centered on zero.
    If supplied, use diagonal pseudo-posterior moments for each latent site.
    """
    shape = (shape1, shape2) if shape2 is not None else (shape1,)
    mean = jnp.zeros(shape)

    pi = _sample_with_optional_posterior_prior(
        f"pi_{param_name}",
        dist.Beta(1, 1).expand(shape),
        posterior_prior=posterior_prior,
        kind="beta",
        scale_floor=posterior_prior_scale_floor,
    )
    normal = _sample_with_optional_posterior_prior(
        f"{param_name}_normal",
        dist.Normal(mean, jnp.ones(shape)),
        posterior_prior=posterior_prior,
        kind="normal",
        scale_floor=posterior_prior_scale_floor,
    )
    laplace = _sample_with_optional_posterior_prior(
        f"{param_name}_laplace",
        dist.Laplace(mean, jnp.ones(shape)),
        posterior_prior=posterior_prior,
        kind="laplace",
        scale_floor=posterior_prior_scale_floor,
    )
    return pi * normal + (1 - pi) * laplace



def noisy_teacher_prob(prob, sensitivity: float, specificity: float):
    """Return P(pseudo=1 | x) = s*p + (1-c)*(1-p), with p=P(GT=1 | x).
    s=P(pseudo=1 | GT=1); c=P(pseudo=0 | GT=0).
    """
    return sensitivity * prob + (1.0 - specificity) * (1.0 - prob)



def model(
    X,
    Y,
    D_H: int,
    D_Y: int = 1,
    posterior_prior: Optional[Dict[str, Any]] = None,
    posterior_prior_scale_floor: float = 1e-3,
    label_mode: str = "gt",
    teacher_sensitivity: Optional[float] = None,
    teacher_specificity: Optional[float] = None,
):
    """Two-layer BNN with prob=P(GT=1 | x, theta).
    GT likelihood: Bernoulli(prob); pseudo: Bernoulli(s*prob+(1-c)*(1-prob)).
    """
    if label_mode not in ("gt", "pseudo"):
        raise ValueError(f"Unknown label_mode `{label_mode}`; expected 'gt' or 'pseudo'.")

    N, D_X = X.shape

    w1 = mix_prior(
        D_X,
        D_H,
        param_name="w1",
        posterior_prior=posterior_prior,
        posterior_prior_scale_floor=posterior_prior_scale_floor,
    )
    b1 = mix_prior(
        D_H,
        param_name="b1",
        posterior_prior=posterior_prior,
        posterior_prior_scale_floor=posterior_prior_scale_floor,
    )
    z1 = nn.relu(jnp.matmul(X, w1) + b1)

    w2 = mix_prior(
        D_H,
        D_Y,
        param_name="w2",
        posterior_prior=posterior_prior,
        posterior_prior_scale_floor=posterior_prior_scale_floor,
    )
    b2 = mix_prior(
        D_Y,
        param_name="b2",
        posterior_prior=posterior_prior,
        posterior_prior_scale_floor=posterior_prior_scale_floor,
    )
    logits = (jnp.matmul(z1, w2) + b2).squeeze(-1)
    prob = nn.sigmoid(logits)

    # Prediction and acquisition use GT probability, not noisy-teacher probability.
    numpyro.deterministic("prob", prob)

    if label_mode == "gt":
        with numpyro.plate("data", N):
            numpyro.sample("Y", dist.Bernoulli(logits=logits), obs=Y)
        return

    if teacher_sensitivity is None or teacher_specificity is None:
        raise ValueError(
            "label_mode='pseudo' requires both teacher_sensitivity and teacher_specificity."
        )
    if not (0.0 < float(teacher_sensitivity) <= 1.0):
        raise ValueError(f"teacher_sensitivity must be in (0, 1]; got {teacher_sensitivity}.")
    if not (0.0 < float(teacher_specificity) <= 1.0):
        raise ValueError(f"teacher_specificity must be in (0, 1]; got {teacher_specificity}.")

    pseudo_prob = noisy_teacher_prob(prob, float(teacher_sensitivity), float(teacher_specificity))
    pseudo_prob = jnp.clip(pseudo_prob, PSEUDO_PROB_EPS, 1.0 - PSEUDO_PROB_EPS)
    with numpyro.plate("data", N):
        numpyro.sample("Y", dist.Bernoulli(probs=pseudo_prob), obs=Y)



def run_inference(
    bayes_model,
    rng_key,
    X,
    Y,
    D_H: int,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    posterior_prior: Optional[Dict[str, Any]] = None,
    posterior_prior_scale_floor: float = 1e-3,
    label_mode: str = "gt",
    teacher_sensitivity: Optional[float] = None,
    teacher_specificity: Optional[float] = None,
):
    start = time.time()

    kernel = NUTS(bayes_model)

    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
    )
    mcmc.run(
        rng_key,
        X,
        Y,
        D_H,
        posterior_prior=posterior_prior,
        posterior_prior_scale_floor=posterior_prior_scale_floor,
        label_mode=label_mode,
        teacher_sensitivity=teacher_sensitivity,
        teacher_specificity=teacher_specificity,
    )
    print(f"\nMCMC elapsed time: {time.time() - start:.2f}s")
    return mcmc.get_samples()



def samples_to_posterior_prior(
    samples: Dict[str, Any],
    scale_floor: float = 1e-3,
    scale_mult: float = 1.0,
) -> Dict[str, jnp.ndarray]:
    """Build diagonal prior moments from latent samples, excluding prob."""
    posterior_prior: Dict[str, jnp.ndarray] = {}
    for key, value in samples.items():
        if key == "prob":
            continue
        arr = np.asarray(value)
        ddof = 1 if arr.shape[0] > 1 else 0
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0, ddof=ddof) * scale_mult
        std = np.nan_to_num(std, nan=scale_floor, posinf=scale_floor, neginf=scale_floor)
        posterior_prior[f"{key}__mean"] = jnp.asarray(mean)
        posterior_prior[f"{key}__std"] = jnp.asarray(np.maximum(std, scale_floor))
    return posterior_prior

