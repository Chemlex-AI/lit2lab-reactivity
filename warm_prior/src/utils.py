"""Data preparation, input validation, checkpoint and output helpers."""

import json
import os
from typing import Any, Dict, List, Optional

import jax.numpy as jnp
import numpy as np
import pandas as pd


def save_named_posterior_prior(
    posterior_prior: Dict[str, jnp.ndarray],
    save_dir: str,
    name: str = "pseudo_prior",
):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{name}.npz")
    np.savez(path, **{k: np.asarray(v) for k, v in posterior_prior.items()})
    return path



def select_or_load_initial_indices(
    pool_indices: np.ndarray,
    initial_random_size: int,
    rng_np: np.random.Generator,
    indices_path: Optional[str] = None,
) -> np.ndarray:
    """Sample or reload initial GT indices for reuse across arms.
    Selection is purely random, independent of labels and predictions.
    """
    pool_set = set(int(i) for i in pool_indices)

    if indices_path and os.path.exists(indices_path):
        loaded = np.asarray(np.load(indices_path)).astype(int).reshape(-1)
        if len(loaded) != len(set(loaded.tolist())):
            raise ValueError(f"initial_gt_indices_path `{indices_path}` contains duplicate indices.")
        missing = sorted(set(loaded.tolist()) - pool_set)
        if missing:
            raise ValueError(
                f"initial_gt_indices_path `{indices_path}` contains {len(missing)} indices "
                f"absent from the pool, e.g. {missing[:10]}."
            )
        if initial_random_size is not None and len(loaded) != int(initial_random_size):
            print(
                f"[WARN] loaded {len(loaded)} initial indices but --initial_random_size="
                f"{initial_random_size}; using the loaded set."
            )
        print(f"[INFO] loaded {len(loaded)} initial random GT indices from {indices_path}")
        return loaded

    if initial_random_size <= 0:
        raise ValueError("--initial_random_size must be positive.")
    if initial_random_size > len(pool_indices):
        raise ValueError(
            f"--initial_random_size={initial_random_size} exceeds pool size {len(pool_indices)}."
        )
    selected = rng_np.choice(np.asarray(pool_indices, dtype=int), size=int(initial_random_size), replace=False)
    selected = np.sort(selected.astype(int))
    print(f"[INFO] sampled {len(selected)} initial random GT indices (pure random, no model involved)")
    if indices_path:
        # Save the same initial set for all arms.
        os.makedirs(os.path.dirname(os.path.abspath(indices_path)), exist_ok=True)
        np.save(indices_path, selected)
        print(f"[INFO] saved initial random GT indices to {indices_path} for reuse across arms")
    return selected



def load_df(path: str) -> pd.DataFrame:
    if path is None:
        raise ValueError("Path cannot be None")
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_pickle(path)



def prepare_binary_labels(
    df: pd.DataFrame,
    gt_label_col: str,
    pseudo_label_col: Optional[str],
    yield_col: str,
    yield_threshold: float,
    require_pseudo: bool,
    split_name: str,
) -> pd.DataFrame:
    df = df.copy()

    if gt_label_col not in df.columns:
        if yield_col in df.columns:
            print(
                f"[{split_name}] `{gt_label_col}` not found; creating it from "
                f"`{yield_col} > {yield_threshold}`."
            )
            df[gt_label_col] = (df[yield_col] > yield_threshold).astype(np.int16)
        else:
            raise ValueError(
                f"[{split_name}] Missing GT label column `{gt_label_col}` and yield column `{yield_col}`."
            )

    df[gt_label_col] = df[gt_label_col].astype(np.int16)

    if require_pseudo:
        if pseudo_label_col is None or pseudo_label_col not in df.columns:
            raise ValueError(f"[{split_name}] Missing pseudo label column `{pseudo_label_col}`.")
    if pseudo_label_col is not None and pseudo_label_col in df.columns:
        # Preserve missing pseudo labels as NaN.
        df[pseudo_label_col] = pd.to_numeric(df[pseudo_label_col], errors="coerce")

    return df



def validate_feature_column(df: pd.DataFrame, fp_type: str, split_name: str):
    if fp_type not in df.columns:
        raise ValueError(f"[{split_name}] Missing feature column `{fp_type}`.")
    if df[fp_type].isna().any():
        raise ValueError(f"[{split_name}] Feature column `{fp_type}` contains NA values.")



def feature_matrix(df: pd.DataFrame, fp_type: str) -> jnp.ndarray:
    return jnp.asarray(np.asarray(df[fp_type].to_list()), dtype=jnp.float32)



def save_selection_log(
    save_path: str,
    round_idx: int,
    candidate_eval_df: pd.DataFrame,
    selected_this_round: List[int],
    gt_label_col: str,
    pseudo_label_col: Optional[str],
    selection_info: Dict[str, Any],
):
    if not selected_this_round:
        return
    selected_set = set(map(int, selected_this_round))
    log_df = candidate_eval_df[candidate_eval_df["ori_index"].astype(int).isin(selected_set)].copy()
    log_df["round"] = round_idx
    log_df["gt_label"] = log_df[gt_label_col].astype(np.int16)
    if pseudo_label_col is not None and pseudo_label_col in log_df.columns:
        log_df["pseudo_label"] = pd.to_numeric(log_df[pseudo_label_col], errors="coerce")
        log_df["was_correct_pseudo"] = np.where(
            log_df["pseudo_label"].notna(),
            (
                log_df["gt_label"].to_numpy(dtype=np.int16)
                == log_df["pseudo_label"].to_numpy(dtype=np.float32)
            ).astype(np.int16),
            np.nan,
        )
    for key, value in selection_info.items():
        if np.isscalar(value) or isinstance(value, str):
            log_df[f"selection_{key}"] = value

    wanted = [
        "round",
        "ori_index",
        "pred",
        "pred_id",
        "model_unc",
        "all_unc",
        "data_unc",
        "gt_label",
        "pseudo_label",
        "was_correct_pseudo",
    ]
    extra = [c for c in log_df.columns if c.startswith("selection_")]
    cols = [c for c in wanted + extra if c in log_df.columns]
    out_path = os.path.join(save_path, "selection_log.csv")
    log_df[cols].to_csv(out_path, mode="a", index=False, header=not os.path.exists(out_path))



def save_args(args, save_path: str):
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)



def check_or_filter_test_overlap(df_pool: pd.DataFrame, df_test: pd.DataFrame, args) -> pd.DataFrame:
    if not args.id_col:
        return df_pool
    if args.id_col not in df_pool.columns or args.id_col not in df_test.columns:
        raise ValueError(f"id_col `{args.id_col}` must exist in both pool and test dataframes.")
    overlap = set(df_pool[args.id_col].dropna().tolist()) & set(df_test[args.id_col].dropna().tolist())
    if not overlap:
        return df_pool
    msg = f"Found {len(overlap)} overlapping `{args.id_col}` values between pool and leave-out test."
    if args.on_test_overlap == "error":
        raise ValueError(msg)
    if args.on_test_overlap == "filter_pool":
        print(f"[WARN] {msg} Filtering those rows from pool.")
        return df_pool[~df_pool[args.id_col].isin(overlap)].copy()
    print(f"[WARN] {msg} Ignoring because --on_test_overlap=ignore.")
    return df_pool

