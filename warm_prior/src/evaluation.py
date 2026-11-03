"""Prediction, uncertainty decomposition and offline evaluation metrics."""

import os
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics


def disentangle_uncertainty(predictions: np.ndarray):
    """Split entropy into aleatoric and epistemic uncertainty."""
    predictions = np.asarray(predictions, dtype=float)
    mean = predictions.mean(axis=1)

    def entropy(prob):
        with np.errstate(divide="ignore", invalid="ignore"):
            return -(
                np.where(prob > 0, prob * np.log(prob), 0.0)
                + np.where(prob < 1, (1 - prob) * np.log1p(-prob), 0.0)
            )

    predictive = entropy(mean)
    aleatoric = entropy(predictions).mean(axis=1)
    epistemic = np.clip(predictive - aleatoric, 0.0, 1.0)
    return mean, predictive, aleatoric, epistemic



def binary_metrics(y_true: Sequence[int], prob: Sequence[float]) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob)
    pred = (prob > 0.5).astype(int)
    if len(y_true) == 0:
        return {"acc": np.nan, "f1_macro": np.nan, "recall": np.nan}
    return {
        "acc": float(sk_metrics.accuracy_score(y_true, pred)),
        "f1_macro": float(sk_metrics.f1_score(y_true, pred, average="macro", zero_division=0)),
        "recall": float(sk_metrics.recall_score(y_true, pred, zero_division=0)),
    }



def evaluate_split(
    save_path: str,
    rng_key_predict,
    predictive_probs,
    df: pd.DataFrame,
    X,
    y_true: np.ndarray,
    D_H: int,
    round_idx: int,
    split_name: str,
    save_pickle: bool = True,
    gt_selected_count: int = 0,
    pseudo_remaining_count: int = 0,
    gt_ratio: float = 0.0,
    compress_outputs: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return predictions and metrics for one split.
    Pool GT scoring is offline reporting only; never used for learning or selection.
    """
    df_eval = df.copy()
    n = len(df_eval)

    if n == 0:
        for col in ["data_unc", "model_unc", "all_unc", "pred", "pred_id"]:
            df_eval[col] = []
        metric_row = {
            "round": round_idx,
            "split": split_name,
            "n_samples": 0,
            "acc": np.nan,
            "f1_macro": np.nan,
            "recall": np.nan,
            "mean_model_unc": np.nan,
            "mean_all_unc": np.nan,
            "model_unc_all_unc_ratio": np.nan,
            "gt_selected_count": gt_selected_count,
            "pseudo_remaining_count": pseudo_remaining_count,
            "gt_ratio": gt_ratio,
        }
        return df_eval, metric_row

    probs_posterior = predictive_probs(rng_key_predict, X, None, D_H)["prob"]
    predictions = np.asarray(probs_posterior.T)  # examples x posterior samples
    mean, predictive_uncertainty, aleatoric_uncertainty, epistemic_uncertainty = disentangle_uncertainty(predictions)

    df_eval["data_unc"] = aleatoric_uncertainty
    df_eval["model_unc"] = epistemic_uncertainty
    df_eval["all_unc"] = predictive_uncertainty
    df_eval["pred"] = mean
    df_eval["pred_id"] = (df_eval["pred"] > 0.5).astype(np.int16)

    m = binary_metrics(y_true, mean)
    model_unc = float(np.mean(epistemic_uncertainty))
    all_unc = float(np.mean(predictive_uncertainty))
    unc_ratio = float((model_unc / all_unc) * 100) if all_unc != 0 else 0.0

    oracle_tag = "" if split_name == "test" else " [OFFLINE ORACLE ONLY]"
    print("==== Evaluation metrics ====")
    print(f"[RESULT] {split_name.upper()} N: {n}{oracle_tag}")
    print(f"[RESULT] {split_name.upper()} ACCURACY: {m['acc'] * 100:.2f}%")
    print(f"[RESULT] {split_name.upper()} F1_MACRO: {m['f1_macro']:.4f}")
    print(f"[RESULT] {split_name.upper()} RECALL: {m['recall']:.4f}")
    print(f"[RESULT] {split_name.upper()} MEAN_MODEL_UNC: {model_unc:.4f}")
    print(f"[RESULT] {split_name.upper()} MODEL_UNC/ALL_UNC_RATIO: {unc_ratio:.2f}%")
    print("===========================")

    if save_pickle:
        split_dir = os.path.join(save_path, split_name)
        os.makedirs(split_dir, exist_ok=True)
        if compress_outputs:
            df_eval.to_pickle(os.path.join(split_dir, f"{round_idx:03d}_unc_data.pkl.gz"), compression="gzip")
        else:
            df_eval.to_pickle(os.path.join(split_dir, f"{round_idx:03d}_unc_data.pkl"))

    metric_row = {
        "round": round_idx,
        "split": split_name,
        "n_samples": n,
        "acc": m["acc"],
        "f1_macro": m["f1_macro"],
        "recall": m["recall"],
        "mean_model_unc": model_unc,
        "mean_all_unc": all_unc,
        "model_unc_all_unc_ratio": unc_ratio,
        "gt_selected_count": gt_selected_count,
        "pseudo_remaining_count": pseudo_remaining_count,
        "gt_ratio": gt_ratio,
    }
    return df_eval, metric_row



def label_source_summary(
    round_idx: int,
    df_pool_all: pd.DataFrame,
    gt_label_col: str,
    pseudo_label_col: Optional[str],
) -> Dict[str, Any]:
    """Summarize label usage; pseudo/GT agreement is offline reporting only."""
    selected = df_pool_all["is_selected_gt"].to_numpy(dtype=bool)
    y_gt = df_pool_all[gt_label_col].to_numpy(dtype=np.int16)
    total = len(df_pool_all)

    row = {
        "round": round_idx,
        "pool_total": total,
        "gt_selected_count": int(selected.sum()),
        "gt_ratio": float(selected.sum() / total) if total else 0.0,
        "selected_pos_gt": int((y_gt[selected] == 1).sum()),
        "selected_neg_gt": int((y_gt[selected] == 0).sum()),
    }

    if pseudo_label_col is None or pseudo_label_col not in df_pool_all.columns:
        row.update(
            {
                "pseudo_remaining_count": 0,
                "pseudo_missing_remaining_count": int((~selected).sum()),
                "pseudo_pos": 0,
                "pseudo_neg": 0,
                "pseudo_correct_remaining_offline_oracle": 0,
                "pseudo_acc_remaining_offline_oracle": np.nan,
            }
        )
        return row

    y_pseudo = pd.to_numeric(df_pool_all[pseudo_label_col], errors="coerce").to_numpy(dtype=np.float32)
    pseudo_valid = ~np.isnan(y_pseudo)
    remaining_pseudo = y_pseudo[(~selected) & pseudo_valid].astype(np.int16)
    remaining_gt_for_pseudo = y_gt[(~selected) & pseudo_valid]
    row.update(
        {
            "pseudo_remaining_count": int(((~selected) & pseudo_valid).sum()),
            "pseudo_missing_remaining_count": int(((~selected) & (~pseudo_valid)).sum()),
            "pseudo_pos": int((remaining_pseudo == 1).sum()),
            "pseudo_neg": int((remaining_pseudo == 0).sum()),
            # Unrevealed GT is used only for offline reporting.
            "pseudo_correct_remaining_offline_oracle": int((remaining_pseudo == remaining_gt_for_pseudo).sum()),
            "pseudo_acc_remaining_offline_oracle": (
                float((remaining_pseudo == remaining_gt_for_pseudo).mean())
                if len(remaining_pseudo)
                else np.nan
            ),
        }
    )
    return row

