"""GT-only active learning workflow and batch acquisition strategies."""

import os
from typing import Any, Dict, List, Optional, Tuple

import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro
import pandas as pd
from numpyro.infer import Predictive

from .arguments import ARMS
from .evaluation import evaluate_split, label_source_summary
from .model import model, run_inference
from .prior import build_pseudo_prior, estimate_teacher_noise_posterior
from .sanity_checks import run_sanity_tests
from .utils import (
    check_or_filter_test_overlap,
    feature_matrix,
    load_df,
    prepare_binary_labels,
    save_args,
    save_selection_log,
    select_or_load_initial_indices,
    validate_feature_column,
)


def select_next_batch(
    candidate_df: pd.DataFrame,
    batch_size: int,
    strategy: str,
    sample_method: str,
    rng_np: np.random.Generator,
) -> Tuple[List[int], Dict[str, Any]]:
    if candidate_df.empty or batch_size <= 0:
        return [], {"strategy": strategy, "reason": "empty_candidate_or_nonpositive_batch"}

    if sample_method not in candidate_df.columns and strategy != "random":
        raise ValueError(f"sample_method `{sample_method}` not found in candidate dataframe.")

    batch_size = min(batch_size, len(candidate_df))

    if strategy == "random":
        selected = rng_np.choice(candidate_df["ori_index"].to_numpy(dtype=int), size=batch_size, replace=False)
        return selected.astype(int).tolist(), {"strategy": strategy, "num_unlabeled": len(candidate_df)}

    if strategy == "top_unc":
        selected_df = candidate_df.sort_values(sample_method, ascending=False).iloc[:batch_size]
        return selected_df["ori_index"].astype(int).to_list(), {
            "strategy": strategy,
            "num_unlabeled": len(candidate_df),
        }

    raise ValueError(f"Unknown selection_strategy `{strategy}`.")



def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    numpyro.set_host_device_count(args.num_chains)
    np.random.seed(args.seed)
    rng_np = np.random.default_rng(args.seed)
    rng_key = random.PRNGKey(args.seed)

    run_sanity_tests(verbose=True)

    if args.arm not in ARMS:
        raise ValueError(f"Unknown --arm `{args.arm}`; expected one of {ARMS}.")
    needs_pseudo = args.arm != "vanilla"

    save_path = args.suffix
    os.makedirs(save_path, exist_ok=True)
    save_args(args, save_path)
    print(f"[INFO] arm: {args.arm}")
    print(f"[INFO] save_path: {save_path}")

    df_pool_all = load_df(args.data_path)
    df_test = load_df(args.test_path)

    df_pool_all = check_or_filter_test_overlap(df_pool_all, df_test, args)

    df_pool_all = prepare_binary_labels(
        df_pool_all,
        gt_label_col=args.gt_label_col,
        pseudo_label_col=args.pseudo_label_col,
        yield_col=args.yield_col,
        yield_threshold=args.yield_threshold,
        require_pseudo=needs_pseudo,
        split_name="pool",
    ).reset_index(drop=True)
    df_test = prepare_binary_labels(
        df_test,
        gt_label_col=args.gt_label_col,
        pseudo_label_col=args.pseudo_label_col if args.pseudo_label_col in df_test.columns else None,
        yield_col=args.yield_col,
        yield_threshold=args.yield_threshold,
        require_pseudo=False,
        split_name="test",
    ).reset_index(drop=True)

    validate_feature_column(df_pool_all, args.fp_type, "pool")
    validate_feature_column(df_test, args.fp_type, "test")

    df_pool_all["ori_index"] = np.arange(len(df_pool_all), dtype=int)
    df_pool_all["is_selected_gt"] = False

    X_pool_all = feature_matrix(df_pool_all, args.fp_type)
    y_pool_gt = df_pool_all[args.gt_label_col].to_numpy(dtype=np.int16)
    X_test = feature_matrix(df_test, args.fp_type)
    y_test = df_test[args.gt_label_col].to_numpy(dtype=np.int16)

    if len(df_pool_all) == 0:
        raise ValueError("Pool dataframe is empty after loading/filtering.")
    if len(df_test) == 0:
        raise ValueError("Leave-out test dataframe is empty.")

    has_pseudo_col = args.pseudo_label_col in df_pool_all.columns
    if has_pseudo_col:
        pseudo_label_available = df_pool_all[args.pseudo_label_col].notna().to_numpy(dtype=bool)
    else:
        pseudo_label_available = np.zeros(len(df_pool_all), dtype=bool)
    if needs_pseudo and not pseudo_label_available.any():
        raise ValueError(f"Pool has no rows with available pseudo labels; arm `{args.arm}` needs them.")

    print(f"[INFO] pool size: {len(df_pool_all)}, test size: {len(df_test)}")
    print(f"[INFO] feature dim: {X_pool_all.shape[1]}, hidden dim: {args.hidden_dim}")
    if has_pseudo_col and pseudo_label_available.any():
        pseudo_acc_all = float(
            (
                df_pool_all.loc[pseudo_label_available, args.pseudo_label_col]
                .to_numpy(dtype=np.float32)
                .astype(np.int16)
                == y_pool_gt[pseudo_label_available]
            ).mean()
        )
        print(
            f"[INFO] pseudo labels available in pool: {int(pseudo_label_available.sum())}/{len(df_pool_all)}; "
            f"OFFLINE ORACLE ONLY full-pool pseudo acc: {pseudo_acc_all:.4f} "
            f"(reporting only; never used for training/prior construction/acquisition)"
        )

    # Choose initial random GT before inference.
    initial_indices = select_or_load_initial_indices(
        pool_indices=df_pool_all["ori_index"].to_numpy(dtype=int),
        initial_random_size=args.initial_random_size,
        rng_np=rng_np,
        indices_path=args.initial_gt_indices_path,
    )
    assert len(initial_indices) == len(set(initial_indices.tolist())), "Initial GT indices are not unique."
    assert set(initial_indices.tolist()).issubset(
        set(df_pool_all["ori_index"].astype(int).tolist())
    ), "Initial GT indices are not all present in the pool."

    initial_gt_mask = np.zeros(len(df_pool_all), dtype=bool)
    initial_gt_mask[initial_indices] = True
    df_pool_all.loc[df_pool_all["ori_index"].isin(initial_indices.tolist()), "is_selected_gt"] = True
    assert int(df_pool_all["is_selected_gt"].sum()) == len(initial_indices)

    np.save(os.path.join(save_path, "initial_selected_indices.npy"), initial_indices.astype(int))
    print(
        f"[INFO] initial GT set: n={len(initial_indices)}, "
        f"pos={int((y_pool_gt[initial_indices] == 1).sum())}, "
        f"neg={int((y_pool_gt[initial_indices] == 0).sum())}"
    )

    ckpt_dir = os.path.join(save_path, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Estimate teacher noise from initial random GT only.
    teacher_sensitivity: Optional[float] = None
    teacher_specificity: Optional[float] = None
    calibration: Optional[Dict[str, Any]] = None

    want_calibration = (args.arm == "reliability_adjust_warm") or (args.report_teacher_calibration and has_pseudo_col)
    if want_calibration:
        if not has_pseudo_col:
            raise ValueError(
                f"arm `{args.arm}` needs pseudo labels, but column `{args.pseudo_label_col}` is missing."
            )
        cal_gt = y_pool_gt[initial_indices].astype(int)
        cal_pseudo = (
            df_pool_all.iloc[initial_indices][args.pseudo_label_col].to_numpy(dtype=np.float32)
        )
        calibration = estimate_teacher_noise_posterior(
            cal_gt, cal_pseudo, alpha0=args.teacher_beta_alpha0, beta0=args.teacher_beta_beta0
        )
        assert calibration["n_calibration"] + calibration["n_dropped_missing_pseudo"] == len(initial_indices), (
            "Teacher reliability estimation used a different number of rows than the initial random GT set."
        )
        calibration["used_for_prior"] = bool(args.arm == "reliability_adjust_warm")
        print(
            f"[RELIABILITY ADJUST WARM PRIOR] n={calibration['n_calibration']} "
            f"TP={calibration['tp']} FP={calibration['fp']} TN={calibration['tn']} FN={calibration['fn']} "
            f"sensitivity={calibration['sensitivity_mean']:.4f} "
            f"specificity={calibration['specificity_mean']:.4f} "
            f"youden_J={calibration['youden_j']:.4f} "
            f"(used_for_prior={calibration['used_for_prior']})"
        )
        if args.arm == "reliability_adjust_warm":
            teacher_sensitivity = float(calibration["sensitivity_mean"])
            teacher_specificity = float(calibration["specificity_mean"])
    else:
        print(f"[RELIABILITY ADJUST WARM PRIOR] teacher reliability reporting skipped for arm `{args.arm}`.")

    # Build the fixed pseudo prior before AL, excluding initial GT reactions.
    pseudo_train_mask = pseudo_label_available & (~initial_gt_mask)
    assert not np.any(pseudo_train_mask & initial_gt_mask), "Initial GT leaked into pseudo-prior training."

    rng_key, pseudo_key = random.split(rng_key)
    pseudo_prior = build_pseudo_prior(
        arm=args.arm,
        rng_key=pseudo_key,
        X_pool_all=X_pool_all,
        df_pool_all=df_pool_all,
        pseudo_train_mask=pseudo_train_mask,
        pseudo_label_col=args.pseudo_label_col,
        args=args,
        teacher_sensitivity=teacher_sensitivity,
        teacher_specificity=teacher_specificity,
        ckpt_dir=ckpt_dir,
    )
    # Vanilla uses no pseudo prior or teacher-noise parameters.
    if args.arm == "vanilla":
        assert pseudo_prior is None, "vanilla arm must not build a pseudo prior."
        assert teacher_sensitivity is None and teacher_specificity is None

    # Run GT-only active learning, starting with initial GT in round 0.
    metrics_rows: List[Dict[str, Any]] = []
    label_source_rows: List[Dict[str, Any]] = []
    selected_history: List[np.ndarray] = [initial_indices.copy()]
    np.save(
        os.path.join(save_path, "selected_indices_by_round.npy"),
        np.array(selected_history, dtype=object),
        allow_pickle=True,
    )

    for round_idx in range(args.total_rounds):
        print("\n" + "=" * 80)
        print(f"[Round {round_idx}] start (arm={args.arm})")

        source_row = label_source_summary(
            round_idx,
            df_pool_all,
            args.gt_label_col,
            args.pseudo_label_col if has_pseudo_col else None,
        )
        label_source_rows.append(source_row)
        gt_selected_count = source_row["gt_selected_count"]
        gt_ratio = source_row["gt_ratio"]

        # Train on cumulative revealed GT only.
        selected_mask = df_pool_all["is_selected_gt"].to_numpy(dtype=bool)
        train_indices = np.where(selected_mask)[0]
        if len(train_indices) == 0:
            raise RuntimeError("No selected GT samples available for target training.")
        assert np.all(selected_mask[train_indices]), "Training set contains unrevealed pool rows."
        if round_idx == 0:
            assert np.array_equal(np.sort(train_indices), np.sort(initial_indices)), (
                "Round 0 must train exactly on the initial random GT set."
            )

        train_x = X_pool_all[train_indices, ...]
        train_y_np = y_pool_gt[train_indices].astype(np.int16)
        train_y = jnp.asarray(train_y_np, dtype=jnp.int16)
        training_mode = f"cumulative_gt_only_{args.arm}"

        source_row["arm"] = args.arm
        source_row["training_mode"] = training_mode
        source_row["train_n"] = int(len(train_y_np))
        source_row["train_gt_count"] = int(len(train_y_np))
        source_row["train_pseudo_count"] = 0
        source_row["unlabeled_remaining_count"] = int((~selected_mask).sum())

        print(
            f"[Round {round_idx}] training_mode={training_mode}: N={len(train_y_np)} GT labels "
            f"(pos={(train_y_np == 1).sum()}, neg={(train_y_np == 0).sum()}), "
            f"selected_gt_ratio={gt_ratio:.4f}"
        )

        # Preserve the audit column; warm-prior moments do not decay across rounds.
        source_row["fixed_prior_std"] = 1.0
        source_row["uses_pseudo_prior"] = bool(pseudo_prior is not None)

        if pseudo_prior is None:
            print(f"[Round {round_idx}] prior: default vague prior (vanilla)")
        else:
            print(
                f"[Round {round_idx}] prior: fixed {args.arm} pseudo warm prior, "
                "fixed_prior_std=1.0000"
            )

        rng_key, infer_key = random.split(rng_key)
        samples = run_inference(
            model,
            infer_key,
            train_x,
            train_y,
            args.hidden_dim,
            num_warmup=args.num_warmup,
            num_samples=args.num_samples,
            num_chains=args.num_chains,
            posterior_prior=pseudo_prior,
            posterior_prior_scale_floor=args.posterior_prior_scale_floor,
            label_mode="gt",
            teacher_sensitivity=None,
            teacher_specificity=None,
        )

        predictive_probs = Predictive(model, samples, return_sites=["prob"])

        # Evaluate the held-out test set.
        rng_key, test_key, all_pool_key, remaining_key = random.split(rng_key, 4)
        _, test_metrics = evaluate_split(
            save_path,
            test_key,
            predictive_probs,
            df_test,
            X_test,
            y_test,
            args.hidden_dim,
            round_idx,
            split_name="test",
            save_pickle=True,
            compress_outputs=args.compress_outputs,
            gt_selected_count=gt_selected_count,
            pseudo_remaining_count=0,
            gt_ratio=gt_ratio,
        )
        metrics_rows.append(test_metrics)

        # Full-pool GT metrics: offline reporting only.
        _, all_pool_metrics = evaluate_split(
            save_path,
            all_pool_key,
            predictive_probs,
            df_pool_all,
            X_pool_all,
            y_pool_gt,
            args.hidden_dim,
            round_idx,
            split_name="all_pool",
            save_pickle=True,
            compress_outputs=args.compress_outputs,
            gt_selected_count=gt_selected_count,
            pseudo_remaining_count=0,
            gt_ratio=gt_ratio,
        )
        metrics_rows.append(all_pool_metrics)

        # Remaining-pool GT metrics: offline reporting only.
        remaining_mask = ~df_pool_all["is_selected_gt"].to_numpy(dtype=bool)
        remaining_indices = np.where(remaining_mask)[0]
        df_remaining = df_pool_all.iloc[remaining_indices].copy()
        X_remaining = X_pool_all[remaining_indices, ...]
        y_remaining_gt = y_pool_gt[remaining_indices]
        remaining_eval_df, remaining_metrics = evaluate_split(
            save_path,
            remaining_key,
            predictive_probs,
            df_remaining,
            X_remaining,
            y_remaining_gt,
            args.hidden_dim,
            round_idx,
            split_name="remaining_pool",
            save_pickle=True,
            compress_outputs=args.compress_outputs,
            gt_selected_count=gt_selected_count,
            pseudo_remaining_count=0,
            gt_ratio=gt_ratio,
        )
        metrics_rows.append(remaining_metrics)

        pd.DataFrame(metrics_rows).to_csv(os.path.join(save_path, "metrics_by_round.csv"), index=False)
        pd.DataFrame(label_source_rows).to_csv(
            os.path.join(save_path, "train_label_source_by_round.csv"), index=False
        )

        if round_idx == args.total_rounds - 1:
            print(f"[Round {round_idx}] final training/eval round; skip acquisition.")
            break

        if len(remaining_eval_df) == 0:
            print(f"[Round {round_idx}] no remaining pool samples; stop AL.")
            break

        selected_this_round, selection_info = select_next_batch(
            remaining_eval_df,
            batch_size=args.num_samples_per_round,
            strategy=args.selection_strategy,
            sample_method=args.sample_method,
            rng_np=rng_np,
        )

        selected_this_round = [int(x) for x in selected_this_round]
        if len(selected_this_round) != len(set(selected_this_round)):
            raise RuntimeError("Selection contains duplicate ori_index values.")
        already_selected = set(df_pool_all.loc[df_pool_all["is_selected_gt"], "ori_index"].astype(int).tolist())
        overlap_selected = already_selected & set(selected_this_round)
        if overlap_selected:
            raise RuntimeError(f"Selection repeated previously selected indices: {sorted(list(overlap_selected))[:10]}")

        print(f"[Round {round_idx}] selected {len(selected_this_round)} samples; info={selection_info}")
        save_selection_log(
            save_path,
            round_idx,
            remaining_eval_df,
            selected_this_round,
            args.gt_label_col,
            args.pseudo_label_col if has_pseudo_col else None,
            selection_info,
        )

        # Reveal acquired GT in the offline simulation.
        df_pool_all.loc[df_pool_all["ori_index"].isin(selected_this_round), "is_selected_gt"] = True
        cumulative_selected = df_pool_all.loc[df_pool_all["is_selected_gt"], "ori_index"].to_numpy(dtype=int)
        selected_history.append(cumulative_selected.copy())

        np.save(os.path.join(save_path, f"round_{round_idx:03d}_selected_this_round.npy"), np.array(selected_this_round))
        np.save(os.path.join(save_path, "selected_indices_by_round.npy"), np.array(selected_history, dtype=object), allow_pickle=True)

    print("\n[INFO] Finished.")
    print(f"[INFO] arm: {args.arm}")
    print(f"[INFO] Results saved to: {save_path}")

