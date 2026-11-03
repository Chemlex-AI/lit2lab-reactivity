"""Command-line arguments and release-relative default paths."""

import argparse
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARMS = ("vanilla", "naive_warm", "reliability_adjust_warm")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gpu_devices", type=str, default="0")
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(PROJECT_DIR / "data" / "acid-amine" / "acidamine_pool.pkl.gz"),
        help="Pool set path. This should exclude leave-out test rows.",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default=str(PROJECT_DIR / "data" / "acid-amine" / "acidamine_test.pkl.gz"),
        help="Independent leave-out test set path. Never sampled by AL.",
    )
    parser.add_argument("--fp_type", type=str, default="drfp")
    parser.add_argument("--gt_label_col", type=str, default="gt")
    parser.add_argument("--pseudo_label_col", type=str, default="pseudo_label")
    parser.add_argument("--yield_col", type=str, default="rxn_yield")
    parser.add_argument("--yield_threshold", type=float, default=20.0)

    parser.add_argument(
        "--arm",
        type=str,
        default="reliability_adjust_warm",
        choices=list(ARMS),
        help=(
            "vanilla: no pseudo prior. naive_warm: pseudo labels treated as clean GT. "
            "reliability_adjust_warm: noise-aware pseudo likelihood using estimated teacher reliability: "
            "sensitivity/specificity."
        ),
    )
    parser.add_argument(
        "--initial_random_size",
        type=int,
        default=50,
        help="Number of PURE RANDOM GT reactions revealed before anything else.",
    )
    parser.add_argument(
        "--initial_gt_indices_path",
        type=str,
        default=None,
        help="Optional .npy of initial GT ori_index values, so all arms share the same 50.",
    )
    parser.add_argument("--teacher_beta_alpha0", type=float, default=2.0, help="Beta smoothing alpha0.")
    parser.add_argument("--teacher_beta_beta0", type=float, default=2.0, help="Beta smoothing beta0.")
    parser.add_argument(
        "--report_teacher_calibration",
        action="store_true",
        help="Also report teacher reliability for vanilla/naive_warm (reporting only, unused by their priors).",
    )

    parser.add_argument("--num_samples_per_round", type=int, default=50)
    parser.add_argument(
        "--total_rounds",
        type=int,
        default=30,
        help="Number of train/eval rounds. Round 0 trains on the initial random GT set.",
    )
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--num_warmup", type=int, default=200)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--num_chains", type=int, default=1)
    parser.add_argument("--seed", type=int, default=666)

    parser.add_argument("--sample_method", type=str, default="model_unc", choices=["model_unc", "all_unc", "data_unc"])
    parser.add_argument(
        "--selection_strategy",
        type=str,
        default="top_unc",
        choices=["top_unc", "random"],
    )

    parser.add_argument(
        "--posterior_prior_scale_floor",
        type=float,
        default=1e-3,
        help="Minimum std used when converting pseudo posterior samples into the warm prior.",
    )
    parser.add_argument(
        "--posterior_prior_scale_mult",
        type=float,
        default=1.0,
        help="Multiplier on posterior std when constructing the warm prior.",
    )

    parser.add_argument("--id_col", type=str, default=None, help="Optional ID column to detect pool/test overlap.")
    parser.add_argument("--on_test_overlap", type=str, default="error", choices=["error", "filter_pool", "ignore"])

    parser.add_argument(
        "--suffix",
        type=str,
        default=str(PROJECT_DIR / "outputs" / "bnn_reliability_adjust_warm_prior"),
        help="Exact output directory; no automatic suffix is appended.",
    )

    parser.add_argument("--compress_outputs", action="store_true", help="Lossless gzip prediction pickles")
    args = parser.parse_args(argv)
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    return args

