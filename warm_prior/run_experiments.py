#!/usr/bin/env python3
"""Run the reliability adjust warm prior experiments on one device."""

import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "active_learning_bnn_reliability_adjust_warm_prior.py"


def load_config(dataset):
    return json.loads((ROOT / "configs" / (dataset + ".json")).read_text())


def data_path(directory, relative):
    path = directory / relative
    if not path.exists() and Path(str(path) + ".gz").exists():
        path = Path(str(path) + ".gz")
    return path


def argument_list(parameters):
    tokens = []
    for key, value in sorted(parameters.items()):
        if value is None or value is False:
            continue
        tokens.append("--" + key)
        if value is not True:
            tokens.append(str(value))
    return tokens


def build_jobs(dataset, seeds, methods, data_dir, output_dir, gpu):
    config = load_config(dataset)
    unknown = set(methods) - set(config["methods"])
    if unknown:
        raise ValueError("Methods unavailable for %s: %s" % (dataset, sorted(unknown)))
    if not set(seeds) <= set(config["seeds"]):
        raise ValueError("Only the saved seeds 41, 42, 43 are supported")
    paths = {split: data_path(data_dir, config[split])
             for split in ("pool", "test")}
    jobs = []
    for seed in seeds:
        for method in methods:
            entry = config["methods"][method]
            parameters = entry["args"].copy()
            output = output_dir / config["campaign"] / ("seed%d" % seed) / entry["output_name"]
            initial = ROOT / "initial_indices" / dataset / ("initial_gt_indices_seed%d.npy" % seed)
            parameters.update(seed=seed, gpu_devices="" if gpu == "cpu" else gpu,
                              data_path=str(paths["pool"]), test_path=str(paths["test"]),
                              initial_gt_indices_path=str(initial), suffix=str(output))
            command = [sys.executable, "-u", str(ENGINE)] + argument_list(parameters)
            jobs.append({"dataset": dataset, "method": method, "seed": seed,
                         "output": output, "parameters": parameters, "command": command})
    return jobs


def audit_completion(output, parameters):
    """Require every round/split and GT budget, not just the last CSV line."""
    with (output / "metrics_by_round.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rounds = parameters["total_rounds"]
    expected = {(r, split) for r in range(rounds)
                for split in ("test", "all_pool", "remaining_pool")}
    actual = [(int(row["round"]), row["split"]) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Incomplete or duplicated metrics: %s" % output)
    for row in rows:
        budget = parameters["initial_random_size"] + int(row["round"]) * parameters["num_samples_per_round"]
        if int(row["gt_selected_count"]) != budget:
            raise ValueError("Wrong GT budget: %s" % output)
    import numpy as np
    history = np.load(output / "selected_indices_by_round.npy", allow_pickle=True)
    if len(history) != rounds:
        raise ValueError("Incomplete selection history: %s" % output)
    for r, selected in enumerate(history):
        budget = parameters["initial_random_size"] + r * parameters["num_samples_per_round"]
        if len(selected) != budget or len(set(map(int, selected))) != budget:
            raise ValueError("Wrong selection history at round %d: %s" % (r, output))


def run_job(job, gpu):
    output, parameters = job["output"], job["parameters"]
    if output.exists():
        raise FileExistsError("Refusing to overwrite %s; choose a new --output-dir" % output)
    # Atomic directory creation also prevents two launchers from taking this job.
    output.mkdir(parents=True, exist_ok=False)
    (output / "launch_args.json").write_text(json.dumps(parameters, indent=2) + "\n")
    (output / "command.txt").write_text(shlex.join(job["command"]) + "\n")
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="" if gpu == "cpu" else gpu,
               JAX_PLATFORMS="cpu" if gpu == "cpu" else "cuda",
               XLA_PYTHON_CLIENT_PREALLOCATE="false", PYTHONUNBUFFERED="1")
    print("START %s seed=%s method=%s -> %s" %
          (job["dataset"], job["seed"], job["method"], output), flush=True)
    with (output / "run.log").open("w") as log:
        subprocess.run(job["command"], env=env, stdin=subprocess.DEVNULL,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    audit_completion(output, parameters)
    print("DONE %s" % output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", "acidamine", "buchwald", "suzuki"], default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    parser.add_argument("--methods", nargs="+", choices=["vanilla", "naive", "RA", "random"])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--gpu", default="0", help="One physical GPU index, or cpu for testing")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without requiring data or launching jobs")
    parser.add_argument("--check-data", action="store_true", help="Check configured input files exist and exit")
    args = parser.parse_args()
    datasets = ["acidamine", "buchwald", "suzuki"] if args.dataset == "all" else [args.dataset]
    jobs = []
    for dataset in datasets:
        methods = args.methods or list(load_config(dataset)["methods"])
        jobs.extend(build_jobs(dataset, args.seeds, methods, args.data_dir.resolve(),
                               args.output_dir.resolve(), args.gpu))
    if args.dry_run:
        for job in jobs:
            print(shlex.join(job["command"]))
        print("# %d jobs; no training launched" % len(jobs))
        return
    required = {ENGINE}
    for job in jobs:
        required.update(Path(job["parameters"][key])
                        for key in ("data_path", "test_path", "initial_gt_indices_path"))
    missing = sorted(str(path) for path in required if not path.is_file())
    if missing:
        raise FileNotFoundError("Missing experiment inputs:\n%s\nPlace downloaded data as described in README.md." %
                                "\n".join(missing))
    if args.check_data:
        print("PASS: all configured data, initial-index and training-script files exist")
        return
    for job in jobs:
        run_job(job, args.gpu)


if __name__ == "__main__":
    main()
