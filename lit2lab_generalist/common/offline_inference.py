"""Shared SGLang offline inference with resumable multi-rollout output."""

import argparse
import json
from pathlib import Path

import pandas as pd
import tqdm

try:
    import sglang as sgl
except ImportError:  # pragma: no cover - exercised only in runtime environments without sglang.
    sgl = None

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - exercised only in runtime environments without transformers.
    AutoTokenizer = None


OUTPUT_COLUMN = "generated_text"
PROMPT_CACHE_COLUMN = "_prompt_text_cache"
SOURCE_ROW_COLUMN = "source_row_pos"
ROLLOUT_ID_COLUMN = "rollout_id"
BATCH_SIZE = 256
SAVE_EVERY_BATCHES = 20
NUM_ROLLOUTS = 1

SAMPLING_PARAMS = {
    "temperature": 0.95,
    "top_k": -1,
    "top_p": 0.7,
    "max_new_tokens": 4096,
}

ENGINE_KWARGS = {
    "tp_size": 8,
    "dp_size": 1,
    "cuda_graph_max_bs": 256,
    "max_running_requests": 4096,
}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


def parse_args(default_input, default_output, argv=None):
    parser = argparse.ArgumentParser(
        description="Run SGLang offline inference with multi-rollout long-format output."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input-path", default=str(default_input))
    parser.add_argument("--output-column", default=OUTPUT_COLUMN)
    parser.add_argument("--output-path", default=str(default_output))
    parser.add_argument("--num-rollouts", type=positive_int, default=NUM_ROLLOUTS)
    parser.add_argument("--batch-size", type=positive_int, default=BATCH_SIZE)
    parser.add_argument("--save-every-batches", type=positive_int, default=SAVE_EVERY_BATCHES)
    parser.add_argument("--tp-size", type=positive_int, default=ENGINE_KWARGS["tp_size"])
    parser.add_argument("--dp-size", type=positive_int, default=ENGINE_KWARGS["dp_size"])
    parser.add_argument(
        "--cuda-graph-max-bs",
        type=positive_int,
        default=ENGINE_KWARGS["cuda_graph_max_bs"],
    )
    parser.add_argument(
        "--max-running-requests",
        type=positive_int,
        default=ENGINE_KWARGS["max_running_requests"],
    )
    return parser.parse_args(argv)


def build_engine_kwargs(args) -> dict:
    return {
        "tp_size": args.tp_size,
        "dp_size": args.dp_size,
        "cuda_graph_max_bs": args.cuda_graph_max_bs,
        "max_running_requests": args.max_running_requests,
    }


def chunks(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def build_output_path(input_path: str, output_column: str, num_rollouts: int) -> Path:
    path = Path(input_path)
    return path.with_name(
        f"{path.stem}_with_{output_column}_{num_rollouts}rollouts_scaling_generalist.parquet"
    )


def build_checkpoint_path(output_path: Path) -> Path:
    return output_path.with_suffix(".checkpoint.jsonl")


def normalize_rollout_records(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame(columns=["row_pos", ROLLOUT_ID_COLUMN, "text"])

    required_columns = {"row_pos", ROLLOUT_ID_COLUMN, "text"}
    missing_columns = required_columns.difference(records.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(f"Rollout records are missing required columns: {missing_str}")

    normalized = records.loc[:, ["row_pos", ROLLOUT_ID_COLUMN, "text"]].copy()
    normalized["row_pos"] = normalized["row_pos"].astype(int)
    normalized[ROLLOUT_ID_COLUMN] = normalized[ROLLOUT_ID_COLUMN].astype(int)
    normalized["text"] = normalized["text"].fillna("").astype(str)
    normalized = normalized.drop_duplicates(
        subset=["row_pos", ROLLOUT_ID_COLUMN],
        keep="last",
    )
    return normalized.sort_values(["row_pos", ROLLOUT_ID_COLUMN]).reset_index(drop=True)


def load_checkpoint(checkpoint_path: Path) -> pd.DataFrame:
    if not checkpoint_path.exists():
        return pd.DataFrame(columns=["row_pos", ROLLOUT_ID_COLUMN, "text"])

    records = []
    with checkpoint_path.open("r") as file_obj:
        for line in file_obj:
            if not line.strip():
                continue
            record = json.loads(line)
            if ROLLOUT_ID_COLUMN not in record:
                raise ValueError(
                    "Checkpoint format is missing rollout_id; delete the old checkpoint and rerun."
                )
            records.append(
                {
                    "row_pos": record["row_pos"],
                    ROLLOUT_ID_COLUMN: record[ROLLOUT_ID_COLUMN],
                    "text": record.get("text", ""),
                }
            )
    return normalize_rollout_records(pd.DataFrame(records))


def get_pending_rollout_pairs(num_rows: int, num_rollouts: int, completed_pairs) -> list[tuple[int, int]]:
    pending_pairs = []
    for row_pos in range(num_rows):
        for rollout_id in range(num_rollouts):
            if (row_pos, rollout_id) not in completed_pairs:
                pending_pairs.append((row_pos, rollout_id))
    return pending_pairs


def build_prompt_texts(prompt_structs, tokenizer):
    return [
        tokenizer.apply_chat_template(
            prompt_struct,
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt_struct in prompt_structs
    ]


def ensure_prompt_cache(df: pd.DataFrame, row_positions, tokenizer):
    if PROMPT_CACHE_COLUMN not in df.columns:
        df[PROMPT_CACHE_COLUMN] = None

    cache_col_idx = df.columns.get_loc(PROMPT_CACHE_COLUMN)
    missing_positions = [
        row_pos for row_pos in row_positions if pd.isna(df.iat[row_pos, cache_col_idx])
    ]
    if not missing_positions:
        return

    prompt_structs = df.iloc[missing_positions]["prompt"].tolist()
    prompt_texts = build_prompt_texts(prompt_structs, tokenizer)
    for row_pos, prompt_text in zip(missing_positions, prompt_texts):
        df.iat[row_pos, cache_col_idx] = prompt_text


def append_checkpoint_rows(checkpoint_path: Path, rollout_pairs, outputs) -> None:
    with checkpoint_path.open("a") as file_obj:
        for (row_pos, rollout_id), output in zip(rollout_pairs, outputs):
            file_obj.write(
                json.dumps(
                    {
                        "row_pos": row_pos,
                        ROLLOUT_ID_COLUMN: rollout_id,
                        "text": output.get("text", ""),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def build_long_format_output(
    source_df: pd.DataFrame,
    rollout_records: pd.DataFrame,
    output_column: str,
) -> pd.DataFrame:
    normalized_records = normalize_rollout_records(rollout_records)
    expanded_source = source_df.iloc[normalized_records["row_pos"].tolist()].reset_index(drop=True)
    expanded_source = expanded_source.drop(columns=[PROMPT_CACHE_COLUMN], errors="ignore")
    expanded_source.insert(0, SOURCE_ROW_COLUMN, normalized_records["row_pos"].tolist())
    expanded_source[ROLLOUT_ID_COLUMN] = normalized_records[ROLLOUT_ID_COLUMN].tolist()
    expanded_source[output_column] = normalized_records["text"].tolist()
    return expanded_source


def write_final_output(
    source_df: pd.DataFrame,
    rollout_records: pd.DataFrame,
    output_path: Path,
    output_column: str,
) -> None:
    final_df = build_long_format_output(source_df, rollout_records, output_column)
    final_df.to_parquet(output_path, index=False)


def run_inference_on_file(
    llm,
    tokenizer,
    input_path: str,
    output_column: str = OUTPUT_COLUMN,
    num_rollouts: int = NUM_ROLLOUTS,
    batch_size: int = BATCH_SIZE,
    save_every_batches: int = SAVE_EVERY_BATCHES,
    output_path: str | None = None,
) -> None:
    df = pd.read_parquet(input_path)
    df = df.drop(columns=[output_column], errors="ignore")

    output_path = Path(output_path) if output_path else build_output_path(
        input_path, output_column, num_rollouts
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = build_checkpoint_path(output_path)

    restored_records = load_checkpoint(checkpoint_path)
    restored = len(restored_records)
    if restored:
        print(f"[resume] restored {restored} rollout rows from {checkpoint_path}")

    completed_pairs = {
        (int(row_pos), int(rollout_id))
        for row_pos, rollout_id in restored_records[["row_pos", ROLLOUT_ID_COLUMN]].itertuples(
            index=False,
            name=None,
        )
    }
    pending_pairs = get_pending_rollout_pairs(len(df), num_rollouts, completed_pairs)
    total_batches = (len(pending_pairs) + batch_size - 1) // batch_size
    batches_done = 0
    rows_done = 0
    generated_records = restored_records.to_dict("records")

    for rollout_pairs in tqdm.tqdm(chunks(pending_pairs, batch_size), total=total_batches):
        row_positions = sorted({row_pos for row_pos, _ in rollout_pairs})
        ensure_prompt_cache(df, row_positions, tokenizer)
        prompt_texts = [df.iat[row_pos, df.columns.get_loc(PROMPT_CACHE_COLUMN)] for row_pos, _ in rollout_pairs]
        outputs = llm.generate(prompt_texts, SAMPLING_PARAMS)

        batch_records = []
        for (row_pos, rollout_id), output in zip(rollout_pairs, outputs):
            batch_records.append(
                {
                    "row_pos": row_pos,
                    ROLLOUT_ID_COLUMN: rollout_id,
                    "text": output.get("text", ""),
                }
            )

        generated_records.extend(batch_records)
        append_checkpoint_rows(checkpoint_path, rollout_pairs, outputs)
        batches_done += 1
        rows_done += len(rollout_pairs)

        if batches_done % save_every_batches == 0:
            print(
                f"[checkpoint] {checkpoint_path} "
                f"(batches_done={batches_done}, rollout_rows_done={rows_done})"
            )

    write_final_output(df, pd.DataFrame(generated_records), output_path, output_column)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(f"[done] saved to {output_path}")


def run_inference_cli(default_input: Path, default_output: Path, argv=None):
    args = parse_args(default_input, default_output, argv)
    if sgl is None:
        raise ImportError("sglang is required to run offline inference.")
    if AutoTokenizer is None:
        raise ImportError("transformers is required to run offline inference.")

    llm = sgl.Engine(model_path=args.model_path, **build_engine_kwargs(args))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    run_inference_on_file(
        llm,
        tokenizer,
        args.input_path,
        output_column=args.output_column,
        num_rollouts=args.num_rollouts,
        batch_size=args.batch_size,
        save_every_batches=args.save_every_batches,
        output_path=args.output_path,
    )
