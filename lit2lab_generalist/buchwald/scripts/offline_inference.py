# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Text-only Buchwald inference using LLaMA-Factory templates and vLLM."""

import argparse
import json
import os
from pathlib import Path
import re


def clean_prediction(text):
    """Keep the existing EOS cleanup and first occurrence of condition fields."""
    markers = [text.find(t) for t in ("<|end_of_text|>", "<|eot_id|>") if t in text]
    if markers:
        text = text[: min(markers)]
    match = re.search(r"<think>([\s\S]*?)</think>", text, re.IGNORECASE)
    if not match:
        return text
    keys = {"Catalyst_system", "Ligand", "Base", "Solvent_class", "Additives"}
    seen, lines = set(), []
    for line in match.group(1).splitlines():
        field = re.match(r"^\s*([A-Za-z_]+)\s*=", line)
        if field and field.group(1) in keys:
            if field.group(1) in seen:
                continue
            seen.add(field.group(1))
        lines.append(line)
    return text[: match.start(1)] + "\n".join(lines) + text[match.end(1) :]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data"
    )
    parser.add_argument("--dataset", default="buchwald_test")
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--template", default="llama3")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--cutoff-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    if args.world < 1 or not 0 <= args.rank < args.world:
        parser.error("Expected 0 <= rank < world")
    if (
        min(
            args.batch_size,
            args.tensor_parallel_size,
            args.cutoff_len,
            args.max_new_tokens,
        )
        < 1
    ):
        parser.error("Batch size, parallel size and token limits must be positive")
    if args.output_path.exists():
        parser.error("Output already exists; choose a new output path")

    os.environ.setdefault("DISABLE_VERSION_CHECK", "1")
    from transformers import Seq2SeqTrainingArguments
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.extras.constants import IGNORE_INDEX
    from llamafactory.hparams import get_infer_args
    from llamafactory.model import load_tokenizer
    from vllm import LLM, SamplingParams

    model_args, data_args, _, generating_args = get_infer_args(
        {
            "model_name_or_path": args.model_path,
            "dataset": args.dataset,
            "dataset_dir": str(args.dataset_dir),
            "template": args.template,
            "cutoff_len": args.cutoff_len,
            "enable_thinking": True,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": -1,
            "max_new_tokens": args.max_new_tokens,
            "preprocessing_num_workers": 1,
        }
    )
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset = get_dataset(
        template,
        model_args,
        data_args,
        Seq2SeqTrainingArguments(output_dir="outputs/cache"),
        "ppo",
        **tokenizer_module,
    )["train_dataset"]
    indices = list(range(args.rank, len(dataset), args.world))
    dataset = dataset.select(indices)
    llm = LLM(
        model=args.model_path,
        trust_remote_code=False,
        dtype=model_args.infer_dtype,
        max_model_len=args.cutoff_len + args.max_new_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        disable_log_stats=True,
    )
    sampling = SamplingParams(
        temperature=generating_args.temperature,
        top_p=generating_args.top_p,
        top_k=-1,
        max_tokens=generating_args.max_new_tokens,
        stop_token_ids=template.get_stop_token_ids(tokenizer),
        seed=args.seed,
        n=1,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("x", encoding="utf-8") as stream:
        for start in range(0, len(dataset), args.batch_size):
            batch = dataset[start : start + args.batch_size]
            # Ground-truth labels are decoded separately and never passed to generation.
            inputs = [
                {"prompt": tokenizer.decode(ids, skip_special_tokens=True)}
                for ids in batch["input_ids"]
            ]
            outputs = llm.generate(inputs, sampling)
            if len(outputs) != len(inputs):
                raise RuntimeError("Inference returned an incomplete batch")
            for offset, result in enumerate(outputs):
                label = tokenizer.decode(
                    [t for t in batch["labels"][offset] if t != IGNORE_INDEX],
                    skip_special_tokens=True,
                )
                record = {
                    "row_index": indices[start + offset],
                    "predict": clean_prediction(result.outputs[0].text or ""),
                    "label": label,
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(indices)} predictions to {args.output_path}")


if __name__ == "__main__":
    main()
