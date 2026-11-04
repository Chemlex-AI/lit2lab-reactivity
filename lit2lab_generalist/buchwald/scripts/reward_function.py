# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""EasyR1 batch adapter for the outcome-only reward shared with Suzuki."""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.outcome_reward import compute_score as outcome_score, extract_answer


def compute_score(reward_inputs):
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use reward_type=batch for this reward function.")

    scores = []
    for item in reward_inputs:
        expected = extract_answer(str(item["ground_truth"]))
        if expected is None:
            raise ValueError("Ground truth must contain a high/low label.")
        result = outcome_score(None, item["response"], expected)
        scores.append(
            {"overall": result["score"], "outcome_reward": result["outcome_reward"]}
        )
    return scores
