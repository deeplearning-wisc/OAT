"""OAT baselines: random step, first step, and all-at-once prompt."""

from __future__ import annotations

import json
import os
import re
from typing import List, Optional, Tuple

import numpy as np

import config

ALL_AT_ONCE_PROMPT = """You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem.
The problem is: {question}.
Identify one or more likely error candidates in this trajectory.
Here's the conversation:
{trajectory}

Additional constraints:
- Candidate agents: {agent_list}
- Return at least 1 candidate
- Candidates do NOT need to be earliest mistakes

Output exactly one JSON object:
{{"candidates": [{{"step": <int>, "agent": "<agent name>", "reason": "<short reason>"}}, ... ]}}"""

SYSTEM_PROMPT = (
    "You are a strict JSON annotator for failure attribution. "
    "Output ONLY one valid JSON object and nothing else."
)


def _score_step_indices(traj: dict) -> List[int]:
    step_ids = [int(x) for x in traj.get("step_indices", list(range(traj["num_steps"])))]
    step_ids = [s for s in step_ids if s >= 0]
    if not step_ids:
        step_ids = list(range(max(1, int(traj.get("num_steps_original", traj["num_steps"])))))
    return step_ids


def _prediction_from_steps(traj: dict, pred_steps: List[int]) -> dict:
    step_ids = _score_step_indices(traj)
    mapped_steps: List[int] = []
    mapped_locals: List[int] = []
    for raw in pred_steps:
        if raw in step_ids:
            local = step_ids.index(raw)
        else:
            local = int(np.argmin([abs(raw - s) for s in step_ids]))
        step = int(step_ids[local])
        if step not in mapped_steps:
            mapped_steps.append(step)
            mapped_locals.append(local)
    if not mapped_steps:
        mapped_steps = [int(step_ids[0])]
        mapped_locals = [0]
    scores = np.zeros(len(step_ids), dtype=np.float32)
    for rank, local in enumerate(mapped_locals):
        scores[local] = max(scores[local], 1.0 / float(rank + 1))
    return {
        "trajectory_id": traj["id"],
        "predicted_step": int(mapped_steps[0]),
        "predicted_local_step": int(mapped_locals[0]),
        "score_step_indices": step_ids,
        "scores": scores,
        "topk_detection": [int(s) for s in mapped_steps],
        "conformal_detection": [int(s) for s in mapped_steps],
        "gt_error_steps": traj["error_steps"],
        "num_steps": traj.get("num_steps_original", traj["num_steps"]),
        "num_scored_steps": len(step_ids),
    }


def random_step_baseline(failures: List[dict], seed: int = config.RANDOM_SEED) -> List[dict]:
    rng = np.random.RandomState(seed)
    preds = []
    for traj in failures:
        step_ids = _score_step_indices(traj)
        preds.append(_prediction_from_steps(traj, [int(step_ids[rng.randint(0, len(step_ids))])]))
    return preds


def first_step_baseline(failures: List[dict]) -> List[dict]:
    return [_prediction_from_steps(traj, [_score_step_indices(traj)[0]]) for traj in failures]


def _strip_thinking_block(response: str) -> str:
    end_tag = "</" + "think>"
    if end_tag in response:
        response = response.split(end_tag)[-1]
    return response.strip()


def parse_all_at_once_response(response: str, max_step: int) -> Tuple[List[int], bool]:
    response = _strip_thinking_block(response)
    steps: List[int] = []
    parsed_ok = False
    try:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            candidates = parsed.get("candidates", [])
            if isinstance(candidates, list):
                for candidate in candidates:
                    if not isinstance(candidate, dict) or "step" not in candidate:
                        continue
                    step = int(candidate["step"])
                    if 0 <= step <= max_step:
                        steps.append(step)
                parsed_ok = bool(steps)
    except Exception:
        parsed_ok = False
    if not steps:
        for raw in re.findall(r'"step"\s*:\s*(-?\d+)', response):
            step = int(raw)
            if 0 <= step <= max_step:
                steps.append(step)
        parsed_ok = bool(steps)
    deduped = []
    seen = set()
    for step in steps:
        if step not in seen:
            deduped.append(step)
            seen.add(step)
    return (deduped or [0]), parsed_ok


def build_all_at_once_prompt(traj: dict) -> str:
    agents = sorted({str(a) for a in traj.get("step_agents", []) if str(a)})
    return ALL_AT_ONCE_PROMPT.format(
        question=traj.get("question") or traj.get("prompt", ""),
        trajectory=traj.get("text", ""),
        agent_list=", ".join(agents) if agents else "unknown",
    )


class _AllAtOnceBackend:
    def __init__(
        self,
        backend: str = config.LLM_JUDGE_BACKEND,
        model_name: str = config.LLM_JUDGE_MODEL_NAME,
        max_new_tokens: int = config.LLM_JUDGE_MAX_NEW_TOKENS,
    ) -> None:
        self.backend = backend
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        if backend == "vllm":
            from vllm import LLM, SamplingParams

            self.sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
            self.client = LLM(model=model_name, trust_remote_code=True)
        elif backend == "azure":
            from openai import AzureOpenAI

            api_key = os.environ.get(config.AZURE_OPENAI_API_KEY_ENV)
            if not api_key:
                raise RuntimeError(f"Missing {config.AZURE_OPENAI_API_KEY_ENV}")
            self.client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
                api_version=config.AZURE_OPENAI_API_VERSION,
            )
        else:
            raise ValueError("backend must be 'vllm' or 'azure'")

    def complete(self, prompt: str) -> str:
        if self.backend == "vllm":
            output = self.client.generate([prompt], self.sampling_params)[0]
            return output.outputs[0].text
        response = self.client.chat.completions.create(
            model=config.AZURE_OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""


def all_at_once_prompt_baseline(
    failures: List[dict],
    backend: Optional[_AllAtOnceBackend] = None,
    backend_name: str = config.LLM_JUDGE_BACKEND,
    model_name: str = config.LLM_JUDGE_MODEL_NAME,
) -> List[dict]:
    client = backend or _AllAtOnceBackend(backend=backend_name, model_name=model_name)
    predictions = []
    for traj in failures:
        prompt = build_all_at_once_prompt(traj)
        response = client.complete(prompt)
        steps, parsed_ok = parse_all_at_once_response(response, max_step=traj.get("num_steps_original", traj["num_steps"]) - 1)
        pred = _prediction_from_steps(traj, steps)
        pred["parsed_ok"] = parsed_ok
        predictions.append(pred)
    return predictions


def run_baselines(
    failures: List[dict],
    include_prompt: bool = False,
    backend_name: str = config.LLM_JUDGE_BACKEND,
    model_name: str = config.LLM_JUDGE_MODEL_NAME,
) -> dict[str, List[dict]]:
    results = {
        "random_step": random_step_baseline(failures),
        "first_step": first_step_baseline(failures),
    }
    if include_prompt:
        results["all_at_once_prompt"] = all_at_once_prompt_baseline(
            failures,
            backend_name=backend_name,
            model_name=model_name,
        )
    return results
