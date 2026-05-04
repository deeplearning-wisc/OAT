"""Hidden-state extraction for OAT datasets."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from data_pipeline import (
    get_step_last_token_indices,
    get_step_token_ranges,
    load_raw_trajectories,
)


def _get_model_input_device(model) -> torch.device:
    if hasattr(model, "hf_device_map"):
        for _, device in model.hf_device_map.items():
            if isinstance(device, str) and device.startswith("cuda"):
                return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def load_model_and_tokenizer(
    model_name: str = config.MODEL_NAME,
    device: str = config.DEVICE,
    dtype: torch.dtype = config.DTYPE,
    attn_implementation: Optional[str] = None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": "auto" if device == "cuda" else device,
        "trust_remote_code": True,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def extract_hidden_states_for_trajectory(
    text: str,
    char_boundaries: list,
    model,
    tokenizer,
    max_length: int = config.MAX_SEQ_LENGTH,
    aggregation: str = config.HIDDEN_STATE_AGGREGATION,
) -> Optional[torch.Tensor]:
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    input_ids = encoding["input_ids"]
    offset_mapping = encoding["offset_mapping"][0].tolist()
    seq_len = input_ids.shape[1]

    if seq_len >= max_length:
        max_char_covered = offset_mapping[-1][1] if offset_mapping[-1][1] > 0 else 0
        valid_steps = [i for i, (_, end) in enumerate(char_boundaries) if end <= max_char_covered]
        if not valid_steps:
            return None
        char_boundaries = char_boundaries[: valid_steps[-1] + 1]

    if aggregation == "mean":
        step_locations = get_step_token_ranges(offset_mapping, char_boundaries)
    elif aggregation == "last":
        step_locations = get_step_last_token_indices(offset_mapping, char_boundaries)
    else:
        raise ValueError("aggregation must be 'last' or 'mean'")

    input_device = _get_model_input_device(model)
    outputs = model(
        input_ids=input_ids.to(input_device),
        attention_mask=encoding["attention_mask"].to(input_device),
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    all_hidden = outputs.hidden_states
    T = len(char_boundaries)
    num_layers = len(all_hidden)
    hidden_dim = all_hidden[0].shape[-1]

    if aggregation == "last":
        token_ids = torch.tensor(step_locations, device=input_device)
        return torch.stack([layer_hs[0].index_select(0, token_ids) for layer_hs in all_hidden], dim=1).float().cpu()

    result = torch.zeros(T, num_layers, hidden_dim, dtype=torch.float32)
    for layer_idx, layer_hs in enumerate(all_hidden):
        layer_hs = layer_hs[0].float()
        for step_idx, (start_tok, end_tok) in enumerate(step_locations):
            if start_tok <= end_tok:
                result[step_idx, layer_idx] = layer_hs[start_tok : end_tok + 1].mean(dim=0).cpu()
            else:
                result[step_idx, layer_idx] = layer_hs[end_tok].cpu()
    return result


def _build_input_text(traj: dict) -> tuple[str, list[tuple[int, int]]]:
    question = traj.get("question", "")
    prefix = f"Question: {question} Trajectory: "
    offset = len(prefix)
    question_start = len("Question: ")
    question_end = max(question_start, question_start + len(question) - 1)
    bounds = [(question_start, question_end)]
    bounds.extend((start + offset, end + offset) for start, end in traj["step_char_boundaries"])
    return prefix + traj["text"], bounds


def extract_all(
    dataset: str,
    model,
    tokenizer,
    aggregation: str = config.HIDDEN_STATE_AGGREGATION,
    force: bool = False,
) -> None:
    trajectories = load_raw_trajectories(dataset, include_success=True)
    out_dir = config.get_hidden_states_dir(aggregation, dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    skipped = 0
    elapsed_times = []
    for traj in tqdm(trajectories, desc=f"extract:{dataset}"):
        safe_id = traj["id"].replace("/", "_").replace(".json", "")
        out_path = out_dir / f"{safe_id}.pt"
        if out_path.exists() and not force:
            continue
        text, bounds = _build_input_text(traj)
        start = time.perf_counter()
        hidden = extract_hidden_states_for_trajectory(text, bounds, model, tokenizer, aggregation=aggregation)
        if hidden is None:
            skipped += 1
            continue
        elapsed_times.append(time.perf_counter() - start)
        actual_t = hidden.shape[0]
        torch.save(
            {
                "trajectory_id": traj["id"],
                "hidden_states": hidden,
                "step_labels": ([0] + traj["step_labels"])[:actual_t],
                "step_agents": (["question"] + traj["step_agents"])[:actual_t],
                "step_is_model": ([True] + traj["step_is_model"])[:actual_t],
                "step_indices": ([-1] + list(range(traj["num_steps"])))[:actual_t],
                "error_steps": traj["error_steps"],
                "mistake_step": traj["mistake_step"],
                "mistake_agent": traj["mistake_agent"],
                "num_steps_original": traj["num_steps"],
                "num_steps_extracted": actual_t,
                "dataset_source": dataset,
                "aggregation": aggregation,
                "model_name": config.MODEL_NAME,
            },
            out_path,
        )

    avg = sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0.0
    print(f"Saved hidden states to {out_dir}; skipped={skipped}; avg_extract_seconds={avg:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract OAT proxy-LLM hidden states.")
    parser.add_argument("--dataset", choices=config.DATASETS, default=config.DEFAULT_DATASET)
    parser.add_argument("--aggregation", choices=["last", "mean"], default=config.HIDDEN_STATE_AGGREGATION)
    parser.add_argument("--model-name", default=config.MODEL_NAME)
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        device=args.device,
        attn_implementation=args.attn_implementation,
    )
    extract_all(args.dataset, model, tokenizer, aggregation=args.aggregation, force=args.force)


if __name__ == "__main__":
    main()
