"""Dataset serialization, hidden-state loading, PCA projection, and CORAL alignment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA

import config


def _is_model_step_who_when(entry: dict, dataset_type: str) -> bool:
    if dataset_type == "hand_crafted":
        return entry.get("role", "") != "human"
    return entry.get("name", "") != "Computer_terminal"


def _agent_from_who_when_step(step: dict, dataset_type: str) -> str:
    if dataset_type == "hand_crafted":
        role = step.get("role", "")
        if role == "human":
            return "human"
        if role.startswith("Orchestrator"):
            return "Orchestrator"
        return role
    return step.get("name", "") or step.get("role", "unknown")


def serialize_who_and_when_trajectory(
    history: List[dict],
    dataset_type: str,
) -> Tuple[str, List[Tuple[int, int]], List[str], List[bool]]:
    parts: List[str] = []
    char_bounds: List[Tuple[int, int]] = []
    agents: List[str] = []
    is_model: List[bool] = []
    cursor = 0

    for t, entry in enumerate(history):
        role = entry.get("role", "unknown")
        agent = _agent_from_who_when_step(entry, dataset_type)
        if dataset_type == "hand_crafted":
            header = f"[STEP {t}] [ROLE: {role}] [AGENT: {agent}]\n"
        else:
            name = entry.get("name", "")
            header = f"[STEP {t}] [ROLE: {role}] [AGENT: {name}]\n" if name else f"[STEP {t}] [ROLE: {role}]\n"
        step_text = header + str(entry.get("content", "")) + "\n"
        start = cursor
        cursor += len(step_text)
        parts.append(step_text)
        char_bounds.append((start, cursor - 1))
        agents.append(agent)
        is_model.append(_is_model_step_who_when(entry, dataset_type))

    return "".join(parts), char_bounds, agents, is_model


def _serialize_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def serialize_mcp_atlas_trajectory(
    history: List[dict],
    max_tool_content_length: int = config.MAX_TOOL_CONTENT_LENGTH,
) -> Tuple[str, List[Tuple[int, int]], List[str], List[bool]]:
    parts: List[str] = []
    char_bounds: List[Tuple[int, int]] = []
    agents: List[str] = []
    is_model_flags: List[bool] = []
    cursor = 0

    for t, entry in enumerate(history):
        role = entry.get("role", "unknown")
        is_model = role == "assistant"
        header = f"[STEP {t}] [ROLE: {role}]\n"
        body_parts: List[str] = []

        if is_model:
            reasoning = (entry.get("original_message") or {}).get("reasoning_content", "")
            if reasoning:
                body_parts.append(f"[REASONING]\n{reasoning}")

        content = _serialize_content(entry.get("content"))
        if not is_model and len(content) > max_tool_content_length:
            content = content[:max_tool_content_length] + "\n...[truncated]"
        if content:
            body_parts.append(content)

        for tc in entry.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            func = tc.get("function", {})
            body_parts.append(f"[TOOL_CALL: {func.get('name', 'unknown')}({func.get('arguments', '')})]")

        step_text = header + ("\n".join(body_parts) + "\n" if body_parts else "\n")
        start = cursor
        cursor += len(step_text)
        parts.append(step_text)
        char_bounds.append((start, cursor - 1))
        agents.append(str(role))
        is_model_flags.append(is_model)

    return "".join(parts), char_bounds, agents, is_model_flags


def _parse_scope(scope: Any) -> Optional[int]:
    if isinstance(scope, int):
        return scope
    if isinstance(scope, str) and scope.lstrip("-").isdigit():
        return int(scope)
    return None


def load_who_and_when_trajectories(data_dir: Path = config.WHO_AND_WHEN_DATA_DIR) -> List[dict]:
    trajectories: List[dict] = []
    for subdir_name, dtype in [("Hand-Crafted", "hand_crafted"), ("Algorithm-Generated", "algorithm_generated")]:
        subdir = Path(data_dir) / subdir_name
        if not subdir.exists():
            continue
        for fp in sorted(subdir.glob("*.json"), key=lambda p: int(p.stem)):
            with open(fp) as f:
                data = json.load(f)
            history = data.get("history", [])
            if not history:
                continue
            mistake_step = int(data.get("mistake_step", -1))
            if not (0 <= mistake_step < len(history)):
                continue
            text, bounds, agents, is_model = serialize_who_and_when_trajectory(history, dtype)
            trajectories.append(
                {
                    "id": f"{subdir_name}/{fp.name}",
                    "is_success": False,
                    "dataset_source": "who_and_when",
                    "text": text,
                    "step_char_boundaries": bounds,
                    "step_agents": agents,
                    "step_is_model": is_model,
                    "step_labels": [int(i == mistake_step) for i in range(len(history))],
                    "error_steps": [mistake_step],
                    "mistake_step": mistake_step,
                    "mistake_agent": data.get("mistake_agent", ""),
                    "num_steps": len(history),
                    "question": data.get("question", ""),
                    "prompt": data.get("question", ""),
                }
            )
    return trajectories


def load_mcp_atlas_trajectories(
    data_dir: Path = config.MCP_ATLAS_DATA_DIR,
    include_success: bool = True,
) -> List[dict]:
    trajectories: List[dict] = []
    for fp in sorted(Path(data_dir).glob("*.json")):
        with open(fp) as f:
            data = json.load(f)
        errors = data.get("errors")
        history = data.get("raw_conversation_history", [])
        if not isinstance(errors, list) or not history:
            continue

        error_steps: List[int] = []
        for err in errors:
            scope = _parse_scope(err.get("scope"))
            if scope is not None and 0 <= scope - 1 < len(history):
                error_steps.append(scope - 1)
        error_steps = sorted(set(error_steps))
        is_success = len(errors) == 0
        if is_success and not include_success:
            continue
        if not is_success and not error_steps:
            continue

        text, bounds, agents, is_model = serialize_mcp_atlas_trajectory(history)
        trajectories.append(
            {
                "id": fp.stem,
                "is_success": is_success,
                "dataset_source": "mcp_atlas",
                "text": text,
                "step_char_boundaries": bounds,
                "step_agents": agents,
                "step_is_model": is_model,
                "step_labels": [int(i in set(error_steps)) for i in range(len(history))],
                "error_steps": error_steps,
                "mistake_step": error_steps[0] if error_steps else -1,
                "mistake_agent": "",
                "num_steps": len(history),
                "question": data.get("PROMPT", ""),
                "prompt": data.get("PROMPT", ""),
            }
        )
    return trajectories


def load_raw_trajectories(dataset: str, include_success: bool = True) -> List[dict]:
    if dataset == "mcp_atlas":
        return load_mcp_atlas_trajectories(include_success=include_success)
    if dataset == "who_and_when":
        return load_who_and_when_trajectories()
    raise ValueError(f"Unsupported dataset: {dataset}")


def get_step_last_token_indices(offset_mapping: List[Tuple[int, int]], char_boundaries: List[Tuple[int, int]]) -> List[int]:
    results: List[int] = []
    for _start_char, end_char in char_boundaries:
        last_tok = 0
        lo, hi = 0, len(offset_mapping) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            tok_start, tok_end = offset_mapping[mid]
            if tok_start == 0 and tok_end == 0:
                lo = mid + 1
            elif tok_start <= end_char:
                last_tok = mid
                lo = mid + 1
            else:
                hi = mid - 1
        results.append(last_tok)
    return results


def get_step_token_ranges(offset_mapping: List[Tuple[int, int]], char_boundaries: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    n_tokens = len(offset_mapping)
    for start_char, end_char in char_boundaries:
        first_tok = n_tokens - 1
        last_tok = 0
        for i, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_start == 0 and tok_end == 0:
                continue
            if tok_end >= start_char and tok_start <= end_char:
                first_tok = min(first_tok, i)
                last_tok = max(last_tok, i)
        ranges.append((first_tok, last_tok))
    return ranges


def load_hidden_state(traj_id: str, layer: int, hs_dir: Path) -> Optional[dict]:
    safe = traj_id.replace("/", "_").replace(".json", "")
    pt_path = hs_dir / f"{safe}.pt"
    if not pt_path.exists():
        return None
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    hs = data["hidden_states"][:, layer, :]
    step_is_model = list(data.get("step_is_model", [True] * hs.shape[0]))
    step_indices = list(data.get("step_indices", list(range(hs.shape[0]))))
    return {
        "hidden_states": hs,
        "step_is_model": step_is_model[: hs.shape[0]],
        "step_indices": [int(x) for x in step_indices[: hs.shape[0]]],
    }


def load_trajectories(
    dataset: str = config.DEFAULT_DATASET,
    layer: int = config.DEFAULT_LAYER,
    aggregation: str = config.HIDDEN_STATE_AGGREGATION,
    model_steps_only: bool = config.MODEL_STEPS_ONLY,
) -> List[dict]:
    metas = load_raw_trajectories(dataset, include_success=True)
    hs_dir = config.get_hidden_states_dir(aggregation, dataset)
    trajectories: List[dict] = []
    for meta in metas:
        record = load_hidden_state(meta["id"], layer, hs_dir)
        if record is None:
            continue
        hs = record["hidden_states"]
        step_is_model = record["step_is_model"]
        step_indices_all = record["step_indices"]
        T = min(hs.shape[0], len(step_is_model), len(step_indices_all))
        hs = hs[:T]
        step_is_model = step_is_model[:T]
        step_indices_all = step_indices_all[:T]

        if model_steps_only:
            kept = [i for i in range(T) if step_indices_all[i] < 0 or step_is_model[i]]
            if not kept:
                continue
            hs = hs[kept]
            step_indices = [int(step_indices_all[i]) for i in kept]
        else:
            step_indices = [int(x) for x in step_indices_all]

        error_steps = [e for e in meta["error_steps"] if e in set(step_indices)]
        if not meta["is_success"] and model_steps_only and not error_steps:
            continue

        trajectories.append(
            {
                **meta,
                "error_steps": error_steps,
                "hidden_states": hs,
                "step_indices": step_indices,
                "num_steps_original": meta["num_steps"],
                "num_steps": len(step_indices),
            }
        )
    return trajectories


def split_success_failure(trajectories: List[dict]) -> Tuple[List[dict], List[dict]]:
    return [t for t in trajectories if t["is_success"]], [t for t in trajectories if not t["is_success"]]


def fit_pca(trajectories: List[dict], n_components: int = config.LATENT_DIM) -> PCA:
    if config.ENCODER_TYPE != "pca":
        raise ValueError("OAT fixes encoder_type to pca.")
    X = np.concatenate([t["hidden_states"].numpy() for t in trajectories], axis=0)
    pca = PCA(n_components=n_components)
    pca.fit(X)
    return pca


def apply_pca(trajectories: List[dict], pca: PCA) -> List[dict]:
    for traj in trajectories:
        z = pca.transform(traj["hidden_states"].numpy()).astype(np.float32)
        traj["latent_states"] = torch.from_numpy(z)
    return trajectories


def compute_normalization_stats(trajectories: List[dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    all_z = torch.cat([t["latent_states"] for t in trajectories], dim=0)
    return all_z.mean(dim=0), all_z.std(dim=0).clamp(min=1e-8)


def normalize_trajectories(trajectories: List[dict], mean: torch.Tensor, std: torch.Tensor) -> List[dict]:
    for traj in trajectories:
        traj["latent_states"] = (traj["latent_states"] - mean) / std
    return trajectories


def get_time_points(length: int, device: Optional[torch.device] = None) -> torch.Tensor:
    t = torch.arange(length, dtype=torch.float32, device=device)
    if length > 1:
        t = t / (length - 1)
    return t


def prepare_data(
    dataset: str,
    layer: int = config.DEFAULT_LAYER,
    aggregation: str = config.HIDDEN_STATE_AGGREGATION,
    latent_dim: int = config.LATENT_DIM,
    model_steps_only: bool = config.MODEL_STEPS_ONLY,
) -> tuple[list[dict], list[dict], dict]:
    trajectories = load_trajectories(dataset, layer, aggregation, model_steps_only)
    success, failure = split_success_failure(trajectories)
    if not success:
        raise RuntimeError(f"No success trajectories found for dataset={dataset}; PCA and one-class OAT need success data.")
    pca = fit_pca(success, n_components=latent_dim)
    apply_pca(success, pca)
    apply_pca(failure, pca)
    mean, std = compute_normalization_stats(success)
    normalize_trajectories(success, mean, std)
    normalize_trajectories(failure, mean, std)
    return success, failure, {"type": "pca", "projector": pca, "latent_norm_mean": mean, "latent_norm_std": std}


def _collect_latents(trajectories: list[dict]) -> torch.Tensor:
    return torch.cat([t["latent_states"] for t in trajectories], dim=0) if trajectories else torch.empty(0)


def _matrix_sqrt_psd(mat: torch.Tensor, eps: float) -> torch.Tensor:
    eigvals, eigvecs = torch.linalg.eigh(mat)
    eigvals = torch.clamp(eigvals, min=eps)
    return eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.t()


def _matrix_invsqrt_psd(mat: torch.Tensor, eps: float) -> torch.Tensor:
    eigvals, eigvecs = torch.linalg.eigh(mat)
    eigvals = torch.clamp(eigvals, min=eps)
    return eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.t()


def apply_coral_alignment(target_trajectories: list[dict], source_trajectories: list[dict], eps: float = config.OOD_ALIGN_EPS) -> None:
    xs = _collect_latents(source_trajectories)
    xt = _collect_latents(target_trajectories)
    if xs.numel() == 0 or xt.numel() == 0:
        return
    ms = xs.mean(dim=0)
    mt = xt.mean(dim=0)
    xs0 = xs - ms
    xt0 = xt - mt
    d = xs.shape[1]
    eye = torch.eye(d, dtype=xs.dtype)
    cs = (xs0.t() @ xs0) / max(xs0.shape[0] - 1, 1) + eps * eye
    ct = (xt0.t() @ xt0) / max(xt0.shape[0] - 1, 1) + eps * eye
    transform = _matrix_invsqrt_psd(ct, eps) @ _matrix_sqrt_psd(cs, eps)
    for traj in target_trajectories:
        traj["latent_states"] = (traj["latent_states"] - mt) @ transform + ms
