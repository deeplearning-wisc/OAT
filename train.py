"""Training and scoring utilities for OAT."""

from __future__ import annotations

import copy
import time
from typing import List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split

import config
from data_pipeline import get_time_points
from models.oat import OATModel


def build_model(latent_dim: int = config.LATENT_DIM) -> OATModel:
    return OATModel(latent_dim=latent_dim)


def _forward_one(model: torch.nn.Module, traj: dict, device: str) -> dict:
    z = traj["latent_states"].to(device)
    t = get_time_points(z.shape[0], device=z.device)
    return model(z, t)


def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    trajectories: List[dict],
    device: str = config.DEVICE,
    batch_size: int = config.BATCH_SIZE,
) -> float:
    model.train()
    indices = np.random.permutation(len(trajectories))
    total_loss = 0.0
    n_trajs = 0
    for start in range(0, len(indices), batch_size):
        optimizer.zero_grad()
        batch_loss = torch.tensor(0.0, device=device)
        valid = 0
        for idx in indices[start : start + batch_size]:
            traj = trajectories[int(idx)]
            if traj["latent_states"].shape[0] < 2:
                continue
            out = _forward_one(model, traj, device)
            batch_loss = batch_loss + out["loss"]
            valid += 1
        if valid:
            loss = batch_loss / valid
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(batch_loss.item())
            n_trajs += valid
    return total_loss / max(n_trajs, 1)


def validate(model: torch.nn.Module, trajectories: List[dict], device: str = config.DEVICE) -> float:
    model.eval()
    total_loss = 0.0
    n_trajs = 0
    with torch.no_grad():
        for traj in trajectories:
            if traj["latent_states"].shape[0] < 2:
                continue
            total_loss += float(_forward_one(model, traj, device)["loss"].item())
            n_trajs += 1
    return total_loss / max(n_trajs, 1)


def train_model(
    model: torch.nn.Module,
    train_trajs: List[dict],
    val_trajs: List[dict],
    device: str = config.DEVICE,
    lr: float = config.LEARNING_RATE,
    epochs: int = config.EPOCHS,
    patience: int = config.PATIENCE,
) -> tuple[torch.nn.Module, dict]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    wait = 0
    history = {"train_loss": [], "val_loss": [], "best_val_loss": best_val, "best_epoch": best_epoch}

    for epoch in range(int(epochs)):
        train_loss = train_epoch(model, optimizer, train_trajs, device=device)
        val_loss = validate(model, val_trajs, device=device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"epoch {epoch + 1:03d}/{epochs} train={train_loss:.6f} val={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    history["best_val_loss"] = best_val
    history["best_epoch"] = best_epoch
    return model, history


def _topk_detection(scores: np.ndarray, step_indices: List[int], k: int) -> List[int]:
    if scores.size == 0:
        return []
    k_eff = max(1, min(int(k), int(scores.size)))
    return sorted(int(step_indices[i]) for i in np.argsort(-scores)[:k_eff])


def _scorable_scores_and_indices(scores: torch.Tensor, traj: dict) -> Tuple[np.ndarray, List[int]]:
    scores_np = np.asarray(scores.detach().cpu().numpy(), dtype=float)
    step_indices = [int(x) for x in traj.get("step_indices", list(range(len(scores_np))))]
    if len(step_indices) != len(scores_np):
        step_indices = list(range(len(scores_np)))
    kept = [i for i, step_idx in enumerate(step_indices) if step_idx >= 0]
    return scores_np[kept], [step_indices[i] for i in kept]


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    scale = max(1.4826 * mad, 1e-8)
    return (scores - median) / scale


def conformal_quantile_threshold(calibration_scores: np.ndarray, alpha: float = config.CONFORMAL_ALPHA) -> float:
    if calibration_scores.size == 0:
        return float("inf")
    alpha = float(np.clip(alpha, 1e-8, 1.0 - 1e-8))
    n = calibration_scores.size
    q = float(np.ceil((n + 1) * (1.0 - alpha)) / n)
    q = float(np.clip(q, 0.0, 1.0))
    try:
        return float(np.quantile(calibration_scores, q, method="higher"))
    except TypeError:
        return float(np.quantile(calibration_scores, q, interpolation="higher"))


def collect_calibration_scores(
    model: torch.nn.Module,
    calibration_trajectories: List[dict],
    device: str = config.DEVICE,
) -> np.ndarray:
    model.eval()
    all_scores: list[np.ndarray] = []
    with torch.no_grad():
        for traj in calibration_trajectories:
            z = traj["latent_states"].to(device)
            if z.shape[0] < 2:
                continue
            t = get_time_points(z.shape[0], device=z.device)
            scores = model.anomaly_scores(z, t)
            scores_np, _ = _scorable_scores_and_indices(scores, traj)
            if scores_np.size > 0:
                all_scores.append(_normalize_scores(scores_np))
    if not all_scores:
        return np.empty(0, dtype=float)
    return np.concatenate(all_scores, axis=0)


def _conformal_detection(
    scores: np.ndarray,
    step_indices: List[int],
    threshold: float,
    min_detections: int,
) -> List[int]:
    normalized = _normalize_scores(scores)
    detected = [
        int(step_idx)
        for step_idx, score in zip(step_indices, normalized)
        if float(score) > float(threshold)
    ]
    if not detected and int(min_detections) > 0:
        return _topk_detection(normalized, step_indices, int(min_detections))
    return sorted(detected)


def score_failure_trajectories(
    model: torch.nn.Module,
    failures: List[dict],
    device: str = config.DEVICE,
    top_k: int = config.DETECTION_TOP_K,
    conformal_threshold: float = float("inf"),
    conformal_min_detections: int = config.CONFORMAL_MIN_DETECTIONS,
) -> List[dict]:
    model.eval()
    predictions = []
    with torch.no_grad():
        for traj in failures:
            z = traj["latent_states"].to(device)
            if z.shape[0] < 2:
                continue
            t = get_time_points(z.shape[0], device=z.device)
            if torch.cuda.is_available() and torch.device(device).type == "cuda":
                torch.cuda.synchronize(torch.device(device))
            start = time.perf_counter()
            scores = model.anomaly_scores(z, t)
            if torch.cuda.is_available() and torch.device(device).type == "cuda":
                torch.cuda.synchronize(torch.device(device))
            elapsed = time.perf_counter() - start
            scores_np, step_indices = _scorable_scores_and_indices(scores, traj)
            if len(step_indices) == 0:
                continue
            pred_local = int(np.argmax(scores_np))
            pred_step = int(step_indices[pred_local])
            predictions.append(
                {
                    "trajectory_id": traj["id"],
                    "predicted_step": pred_step,
                    "predicted_local_step": pred_local,
                    "score_step_indices": step_indices,
                    "scores": scores_np.astype(np.float32),
                    "topk_detection": _topk_detection(scores_np, step_indices, top_k),
                    "conformal_detection": _conformal_detection(
                        scores_np,
                        step_indices,
                        conformal_threshold,
                        conformal_min_detections,
                    ),
                    "gt_error_steps": traj["error_steps"],
                    "num_steps": traj.get("num_steps_original", traj["num_steps"]),
                    "num_scored_steps": len(step_indices),
                    "inference_time_seconds": elapsed,
                    "inference_time_ms": elapsed * 1000.0,
                }
            )
    return predictions


def split_train_val(success: List[dict], seed: int = config.RANDOM_SEED) -> tuple[list[dict], list[dict]]:
    if len(success) < 2:
        raise RuntimeError("Need at least two success trajectories for train/validation split.")
    train_idx, val_idx = train_test_split(
        np.arange(len(success)),
        test_size=0.2,
        random_state=seed,
        shuffle=True,
    )
    return [success[int(i)] for i in train_idx], [success[int(i)] for i in val_idx]
