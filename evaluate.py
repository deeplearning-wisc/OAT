"""Focused OAT metrics."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import config


def _scored_steps_and_scores(prediction: dict) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(prediction["scores"], dtype=float)
    step_ids = np.asarray(prediction.get("score_step_indices", list(range(len(scores)))), dtype=int)
    if len(step_ids) != len(scores):
        step_ids = np.arange(len(scores), dtype=int)
    return step_ids, scores


def _set_metrics(predictions: List[dict], detection_key: str) -> dict:
    precisions, recalls, f1s, hits = [], [], [], []
    for pred in predictions:
        decoded = pred.get(detection_key, [pred["predicted_step"]])
        pred_set = set(int(x) for x in decoded)
        gt_set = set(int(x) for x in pred.get("gt_error_steps", []))
        inter = pred_set & gt_set
        precision = len(inter) / len(pred_set) if pred_set else 0.0
        recall = len(inter) / len(gt_set) if gt_set else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        precisions.append(float(precision))
        recalls.append(float(recall))
        f1s.append(float(f1))
        hits.append(1.0 if inter else 0.0)
    return {
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "f1": float(np.mean(f1s)) if f1s else 0.0,
        "hit_rate": float(np.mean(hits)) if hits else 0.0,
    }


def step_auroc(predictions: List[dict]) -> float:
    labels, scores_all = [], []
    for pred in predictions:
        gt_set = set(pred.get("gt_error_steps", []))
        step_ids, scores = _scored_steps_and_scores(pred)
        for step_id, score in zip(step_ids, scores):
            labels.append(1 if int(step_id) in gt_set else 0)
            scores_all.append(float(score))
    if len(set(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, scores_all))


def step_auprc(predictions: List[dict]) -> float:
    labels, scores_all = [], []
    for pred in predictions:
        gt_set = set(pred.get("gt_error_steps", []))
        step_ids, scores = _scored_steps_and_scores(pred)
        for step_id, score in zip(step_ids, scores):
            labels.append(1 if int(step_id) in gt_set else 0)
            scores_all.append(float(score))
    if len(set(labels)) < 2:
        return 0.5
    return float(average_precision_score(labels, scores_all))


def compute_metrics(predictions: List[dict]) -> dict:
    topk = _set_metrics(predictions, detection_key="topk_detection")
    conformal = _set_metrics(predictions, detection_key="conformal_detection")
    metrics = {f"topk_detection_{key}": value for key, value in topk.items()}
    metrics.update({f"conformal_detection_{key}": value for key, value in conformal.items()})
    metrics["auroc"] = step_auroc(predictions)
    metrics["auprc"] = step_auprc(predictions)
    metrics["n"] = len(predictions)
    return metrics


def aggregate_multi_seed(all_seed_metrics: List[dict]) -> dict:
    out = {}
    if not all_seed_metrics:
        return out
    for key in config.REPORT_METRICS:
        vals = [m[key] for m in all_seed_metrics if key in m]
        if vals:
            out[f"{key}_mean"] = float(np.mean(vals))
            out[f"{key}_std"] = float(np.std(vals))
    out["n"] = int(sum(m.get("n", 0) for m in all_seed_metrics))
    return out


def print_metrics(metrics: dict, title: str = "") -> None:
    if title:
        print(f"\n{title}")
        print("=" * len(title))
    for key in config.REPORT_METRICS:
        mean_key = f"{key}_mean"
        std_key = f"{key}_std"
        if mean_key in metrics and std_key in metrics:
            print(f"{key:10s}: {metrics[mean_key]:.4f} +/- {metrics[std_key]:.4f}")
        elif key in metrics:
            print(f"{key:10s}: {metrics[key]:.4f}")
    if "n" in metrics:
        print(f"{'n':10s}: {metrics['n']}")


def print_comparison_table(results: Dict[str, dict], metrics_keys: Optional[List[str]] = None) -> None:
    keys = metrics_keys or list(config.REPORT_METRICS)
    header = f"{'method':24s}" + "".join(f"{k:>12s}" for k in keys)
    print("\n" + header)
    print("-" * len(header))
    for name, metrics in results.items():
        row = f"{name:24s}"
        for key in keys:
            if key in metrics:
                row += f"{metrics[key]:12.4f}"
            elif f"{key}_mean" in metrics:
                row += f"{metrics[f'{key}_mean']:12.4f}"
            else:
                row += f"{'N/A':>12s}"
        print(row)
