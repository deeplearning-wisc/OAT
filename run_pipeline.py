"""Focused OAT command-line pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import config
from baselines import run_baselines
from data_pipeline import (
    apply_coral_alignment,
    apply_pca,
    compute_normalization_stats,
    fit_pca,
    load_trajectories,
    normalize_trajectories,
    prepare_data,
    split_success_failure,
)
from evaluate import aggregate_multi_seed, compute_metrics, print_comparison_table, print_metrics
from train import (
    build_model,
    collect_calibration_scores,
    conformal_quantile_threshold,
    score_failure_trajectories,
    split_train_val,
    train_model,
)


def _jsonable_prediction(pred: dict) -> dict:
    out = dict(pred)
    if hasattr(out.get("scores"), "tolist"):
        out["scores"] = out["scores"].tolist()
    return out


def _save_outputs(out_dir: Path, metrics: dict, predictions: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "predictions.json", "w") as f:
        json.dump([_jsonable_prediction(p) for p in predictions], f, indent=2)
    print(f"Saved metrics and predictions to {out_dir}")


def cmd_extract(args) -> None:
    from extract_states import extract_all, load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        device=args.device,
        attn_implementation=args.attn_implementation,
    )
    extract_all(args.dataset, model, tokenizer, aggregation=args.aggregation, force=args.force)


def cmd_baselines(args) -> dict:
    trajectories = load_trajectories(
        dataset=args.dataset,
        layer=args.layer,
        aggregation=args.aggregation,
        model_steps_only=args.model_steps_only,
    )
    _, failure = split_success_failure(trajectories)
    results = {}
    predictions_by_name = run_baselines(
        failure,
        include_prompt=args.include_all_at_once_prompt,
        backend_name=args.llm_backend,
        model_name=args.llm_model_name,
    )
    for name, preds in predictions_by_name.items():
        results[name] = compute_metrics(preds)
        out_dir = config.RESULTS_DIR / args.dataset / "baselines" / name
        _save_outputs(out_dir, results[name], preds)
    print_comparison_table(results)
    return results


def _train_eval_once(success: list[dict], failure: list[dict], seed: int, args) -> tuple[dict, list[dict]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_set, val_set = split_train_val(success, seed=seed)
    model = build_model(latent_dim=args.latent_dim)
    model, history = train_model(
        model,
        train_set,
        val_set,
        device=args.device,
        epochs=args.epochs,
        patience=args.patience,
    )
    print(f"best_val_loss={history['best_val_loss']:.6f} best_epoch={history['best_epoch']}")
    calibration_scores = collect_calibration_scores(model, val_set, device=args.device)
    conformal_threshold = conformal_quantile_threshold(
        calibration_scores,
        alpha=args.conformal_alpha,
    )
    preds = score_failure_trajectories(
        model,
        failure,
        device=args.device,
        top_k=args.detection_top_k,
        conformal_threshold=conformal_threshold,
        conformal_min_detections=args.conformal_min_detections,
    )
    return compute_metrics(preds), preds


def cmd_train(args) -> dict:
    success, failure, projector_bundle = prepare_data(
        dataset=args.dataset,
        layer=args.layer,
        aggregation=args.aggregation,
        latent_dim=args.latent_dim,
        model_steps_only=args.model_steps_only,
    )
    pca = projector_bundle["projector"]
    print(f"Loaded {len(success)} success + {len(failure)} failure trajectories")
    print(f"PCA explained variance: {sum(pca.explained_variance_ratio_) * 100:.1f}%")

    seed_metrics = []
    last_preds = []
    for run_idx in range(args.n_seeds):
        seed = config.RANDOM_SEED + run_idx
        print(f"\nSeed {run_idx + 1}/{args.n_seeds} (seed={seed})")
        metrics, preds = _train_eval_once(success, failure, seed, args)
        seed_metrics.append(metrics)
        last_preds = preds
        print_metrics(metrics, title=f"OAT seed {run_idx + 1}")

    agg = aggregate_multi_seed(seed_metrics)
    print_metrics(agg, title="OAT aggregate")
    out_dir = config.RESULTS_DIR / args.dataset / "OAT"
    _save_outputs(out_dir, agg, last_preds)
    return {"OAT": agg}


def cmd_ood(args) -> dict:
    source = load_trajectories(
        dataset=args.train_dataset,
        layer=args.layer,
        aggregation=args.aggregation,
        model_steps_only=args.model_steps_only,
    )
    target = load_trajectories(
        dataset=args.test_dataset,
        layer=args.layer,
        aggregation=args.aggregation,
        model_steps_only=args.model_steps_only,
    )
    source_success, _source_failure = split_success_failure(source)
    _target_success, target_failure = split_success_failure(target)
    if not source_success:
        raise RuntimeError(f"No success trajectories in train_dataset={args.train_dataset}")
    if not target_failure:
        raise RuntimeError(f"No failure trajectories in test_dataset={args.test_dataset}")

    pca = fit_pca(source_success, n_components=args.latent_dim)
    apply_pca(source_success, pca)
    apply_pca(target_failure, pca)
    mean, std = compute_normalization_stats(source_success)
    normalize_trajectories(source_success, mean, std)
    normalize_trajectories(target_failure, mean, std)
    apply_coral_alignment(target_failure, source_success, eps=args.ood_align_eps)
    print(f"Loaded OOD source_success={len(source_success)} target_failure={len(target_failure)}")
    print(f"PCA explained variance: {sum(pca.explained_variance_ratio_) * 100:.1f}%")
    print("Applied OOD alignment: coral")

    seed_metrics = []
    last_preds = []
    for run_idx in range(args.n_seeds):
        seed = config.RANDOM_SEED + run_idx
        print(f"\nSeed {run_idx + 1}/{args.n_seeds} (seed={seed})")
        metrics, preds = _train_eval_once(source_success, target_failure, seed, args)
        seed_metrics.append(metrics)
        last_preds = preds
        print_metrics(metrics, title=f"OOD OAT seed {run_idx + 1}")

    agg = aggregate_multi_seed(seed_metrics)
    print_metrics(agg, title="OOD OAT aggregate")
    out_dir = config.RESULTS_DIR / "ood" / f"{args.train_dataset}_to_{args.test_dataset}" / "OAT_coral"
    _save_outputs(out_dir, agg, last_preds)
    return {"OAT_coral": agg}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OAT: focused failure attribution pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", choices=config.DATASETS, default=config.DEFAULT_DATASET)
    common.add_argument("--layer", type=int, default=config.DEFAULT_LAYER)
    common.add_argument("--aggregation", choices=["last", "mean"], default=config.HIDDEN_STATE_AGGREGATION)
    common.add_argument("--model-steps-only", action=argparse.BooleanOptionalAction, default=config.MODEL_STEPS_ONLY)
    common.add_argument("--device", default=config.DEVICE)

    p_extract = subparsers.add_parser("extract", parents=[common], help="Extract proxy hidden states")
    p_extract.add_argument("--model-name", default=config.MODEL_NAME)
    p_extract.add_argument("--attn-implementation", default=None)
    p_extract.add_argument("--force", action="store_true")

    p_baselines = subparsers.add_parser("baselines", parents=[common], help="Run selected baselines")
    p_baselines.add_argument("--include-all-at-once-prompt", action="store_true")
    p_baselines.add_argument("--llm-backend", choices=["vllm", "azure"], default=config.LLM_JUDGE_BACKEND)
    p_baselines.add_argument("--llm-model-name", default=config.LLM_JUDGE_MODEL_NAME)

    p_train = subparsers.add_parser("train", parents=[common], help="Train/evaluate OAT")
    p_train.add_argument("--latent-dim", type=int, default=config.LATENT_DIM)
    p_train.add_argument("--n-seeds", type=int, default=config.N_SEED_RUNS)
    p_train.add_argument("--epochs", type=int, default=config.EPOCHS)
    p_train.add_argument("--patience", type=int, default=config.PATIENCE)
    p_train.add_argument("--detection-top-k", type=int, default=config.DETECTION_TOP_K)
    p_train.add_argument("--conformal-alpha", type=float, default=config.CONFORMAL_ALPHA)
    p_train.add_argument("--conformal-min-detections", type=int, default=config.CONFORMAL_MIN_DETECTIONS)

    p_ood = subparsers.add_parser("ood", parents=[common], help="Train on one dataset, test on another with CORAL")
    p_ood.add_argument("--train-dataset", choices=config.DATASETS, default="mcp_atlas")
    p_ood.add_argument("--test-dataset", choices=config.DATASETS, default="who_and_when")
    p_ood.add_argument("--latent-dim", type=int, default=config.LATENT_DIM)
    p_ood.add_argument("--n-seeds", type=int, default=config.N_SEED_RUNS)
    p_ood.add_argument("--epochs", type=int, default=config.EPOCHS)
    p_ood.add_argument("--patience", type=int, default=config.PATIENCE)
    p_ood.add_argument("--detection-top-k", type=int, default=config.DETECTION_TOP_K)
    p_ood.add_argument("--conformal-alpha", type=float, default=config.CONFORMAL_ALPHA)
    p_ood.add_argument("--conformal-min-detections", type=int, default=config.CONFORMAL_MIN_DETECTIONS)
    p_ood.add_argument("--ood-align-eps", type=float, default=config.OOD_ALIGN_EPS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "extract":
        cmd_extract(args)
    elif args.command == "baselines":
        cmd_baselines(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "ood":
        cmd_ood(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
