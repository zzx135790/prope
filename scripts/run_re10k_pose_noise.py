#!/usr/bin/env python3
"""Launch the aligned PRoPE RealEstate10K pose-noise experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence


LOCKED_FINGERPRINTS = {
    "train_scene_count": 39,
    "train_scene_names_sha256": "11f615d4b74c7d65878d2a0353d9117949f6aa44deb8fb8ca0878698e1d1f610",
    "test_scene_count": 41,
    "test_scene_names_sha256": "8e40496da2124a948f47108833e06ca4c7087577d80de97b58c67d4526baf1c5",
    "index_sha256": "df5cbdf976e1afb05d7b6982b34c6b2da14e48ac09c36d5a1e90d3cc66172bfb",
    "effective_scene_count": 38,
    "effective_scene_names_sha256": "8f2e9370367016d96e3e5366824a6857200506e1a1bc2c37e7a5b7e7e6c8d2d8",
}

_REQUIRED_ENVIRONMENT = (
    "WORKSPACE_ROOT",
    "WORKSPACE_RAYROPE_WORKTREE",
    "WORKSPACE_PROPE_WORKTREE",
    "WORKSPACE_ARTIFACT_DIR",
    "WORKSPACE_LOG_DIR",
    "WORKSPACE_CHECKPOINT_DIR",
    "RE10K_TRAIN_DIR",
    "RE10K_TEST_DIR",
)


def _absolute_path(environment: Mapping[str, str], name: str) -> Path:
    value = environment.get(name)
    if not value:
        raise RuntimeError(f"workspace run environment must set {name}")
    path = Path(value)
    if not path.is_absolute():
        raise RuntimeError(f"{name} must be an absolute path: {value}")
    return path.resolve()


def _require_path(observed: Path, expected: Path, label: str) -> None:
    if observed != expected.resolve():
        raise RuntimeError(
            f"{label} must resolve to canonical {expected}, observed {observed}"
        )


def _scene_names(directory: Path, label: str) -> list[str]:
    if not directory.is_dir():
        raise RuntimeError(f"{label} directory does not exist: {directory}")
    return sorted(entry.name for entry in directory.iterdir() if entry.is_dir())


def _names_hash(names: Sequence[str]) -> str:
    payload = "".join(f"{name}\n" for name in sorted(names)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f"required file does not exist: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_index(index_path: Path, test_dir: Path, test_names: Sequence[str]) -> list[str]:
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read evaluation index {index_path}: {error}") from error
    if not isinstance(index, dict):
        raise RuntimeError("evaluation index must be a JSON object")

    effective: list[str] = []
    for scene_name in sorted(test_names):
        if scene_name not in index:
            raise RuntimeError(f"evaluation index has no entry for test scene {scene_name}")
        selection = index[scene_name]
        if selection is None:
            continue
        if not isinstance(selection, dict):
            raise RuntimeError(f"evaluation index entry {scene_name} must be an object or null")
        context = selection.get("context")
        target = selection.get("target")
        if not isinstance(context, list) or len(context) != 4:
            raise RuntimeError(f"evaluation index {scene_name} context must contain 4 indices")
        if not isinstance(target, list) or len(target) != 3:
            raise RuntimeError(f"evaluation index {scene_name} target must contain 3 indices")

        transforms_path = test_dir / scene_name / "transforms.json"
        try:
            transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
            frames = transforms["frames"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise RuntimeError(f"cannot read frames for test scene {scene_name}: {error}") from error
        if not isinstance(frames, list):
            raise RuntimeError(f"transforms frames for {scene_name} must be a list")
        for frame_index in context + target:
            if isinstance(frame_index, bool) or not isinstance(frame_index, int):
                raise RuntimeError(f"evaluation index {scene_name} contains a non-integer frame index")
            if frame_index < 0 or frame_index >= len(frames):
                raise RuntimeError(
                    f"evaluation index {scene_name} frame {frame_index} is outside [0, {len(frames)})"
                )
        effective.append(scene_name)
    return effective


def _check_fingerprints(observed: Mapping[str, object], expected: Mapping[str, object]) -> None:
    missing = sorted(set(LOCKED_FINGERPRINTS) - set(expected))
    if missing:
        raise RuntimeError(f"expected fingerprints are missing keys: {', '.join(missing)}")
    for key in LOCKED_FINGERPRINTS:
        if observed[key] != expected[key]:
            raise RuntimeError(
                f"canonical data fingerprint mismatch for {key}: "
                f"expected {expected[key]!r}, observed {observed[key]!r}"
            )


def _protocol_facts(paths: Mapping[str, Path], effective_count: int) -> dict[str, object]:
    return {
        "amp": False,
        "batch_scenes": 4,
        "checkpoint_every": 5000,
        "checkpoint_root": str(paths["output"] / "ckpts"),
        "effective_test_scenes": effective_count,
        "image_shape": [256, 256, 3],
        "camera_shape": [256, 256, 6],
        "learning_rate": 4e-4,
        "loss": {"mse": 1.0, "perceptual": 0.5},
        "metrics": ["PSNR", "SSIM", "LPIPS", "metrics_<level>.json"],
        "model": {"layers": 6, "d_model": 768, "nhead": 16, "ffn": 3072},
        "optimizer": {
            "name": "AdamW",
            "betas": [0.9, 0.95],
            "weight_decay": [0.5, 0.0],
        },
        "output_dir": str(paths["output"]),
        "patch_size": 8,
        "pose_noise": {
            "train_rot": [0.0, 0.03],
            "train_trans": [0.0, 0.03],
            "train_context_corrupt_count": [0, 2],
            "test_levels": [0.0, 0.01, 0.02, 0.03],
            "test_context_indices": [0, 1],
            "seed": 1234,
        },
        "prope": {"ray_encoding": "camray", "pos_enc": "prope", "implementation": "official"},
        "qk_norm": False,
        "scheduler": "ChainedScheduler(LinearLR,CosineAnnealingLR)",
        "seed": 0,
        "steps": 15000,
        "test_context_views": 4,
        "test_target_views": 3,
        "train_context_views": 4,
        "train_supervised_views": 1,
        "warmup_steps": 2500,
        "wandb": {"group": "re10k-pose-noise", "name": "pose_prope"},
    }


def _command(paths: Mapping[str, Path]) -> list[str]:
    return [
        sys.executable,
        "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
        "--nproc-per-node=1", "--module", "nvs.trainval", "lvsm",
        "--no-amp", "--fixed-seed", "--seed", "0",
        "--dataset", "re10k", "--dataset-patch-size", "256",
        "--dataset-input-views", "4", "--dataset-supervise-views", "1",
        "--dataset-batch-scenes", "4", "--test-input-views", "4",
        "--test-supervise-views", "3", "--test-index-fp", str(paths["index"]),
        "--max-steps", "15000", "--test-every", "15000",
        "--ckpt-every", "5000", "--print-every", "200",
        "--perceptual-loss-w", "0.5", "--lr", "4e-4", "--warmup-steps", "2500",
        "--model-config.ref-views", "4", "--model-config.tar-views", "1",
        "--model-config.img-shape", "256", "256", "3",
        "--model-config.cam-shape", "256", "256", "6",
        "--model-config.patch-size", "8",
        "--model-config.encoder.num-layers", "6",
        "--model-config.encoder.layer.d-model", "768",
        "--model-config.encoder.layer.nhead", "16",
        "--model-config.encoder.layer.dim-feedforward", "3072",
        "--model-config.encoder.layer.no-qk-norm",
        "--model-config.ray-encoding", "camray",
        "--model-config.pos-enc", "prope", "--model-config.prope-impl", "official",
        "--model-config.depth-type", "predict_dsig", "--model-config.init-sig", "6.0",
        "--pose-noise-enabled", "--pose-noise-rot-train-lo", "0.0",
        "--pose-noise-rot-train-hi", "0.03", "--pose-noise-trans-train-lo", "0.0",
        "--pose-noise-trans-train-hi", "0.03", "--pose-noise-max-corrupt", "2",
        "--pose-noise-test-levels", "0,0.01,0.02,0.03",
        "--pose-noise-test-corrupt", "2", "--pose-noise-seed", "1234",
        "--wandb-enabled", "--wandb-mode", "online",
        "--wandb-group", "re10k-pose-noise", "--wandb-name", "pose_prope",
        "--output-dir", str(paths["output"]),
    ]


def build_launch_spec(
    environment: Mapping[str, str],
    *,
    expected_fingerprints: Mapping[str, object] = LOCKED_FINGERPRINTS,
) -> dict[str, object]:
    paths = {name: _absolute_path(environment, name) for name in _REQUIRED_ENVIRONMENT}
    workspace = paths["WORKSPACE_ROOT"]
    if not (workspace / "workspace.toml").is_file():
        raise RuntimeError(f"WORKSPACE_ROOT has no workspace.toml marker: {workspace}")

    canonical = {
        "ray": workspace / "worktrees/baselines/rayrope/workspace",
        "prope": workspace / "worktrees/baselines/prope/workspace",
        "train": workspace / "data/datasets/re10k/train",
        "test": workspace / "data/datasets/re10k/test",
    }
    _require_path(paths["WORKSPACE_RAYROPE_WORKTREE"], canonical["ray"], "Ray worktree")
    _require_path(paths["WORKSPACE_PROPE_WORKTREE"], canonical["prope"], "PRoPE worktree")
    _require_path(paths["RE10K_TRAIN_DIR"], canonical["train"], "RE10K_TRAIN_DIR canonical re10k")
    _require_path(paths["RE10K_TEST_DIR"], canonical["test"], "RE10K_TEST_DIR canonical re10k")
    if not canonical["ray"].is_dir() or not canonical["prope"].is_dir():
        raise RuntimeError("canonical RayRoPE and PRoPE worktrees must exist")

    artifact = paths["WORKSPACE_ARTIFACT_DIR"]
    for key in ("WORKSPACE_LOG_DIR", "WORKSPACE_CHECKPOINT_DIR"):
        try:
            paths[key].relative_to(artifact)
        except ValueError as error:
            raise RuntimeError(f"{key} must stay under WORKSPACE_ARTIFACT_DIR") from error

    index_path = canonical["ray"] / "assets/evaluation_index_re10k_4ctx.json"
    train_names = _scene_names(canonical["train"], "RE10K train")
    test_names = _scene_names(canonical["test"], "RE10K test")
    effective_names = _validate_index(index_path, canonical["test"], test_names)
    data = {
        "train_scene_count": len(train_names),
        "train_scene_names_sha256": _names_hash(train_names),
        "test_scene_count": len(test_names),
        "test_scene_names_sha256": _names_hash(test_names),
        "index_sha256": _sha256(index_path),
        "effective_scene_count": len(effective_names),
        "effective_scene_names_sha256": _names_hash(effective_names),
    }
    _check_fingerprints(data, expected_fingerprints)

    launch_paths = {
        "ray": canonical["ray"],
        "index": index_path.resolve(),
        "output": artifact / "re10k_pose_prope",
    }
    return {
        "argv": _command(launch_paths),
        "cwd": str(canonical["ray"]),
        "data": data,
        "protocol": _protocol_facts(launch_paths, len(effective_names)),
        "runtime": {
            "workspace_root": str(workspace),
            "prope_worktree": str(canonical["prope"]),
            "rayrope_worktree": str(canonical["ray"]),
            "train_dir": str(canonical["train"]),
            "test_dir": str(canonical["test"]),
            "log_dir": str(paths["WORKSPACE_LOG_DIR"]),
            "workspace_checkpoint_dir": str(paths["WORKSPACE_CHECKPOINT_DIR"]),
        },
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    environment: Mapping[str, str] | None = None,
    expected_fingerprints: Mapping[str, object] = LOCKED_FINGERPRINTS,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    runtime_environment = os.environ if environment is None else environment
    report = build_launch_spec(
        runtime_environment, expected_fingerprints=expected_fingerprints
    )
    if args.check_only:
        payload = {"status": "ok", "mode": "check-only", **report}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    subprocess.run(
        report["argv"],
        cwd=report["cwd"],
        env=dict(runtime_environment),
        check=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(json.dumps({"status": "error", "error": str(error)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
