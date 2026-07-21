from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "run_re10k_pose_noise.py"


def _load_launcher():
    assert SCRIPT.is_file(), f"missing aligned launcher: {SCRIPT}"
    spec = importlib.util.spec_from_file_location("run_re10k_pose_noise", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _names_hash(names: list[str]) -> str:
    return hashlib.sha256("".join(f"{name}\n" for name in sorted(names)).encode()).hexdigest()


@pytest.fixture
def synthetic_workspace(tmp_path: Path):
    workspace = tmp_path / "workspace"
    ray = workspace / "worktrees/baselines/rayrope/workspace"
    prope = workspace / "worktrees/baselines/prope/workspace"
    train = workspace / "data/datasets/re10k/train"
    test = workspace / "data/datasets/re10k/test"
    artifact = workspace / "artifacts/runs/prope"
    log = artifact / "logs"
    checkpoint = artifact / "checkpoints"
    for directory in (ray / "assets", prope, train, test, artifact, log, checkpoint):
        directory.mkdir(parents=True, exist_ok=True)
    (workspace / "workspace.toml").write_text("schema_version = 3\n", encoding="utf-8")

    train_names = ["train-a", "train-b"]
    test_names = ["scene-a", "scene-b", "scene-c"]
    for name in train_names:
        (train / name).mkdir()
    for name in test_names:
        scene = test / name
        scene.mkdir()
        (scene / "transforms.json").write_text(
            json.dumps({"frames": [{} for _ in range(8)]}), encoding="utf-8"
        )

    index = {
        "scene-a": {"context": [0, 1, 2, 3], "target": [4, 5, 6]},
        "scene-b": None,
        "scene-c": {"context": [1, 2, 3, 4], "target": [5, 6, 7]},
    }
    index_path = ray / "assets/evaluation_index_re10k_4ctx.json"
    index_path.write_text(json.dumps(index, sort_keys=True), encoding="utf-8")

    environment = {
        "WORKSPACE_ROOT": str(workspace),
        "WORKSPACE_RAYROPE_WORKTREE": str(ray),
        "WORKSPACE_PROPE_WORKTREE": str(prope),
        "WORKSPACE_ARTIFACT_DIR": str(artifact),
        "WORKSPACE_LOG_DIR": str(log),
        "WORKSPACE_CHECKPOINT_DIR": str(checkpoint),
        "RE10K_TRAIN_DIR": str(train),
        "RE10K_TEST_DIR": str(test),
    }
    expected = {
        "train_scene_count": len(train_names),
        "train_scene_names_sha256": _names_hash(train_names),
        "test_scene_count": len(test_names),
        "test_scene_names_sha256": _names_hash(test_names),
        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "effective_scene_count": 2,
        "effective_scene_names_sha256": _names_hash(["scene-a", "scene-c"]),
    }
    return environment, expected


def test_preflight_validates_synthetic_inventory_and_indices(synthetic_workspace) -> None:
    launcher = _load_launcher()
    environment, expected = synthetic_workspace

    report = launcher.build_launch_spec(environment, expected_fingerprints=expected)

    assert report["data"] == expected
    assert report["protocol"]["effective_test_scenes"] == 2
    assert report["protocol"]["test_context_views"] == 4
    assert report["protocol"]["test_target_views"] == 3


def test_preflight_rejects_re10k_lvsm_binding(synthetic_workspace) -> None:
    launcher = _load_launcher()
    environment, expected = synthetic_workspace
    workspace = Path(environment["WORKSPACE_ROOT"])
    legacy_train = workspace / "data/datasets/re10k-lvsm/train"
    legacy_test = workspace / "data/datasets/re10k-lvsm/test"
    legacy_train.mkdir(parents=True)
    legacy_test.mkdir(parents=True)
    environment["RE10K_TRAIN_DIR"] = str(legacy_train)
    environment["RE10K_TEST_DIR"] = str(legacy_test)

    with pytest.raises(RuntimeError, match="canonical re10k"):
        launcher.build_launch_spec(environment, expected_fingerprints=expected)


def test_launcher_argv_fully_locks_the_aligned_protocol(synthetic_workspace) -> None:
    launcher = _load_launcher()
    environment, expected = synthetic_workspace
    report = launcher.build_launch_spec(environment, expected_fingerprints=expected)
    ray = environment["WORKSPACE_RAYROPE_WORKTREE"]
    output = f'{environment["WORKSPACE_ARTIFACT_DIR"]}/re10k_pose_prope'
    index = f"{ray}/assets/evaluation_index_re10k_4ctx.json"

    assert report["cwd"] == ray
    assert report["argv"] == [
        sys.executable,
        "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
        "--nproc-per-node=1", "--module", "nvs.trainval", "lvsm",
        "--no-amp", "--fixed-seed", "--seed", "0",
        "--dataset", "re10k", "--dataset-patch-size", "256",
        "--dataset-input-views", "4", "--dataset-supervise-views", "1",
        "--dataset-batch-scenes", "4", "--test-input-views", "4",
        "--test-supervise-views", "3", "--test-index-fp", index,
        "--max-steps", "15000", "--test-every", "15000",
        "--ckpt-every", "5000", "--print-every", "200",
        "--perceptual-loss-w", "0.5", "--lr", "4e-4",
        "--warmup-steps", "2500",
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
        "--model-config.pos-enc", "prope",
        "--model-config.prope-impl", "official",
        "--model-config.depth-type", "predict_dsig",
        "--model-config.init-sig", "6.0",
        "--pose-noise-enabled", "--pose-noise-rot-train-lo", "0.0",
        "--pose-noise-rot-train-hi", "0.03",
        "--pose-noise-trans-train-lo", "0.0",
        "--pose-noise-trans-train-hi", "0.03",
        "--pose-noise-max-corrupt", "2",
        "--pose-noise-test-levels", "0,0.01,0.02,0.03",
        "--pose-noise-test-corrupt", "2", "--pose-noise-seed", "1234",
        "--wandb-enabled", "--wandb-mode", "online",
        "--wandb-project", "tokenmap-flag-rope",
        "--wandb-group", "re10k-pose-noise",
        "--wandb-name", "pose_prope",
        "--wandb-id", "prope-prope", "--wandb-resume", "never",
        "--wandb-required", "--output-dir", output,
    ]
    assert report["protocol"]["optimizer"] == {
        "name": "AdamW", "betas": [0.9, 0.95], "weight_decay": [0.5, 0.0]
    }
    assert report["protocol"]["scheduler"] == "ChainedScheduler(LinearLR,CosineAnnealingLR)"
    assert report["protocol"]["metrics"] == ["PSNR", "SSIM", "LPIPS", "metrics_<level>.json"]
    assert report["runtime"]["wandb_id"] == "prope-prope"
    assert report["runtime"]["wandb_required"] is True
    assert "--ckpt-subdir" not in report["argv"]
    assert report["protocol"]["checkpoint_root"] == f"{output}/ckpts"
    assert report["runtime"]["workspace_checkpoint_dir"] == environment[
        "WORKSPACE_CHECKPOINT_DIR"
    ]
    assert "checkpoint_dir" not in report["runtime"]


def test_check_only_prints_json_without_dispatch_or_output_creation(
    synthetic_workspace, monkeypatch, capsys
) -> None:
    launcher = _load_launcher()
    environment, expected = synthetic_workspace
    output = Path(environment["WORKSPACE_ARTIFACT_DIR"]) / "re10k_pose_prope"

    def forbidden_dispatch(*args, **kwargs):
        raise AssertionError("check-only must not dispatch the training process")

    monkeypatch.setattr(launcher.subprocess, "run", forbidden_dispatch)
    result = launcher.main(
        ["--check-only"], environment=environment, expected_fingerprints=expected
    )

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["status"] == "ok"
    assert payload["mode"] == "check-only"
    assert payload["argv"][0] == sys.executable
    assert not output.exists()


def test_resume_requires_explicit_wandb_identity(synthetic_workspace, tmp_path) -> None:
    launcher = _load_launcher()
    environment, expected = synthetic_workspace
    checkpoint = tmp_path / "step-000005000.pt"
    checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(RuntimeError, match="requires --wandb-run-id"):
        launcher.build_launch_spec(
            environment,
            expected_fingerprints=expected,
            resume=checkpoint,
        )


def test_resume_and_test_only_continue_same_wandb_run(synthetic_workspace, tmp_path) -> None:
    launcher = _load_launcher()
    environment, expected = synthetic_workspace
    checkpoint = tmp_path / "step-000015000.pt"
    checkpoint.write_bytes(b"checkpoint")

    report = launcher.build_launch_spec(
        environment,
        expected_fingerprints=expected,
        resume=checkpoint,
        wandb_run_id="original-prope-run",
        test_only=True,
    )

    assert report["runtime"]["mode"] == "test-only"
    assert report["runtime"]["wandb_id"] == "original-prope-run"
    assert report["runtime"]["wandb_resume"] == "must"
    assert report["argv"][-3:] == [
        "--resume",
        str(checkpoint.resolve()),
        "--test-only",
    ]
    assert report["argv"][report["argv"].index("--wandb-resume") + 1] == "must"


def test_pilot_preserves_full_model_and_changes_only_runtime_scale(synthetic_workspace) -> None:
    launcher = _load_launcher()
    environment, expected = synthetic_workspace

    report = launcher.build_launch_spec(
        environment,
        expected_fingerprints=expected,
        pilot=True,
    )
    argv = report["argv"]

    assert report["runtime"]["mode"] == "pilot"
    assert report["protocol"]["execution_mode"] == "pilot"
    assert report["protocol"]["model"] == {
        "layers": 6,
        "d_model": 768,
        "nhead": 16,
        "ffn": 3072,
    }
    assert argv[argv.index("--max-steps") + 1] == "1"
    assert argv[argv.index("--test-every") + 1] == "-1"
    assert argv[argv.index("--dataset-batch-scenes") + 1] == "4"
    assert argv[argv.index("--wandb-group") + 1] == "re10k-pose-noise-pilot"
    assert argv[argv.index("--wandb-name") + 1] == "pilot_pose_prope"
    assert report["runtime"]["wandb_id"] == "prope-prope-pilot"
    assert argv[-2:] == [
        "--output-dir",
        f'{environment["WORKSPACE_ARTIFACT_DIR"]}/re10k_pose_prope_pilot',
    ]


def test_normal_mode_dispatches_only_after_preflight(
    synthetic_workspace, monkeypatch
) -> None:
    launcher = _load_launcher()
    environment, expected = synthetic_workspace
    calls = []

    def capture_dispatch(argv, **kwargs):
        calls.append((argv, kwargs))

    monkeypatch.setattr(launcher.subprocess, "run", capture_dispatch)

    result = launcher.main([], environment=environment, expected_fingerprints=expected)

    assert result == 0
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv[0] == sys.executable
    assert kwargs["cwd"] == environment["WORKSPACE_RAYROPE_WORKTREE"]
    assert kwargs["env"]["WORKSPACE_PROPE_WORKTREE"] == environment["WORKSPACE_PROPE_WORKTREE"]
    assert kwargs["check"] is True


def test_real_canonical_dataset_passes_locked_preflight() -> None:
    launcher = _load_launcher()
    workspace = REPOSITORY_ROOT.parents[3]
    artifact = workspace / "artifacts/test-preflight-placeholder"
    environment = {
        "WORKSPACE_ROOT": str(workspace),
        "WORKSPACE_RAYROPE_WORKTREE": str(workspace / "worktrees/baselines/rayrope/workspace"),
        "WORKSPACE_PROPE_WORKTREE": str(REPOSITORY_ROOT),
        "WORKSPACE_ARTIFACT_DIR": str(artifact),
        "WORKSPACE_LOG_DIR": str(artifact / "logs"),
        "WORKSPACE_CHECKPOINT_DIR": str(artifact / "checkpoints"),
        "RE10K_TRAIN_DIR": str(workspace / "data/datasets/re10k/train"),
        "RE10K_TEST_DIR": str(workspace / "data/datasets/re10k/test"),
    }

    report = launcher.build_launch_spec(environment)

    assert report["data"]["train_scene_count"] == 39
    assert report["data"]["test_scene_count"] == 41
    assert report["data"]["effective_scene_count"] == 38
