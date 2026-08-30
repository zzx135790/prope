from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest
import torch

from nvs.dataset_contract import ContractEvalDataset
from rope_contract.dataset.migration import DatasetParityError


def _fixture(root: Path) -> Path:
    scene = root / "test" / "scene-a"
    images = scene / "images"
    images.mkdir(parents=True)
    frames = []
    for index in range(5):
        Image.new("RGB", (64, 48), color=(index, index + 1, index + 2)).save(images / f"{index:05d}.png")
        frames.append({
            "file_path": f"images/{index:05d}.png",
            "transform_matrix": [
                [1, 0, 0, index],
                [0, 1, 0, 2 * index],
                [0, 0, 1, 3 * index],
                [0, 0, 0, 1],
            ],
        })
    (scene / "transforms.json").write_text(json.dumps({
        "w": 64, "h": 48, "fl_x": 40, "fl_y": 41, "cx": 32, "cy": 24, "frames": frames,
    }), encoding="utf-8")
    index = root / "index.json"
    index.write_text(json.dumps({"scene-a": {"context": [0, 4], "target": [1, 2, 3]}}), encoding="utf-8")
    return index


def _environment(monkeypatch, root: Path, mode: str, artifact: Path):
    for key, value in {
        "WORKSPACE_DATASET_ID": "re10k",
        "WORKSPACE_DATASET_LOGICAL_ID": "re10k",
        "WORKSPACE_DATASET_ROOT": str(root),
        "WORKSPACE_DATASET_PROVIDER_ID": "re10k-transforms-v1",
        "WORKSPACE_DATASET_CONTRACT_VERSION": "1.0.0",
        "WORKSPACE_DATASET_PROFILE": "multiview-nvs-v1",
        "WORKSPACE_DATASET_CONSUMER_ADAPTER": "prope-multiview-nvs-v1",
        "WORKSPACE_DATASET_MODE": mode,
        "WORKSPACE_ARTIFACT_DIR": str(artifact),
    }.items():
        monkeypatch.setenv(key, value)


def test_prope_contract_eval_preserves_canonical_opencv_pose(monkeypatch, tmp_path):
    root = tmp_path / "re10k"
    index = _fixture(root)
    _environment(monkeypatch, root, "contract", tmp_path / "artifacts")
    contract = ContractEvalDataset(
        str(root / "test"), patch_size=32, input_views=2, supervise_views=3,
        test_index_fp=str(index.resolve()),
    )[0]
    from nvs.dataset import EvalDataset
    legacy = EvalDataset(
        str(root / "test"), patch_size=32, input_views=2, supervise_views=3,
        test_index_fp=str(index.resolve()),
    )[0]
    assert torch.equal(contract["image"], legacy["image"])
    assert torch.equal(contract["K"], legacy["K"])
    direction = torch.tensor([1.0, 2.0, 3.0])
    direction /= torch.linalg.norm(direction)
    assert torch.allclose(contract["camtoworld"][1, :3, 3], direction)
    assert torch.allclose(
        legacy["camtoworld"][1, :3, 3],
        direction * torch.tensor([1.0, -1.0, -1.0]),
    )


def test_prope_compare_rejects_legacy_pose_divergence_and_routes(monkeypatch, tmp_path):
    root = tmp_path / "re10k"
    index = _fixture(root)
    artifact = tmp_path / "artifacts"
    _environment(monkeypatch, root, "compare", artifact)
    with pytest.raises(DatasetParityError):
        ContractEvalDataset(
            str(root / "test"), patch_size=32, input_views=2, supervise_views=3,
            test_index_fp=str(index.resolve()),
        )[0]
    assert not list(
        (artifact / "dataset-contract-receipts" / "prope-multiview-nvs-v1").glob(
            "*.json"
        )
    )
    from nvs.trainval import _re10k_dataset_types
    assert _re10k_dataset_types()[0].__module__ == "nvs.dataset_contract"
