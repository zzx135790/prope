"""PRoPE-owned RE10K adapter for the shared dataset contract."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from rope_contract.dataset.migration import (
    DatasetMigrationDispatcher,
    DatasetRuntimeBinding,
    FrozenSamplePlan,
    source_snapshot_digest,
)

from nvs.dataset import (
    EvalDataset,
    TrainDataset,
    _normalize_poses_identity_unit_distance,
    load_and_maybe_update_meta_info,
    load_frames_from_meta_info,
)


CONSUMER_ADAPTER_ID = "prope-multiview-nvs-v1"
_OPENCV_TO_LEGACY_BLENDER = np.diag([1.0, -1.0, -1.0, 1.0])


def _consumer_source_digest() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = ("nvs/dataset.py", "nvs/dataset_contract.py", "nvs/trainval.py")
    return source_snapshot_digest({path: root / path for path in paths})


def _binding() -> DatasetRuntimeBinding:
    binding = DatasetRuntimeBinding.from_environment(
        expected_consumer_adapter_id=CONSUMER_ADAPTER_ID
    )
    if binding.dataset_id != "re10k" or binding.provider_id != "re10k-transforms-v1":
        raise ValueError("PRoPE contract consumer requires the declared canonical re10k binding")
    return binding


def _scene_metadata(scene) -> Dict[str, Any]:
    first = scene.frames[0].camera
    width, height = first.image_size
    fx, _, cx, _, fy, cy, _, _, _ = first.intrinsics
    frames = []
    for frame in scene.frames:
        if frame.image is None:
            raise ValueError(f"contract scene frame has no image: {frame.frame_id}")
        matrix = np.asarray(
            frame.camera.camera_to_world, dtype=np.float64
        ).reshape(4, 4)
        legacy_matrix = matrix @ _OPENCV_TO_LEGACY_BLENDER
        frames.append({
            "file_path": frame.image.uri,
            "transform_matrix": legacy_matrix.tolist(),
        })
    return {
        "w": width, "h": height, "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy,
        "frames": frames,
    }


def _sample_id(scene_id: str, frame_ids) -> str:
    text = f"{scene_id}:{','.join(str(int(value)) for value in frame_ids)}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _ContractMixin:
    def _load_selected(self, scene_id: str, frame_ids, legacy_root: Path) -> Dict[str, Any]:
        scene = self.provider.load_scene(scene_id, self.binding.profile_id, split=self.split)
        random_state = np.random.get_state()
        contract = load_frames_from_meta_info(
            str(self.provider.root),
            _scene_metadata(scene),
            frame_ids,
            patch_size=self.patch_size,
            zoom_factor=self.zoom_factor,
            random_zoom=getattr(self, "random_zoom", False),
        )
        if self.binding.mode == "contract":
            return contract
        legacy_scene = legacy_root / scene_id
        valid, metadata = load_and_maybe_update_meta_info(str(legacy_scene / "transforms.json"))
        if not valid:
            raise ValueError(f"invalid legacy scene for compare mode: {legacy_scene}")
        np.random.set_state(random_state)
        legacy = load_frames_from_meta_info(
            str(legacy_scene), metadata, frame_ids,
            patch_size=self.patch_size,
            zoom_factor=self.zoom_factor,
            random_zoom=getattr(self, "random_zoom", False),
        )
        artifact_root = os.environ.get("WORKSPACE_ARTIFACT_DIR")
        if not artifact_root or not os.path.isabs(artifact_root):
            raise ValueError("compare mode requires an absolute WORKSPACE_ARTIFACT_DIR")
        sample_id = _sample_id(scene_id, frame_ids)
        project = lambda value: {key: value[key] for key in ("image", "K", "camtoworld")}
        DatasetMigrationDispatcher(
            mode="compare",
            plan=FrozenSamplePlan((sample_id,)),
            legacy_loader=lambda _: project(legacy),
            contract_loader=lambda _: project(contract),
            dataset_id=self.binding.dataset_id,
            provider_id=self.binding.provider_id,
            profile_id=self.binding.profile_id,
            consumer_adapter_id=self.binding.consumer_adapter_id,
            receipt_dir=Path(artifact_root) / "dataset-contract-receipts" / CONSUMER_ADAPTER_ID,
            absolute_tolerance=1e-5,
            relative_tolerance=1e-6,
            consumer_source_digest=_consumer_source_digest(),
        ).load(sample_id)
        return legacy


class ContractTrainDataset(_ContractMixin, TrainDataset):
    def __init__(self, data_dirs: List[str], **kwargs) -> None:
        self.binding = _binding()
        if self.binding.mode not in {"contract", "compare"}:
            raise ValueError("contract adapter requires contract or compare mode")
        self.provider = self.binding.create_provider()
        self.split = "train"
        super().__init__([Path(value).name for value in data_dirs], **kwargs)

    def __getitem__(self, _: Any) -> Dict[str, Any]:
        scene_id = str(np.random.choice(self.data_dirs), encoding="utf-8")
        scene = self.provider.load_scene(scene_id, self.binding.profile_id, split="train")
        frame_ids = self._select_views(len(scene.frames))
        if frame_ids is None:
            return self.__getitem__(None)
        frame_ids = sorted(frame_ids)
        frame_ids = [frame_ids[0], frame_ids[-1]] + frame_ids[1:-1]
        root = Path(os.environ.get("RE10K_TRAIN_DIR", str(self.provider.root / "train")))
        loaded = self._load_selected(scene_id, frame_ids, root)
        camtoworld = _normalize_poses_identity_unit_distance(
            torch.from_numpy(loaded["camtoworld"]).float(), 0, self.input_views - 1
        )
        return {
            "camtoworld": camtoworld,
            "K": torch.from_numpy(loaded["K"]).float(),
            "image": torch.from_numpy(loaded["image"]).float(),
            "image_path": loaded["image_path"],
        }


class ContractEvalDataset(_ContractMixin, EvalDataset):
    def __init__(
        self,
        folder: str,
        patch_size: int = 256,
        zoom_factor: float = 1.0,
        verbose: bool = False,
        first_n: Optional[int] = None,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        input_views: int = 2,
        supervise_views: int = 3,
        render_video: bool = False,
        test_index_fp: Optional[str] = None,
    ) -> None:
        self.binding = _binding()
        if self.binding.mode not in {"contract", "compare"}:
            raise ValueError("contract adapter requires contract or compare mode")
        self.provider = self.binding.create_provider()
        self.split = "test"
        self.patch_size = patch_size
        self.zoom_factor = zoom_factor
        self.input_views = input_views
        self.supervise_views = supervise_views
        self.render_video = render_video
        if test_index_fp is None:
            test_index_fp = "evaluation_index_re10k_video.json" if render_video else "evaluation_index_re10k.json"
        index_path = Path(test_index_fp)
        if not index_path.is_absolute():
            index_path = Path(__file__).resolve().parent.parent / "assets" / test_index_fp
        index = json.loads(index_path.read_text(encoding="utf-8"))
        available = set(self.provider.list_scene_ids(split="test"))
        scenes = sorted(key for key, value in index.items() if value is not None and key in available)
        if first_n is not None:
            scenes = scenes[:first_n]
        if rank is not None and world_size is not None:
            scenes = scenes[rank::world_size]
        if verbose:
            print(f"[PRoPE Contract EvalDataset] Using {len(scenes)} scenes.")
        self.scene_ids = np.array(scenes).astype(np.bytes_)
        self.data_dirs = self.scene_ids
        self.contexts = np.array([index[scene]["context"] for scene in scenes])
        self.targets = [index[scene]["target"] for scene in scenes] if render_video else np.array([index[scene]["target"] for scene in scenes])
        self.legacy_root = Path(folder)

    def __getitem__(self, scene_index: int) -> Dict[str, Any]:
        scene_id = str(self.scene_ids[scene_index], encoding="utf-8")
        context = self.contexts[scene_index][: self.input_views]
        target = self.targets[scene_index] if self.render_video else self.targets[scene_index][: self.supervise_views]
        frame_ids = np.concatenate([context, target])
        loaded = self._load_selected(scene_id, frame_ids, self.legacy_root)
        camtoworld = _normalize_poses_identity_unit_distance(
            torch.from_numpy(loaded["camtoworld"]).float(), 0, self.input_views - 1
        )
        return {
            "camtoworld": camtoworld,
            "K": torch.from_numpy(loaded["K"]).float(),
            "image": torch.from_numpy(loaded["image"]).float(),
            "image_path": loaded["image_path"],
            "scene": scene_index,
        }


__all__ = ["ContractTrainDataset", "ContractEvalDataset"]
