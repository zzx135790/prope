"""Focused contract tests for the PRoPE provider boundary."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[5]
CONTRACT_SRC = ROOT / "worktrees" / "rope-contract" / "mainline" / "src"
if CONTRACT_SRC.is_dir():
    sys.path.insert(0, str(CONTRACT_SRC))

from prope.integrations.rope_contract_provider import (  # noqa: E402
    ProPEGeometry,
    ProPEProvider,
    make_transform_request,
)
from rope_contract import UnsupportedCapability  # noqa: E402


def _geometry() -> ProPEGeometry:
    w2c = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 2, 1, 1)
    w2c[:, 1, 0, 3] = 0.25
    intrinsics = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1)
    intrinsics[..., 0, 0] = intrinsics[..., 1, 1] = 2.0
    return ProPEGeometry(w2c, intrinsics)


def _qkv() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(6)
    return tuple(torch.randn(1, 1, 8, 128) for _ in range(3))  # type: ignore[return-value]


def test_manifest_and_legacy_rollback_are_explicit() -> None:
    provider = ProPEProvider(patches_x=2, patches_y=2, image_width=2, image_height=2)
    profile = provider.describe().profile("prope_v1")
    assert profile.metadata["attention_kernel_owner"] == "consumer"
    assert provider.legacy_native().head_dim == 128


def test_transform_does_not_execute_sdpa_and_restores_output(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = ProPEProvider(patches_x=2, patches_y=2, image_width=2, image_height=2)
    session = provider.open_session_for_geometry(_geometry())
    q, k, v = _qkv()

    def fail(*args, **kwargs):  # pragma: no cover - called only on regression
        raise AssertionError("PRoPE provider must not execute consumer SDPA")

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fail)
    prepared = session.transform_inputs(make_transform_request(q, k, v))
    assert prepared.output is None
    assert prepared.query.shape == prepared.key.shape == prepared.value.shape == q.shape

    monkeypatch.undo()
    message = torch.nn.functional.scaled_dot_product_attention(prepared.query, prepared.key, prepared.value)
    restored = session.restore_output(message, continuation=prepared.continuation)
    assert restored.output is not None and restored.output.shape == q.shape
    assert torch.isfinite(restored.output).all()


def test_provider_matches_legacy_callable_attention() -> None:
    provider = ProPEProvider(patches_x=2, patches_y=2, image_width=2, image_height=2)
    geometry = _geometry()
    q, k, v = _qkv()
    expected = provider.legacy_native()(q, k, v, geometry.w2cs, geometry.intrinsics)
    session = provider.open_session_for_geometry(geometry)
    prepared = session.transform_inputs(make_transform_request(q, k, v))
    message = torch.nn.functional.scaled_dot_product_attention(prepared.query, prepared.key, prepared.value)
    restored = session.restore_output(message, continuation=prepared.continuation).output
    torch.testing.assert_close(restored, expected, rtol=2e-4, atol=2e-4)


def test_cross_attention_fails_closed_and_native_path_remains_available() -> None:
    provider = ProPEProvider(patches_x=2, patches_y=2, image_width=2, image_height=2)
    session = provider.open_session_for_geometry(_geometry())
    q, k, v = _qkv()
    request = make_transform_request(q, k, v)
    request = type(request)(**{**request.__dict__, "attention_kind": "cross"})
    with pytest.raises(UnsupportedCapability):
        session.transform_inputs(request)
    native = provider.legacy_native()
    native._precompute_and_cache_apply_fns(session.geometry.w2cs, session.geometry.intrinsics)
    assert callable(native._apply_to_q)
