"""PRoPE provider adapter for the shared ``rope-contract`` middleware.

The adapter uses PRoPE's documented precompute/apply lifecycle and leaves
scaled-dot-product attention to the consumer.  The original callable remains
available via ``legacy_native`` for an explicit rollback.  This baseline's
Torch implementation is self-attention only; cross-attention requests fail
closed because the upstream package has no cross geometry helper.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

try:
    from rope_contract import (
        CANONICAL_LAYOUT_ID,
        OP_TRANSFORM_INPUTS,
        CapabilityManifest,
        OpaqueContinuation,
        PrepareRequest,
        ProviderProfile,
        RopeProvider,
        RopeSession,
        SemanticChannel,
        ShapeMismatch,
        TensorDescriptor,
        TransformRequest,
        TransformResult,
        UnsupportedCapability,
        InvalidContinuation,
        ValidationError,
    )
except ImportError as exc:  # pragma: no cover - depends on workspace env
    raise ImportError(
        "PRoPE contract integration requires rope-contract on PYTHONPATH"
    ) from exc

from prope.torch import PropeDotProductAttention


PROVIDER_ID = "prope"
ADAPTER_ID = "prope-contract-provider"
ADAPTER_VERSION = "1"
TOKEN_ORDER = "camera_major_token_major"


def _tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ShapeMismatch(f"{name} must be a torch.Tensor", actual=type(value).__name__)
    if not value.is_floating_point():
        raise ValidationError(f"{name} must be floating point")
    if not torch.isfinite(value).all():
        raise ValidationError(f"{name} must contain finite values")
    return value


def _descriptor(value: torch.Tensor, *, role: str, token_order: str = TOKEN_ORDER) -> TensorDescriptor:
    return TensorDescriptor.from_tensor(
        value, layout_id=CANONICAL_LAYOUT_ID, role=role, token_order=token_order
    )


def _same(left: Any, right: Any) -> bool:
    return left is right or (
        isinstance(left, torch.Tensor)
        and isinstance(right, torch.Tensor)
        and left.shape == right.shape
        and left.dtype == right.dtype
        and left.device == right.device
        and torch.equal(left, right)
    )


def _select(payload: Mapping[str, Any], aliases: tuple[str, ...], *, name: str) -> Any:
    found = [(key, payload[key]) for key in aliases if key in payload]
    if not found:
        return None
    value = found[0][1]
    if any(not _same(value, candidate) for _, candidate in found[1:]):
        raise ValidationError(
            f"conflicting aliases were supplied for {name}",
            actual=[key for key, _ in found],
        )
    return value


@dataclass(frozen=True)
class ProPEGeometry:
    """Camera geometry used by the canonical PRoPE Torch implementation."""

    w2cs: torch.Tensor
    intrinsics: torch.Tensor
    batch_id: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        w2cs = _tensor(self.w2cs, "w2cs")
        intrinsics = _tensor(self.intrinsics, "intrinsics")
        if w2cs.ndim != 4 or tuple(w2cs.shape[-2:]) != (4, 4):
            raise ShapeMismatch("w2cs must have shape [B,C,4,4]", actual=tuple(w2cs.shape))
        if intrinsics.ndim != 4 or tuple(intrinsics.shape[-2:]) != (3, 3):
            raise ShapeMismatch(
                "intrinsics must have shape [B,C,3,3]", actual=tuple(intrinsics.shape)
            )
        if w2cs.shape[:2] != intrinsics.shape[:2]:
            raise ShapeMismatch("w2cs and intrinsics must share [B,C]")
        if w2cs.shape[0] <= 0 or w2cs.shape[1] <= 0:
            raise ShapeMismatch("w2cs must contain at least one batch and camera")
        if w2cs.device != intrinsics.device or w2cs.dtype != intrinsics.dtype:
            raise ValidationError("w2cs and intrinsics must share dtype and device")
        object.__setattr__(self, "w2cs", w2cs)
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def batch_size(self) -> int:
        return int(self.w2cs.shape[0])

    @property
    def camera_count(self) -> int:
        return int(self.w2cs.shape[1])

    def payload(self) -> dict[str, Any]:
        return {
            "w2cs": self.w2cs,
            "intrinsics": self.intrinsics,
            "geometry_metadata": dict(self.metadata),
        }


def _channels(geometry: ProPEGeometry) -> dict[str, SemanticChannel]:
    dtype = str(geometry.w2cs.dtype).removeprefix("torch.")
    return {
        "w2c": SemanticChannel(
            "w2c", shape=tuple(geometry.w2cs.shape), frame="world_to_camera",
            units="scene_units", ownership="camera", alignment="camera_major",
            dtype=dtype, source="geometry", physical_length=6,
        ),
        "intrinsics": SemanticChannel(
            "intrinsics", shape=tuple(geometry.intrinsics.shape), frame="camera",
            units="pixel", ownership="camera", alignment="camera_major",
            dtype=dtype, source="geometry", physical_length=4,
        ),
    }


def make_prepare_request(
    geometry: ProPEGeometry,
    *,
    patches_x: int,
    patches_y: int,
    profile_id: str = "prope_v1",
    provider_id: str = PROVIDER_ID,
    consumer_id: Optional[str] = None,
    consumer_version: Optional[str] = None,
) -> PrepareRequest:
    if isinstance(patches_x, bool) or not isinstance(patches_x, int) or patches_x <= 0:
        raise ShapeMismatch("patches_x must be a positive integer", actual=patches_x)
    if isinstance(patches_y, bool) or not isinstance(patches_y, int) or patches_y <= 0:
        raise ShapeMismatch("patches_y must be a positive integer", actual=patches_y)
    token_count = geometry.camera_count * patches_x * patches_y
    metadata = {**dict(geometry.metadata), "patches_x": patches_x, "patches_y": patches_y,
                "token_count": token_count}
    return PrepareRequest(
        profile_id=profile_id, provider_id=provider_id, consumer_id=consumer_id,
        consumer_version=consumer_version, attention_kind="multi_camera_dense_self",
        operation=OP_TRANSFORM_INPUTS, layout_id=CANONICAL_LAYOUT_ID,
        tensor_descriptors={
            "w2cs": _descriptor(geometry.w2cs, role="attention_message", token_order="camera_major"),
            "intrinsics": _descriptor(geometry.intrinsics, role="attention_message", token_order="camera_major"),
        }, semantic_channels=_channels(geometry), logical_shape=tuple(geometry.w2cs.shape),
        token_order=TOKEN_ORDER, batch_id=geometry.batch_id, payload=geometry.payload(),
        metadata=metadata,
    )


def make_transform_request(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    profile_id: str = "prope_v1",
    continuation: Optional[OpaqueContinuation] = None,
    dropout_p: float = 0.0,
    training: bool = False,
    batch_id: Optional[str] = None,
    session_generation: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> TransformRequest:
    q, k, v = (_tensor(query, "query"), _tensor(key, "key"), _tensor(value, "value"))
    return TransformRequest(
        profile_id=profile_id, attention_kind="multi_camera_dense_self",
        operation=OP_TRANSFORM_INPUTS, query=q, key=k, value=v,
        tensor_descriptors={
            "query": _descriptor(q, role="query"),
            "key": _descriptor(k, role="key"),
            "value": _descriptor(v, role="value"),
        }, continuation=continuation, dropout_p=dropout_p, training=training,
        batch_id=batch_id, session_generation=session_generation,
        metadata=dict(metadata or {}),
    )


def _profile(*, head_dim: int, config: Mapping[str, Any]) -> ProviderProfile:
    return ProviderProfile(
        profile_id="prope_v1", profile_version="1", feature_id="prope_v1",
        family="projective", strategy="deterministic", operations=("transform_inputs", "restore_output"),
        attention_kinds=("self", "multi_camera_dense_self"),
        transformed_roles=("query", "key", "value"), head_roles=("content",),
        required_semantic_channels=("w2c", "intrinsics"), head_dim_values=(head_dim,),
        dtype_values=("float16", "float32", "float64", "bfloat16"),
        device_values=("cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"),
        supports_gradients=True, supports_masks=False, supports_causal=False,
        supports_dropout=True, stateless=False, session_isolation=True, rng_isolation=True,
        metadata={
            "execution_mode": "transformed_operands",
            "attention_kernel_owner": "consumer",
            "native_callable": "prope.torch.PropeDotProductAttention",
            "geometry_convention": "world_to_camera+pixel_intrinsics",
            "token_order": TOKEN_ORDER,
            "config": dict(config),
        },
    )


class ProPESession(RopeSession):
    def __init__(self, *, provider: "ProPEProvider", profile: ProviderProfile,
                 prepare_request: PrepareRequest, geometry: ProPEGeometry) -> None:
        super().__init__(provider=provider, manifest=provider.manifest, profile=profile,
                         prepare_request=prepare_request)
        self.provider = provider
        self.geometry = geometry
        self._pending: dict[str, tuple[torch.Tensor, Any]] = {}

    def close(self) -> None:
        self._pending.clear()
        super().close()

    @staticmethod
    def _heads(value: torch.Tensor, nhead: int) -> torch.Tensor:
        if value.ndim != 4:
            raise ShapeMismatch("PRoPE expects Q/K/V [B,H,T,D]")
        return value

    def _transform_inputs_impl(self, request: TransformRequest) -> TransformResult:
        if request.attention_kind not in {"self", "multi_camera_dense_self"}:
            raise UnsupportedCapability("PRoPE baseline provider supports self attention only", actual=request.attention_kind)
        q, k, v = request.query, request.key, request.value
        assert isinstance(q, torch.Tensor) and isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor)
        if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
            raise ShapeMismatch("PRoPE expects Q/K/V [B,H,C*P,D] with equal shapes")
        if q.shape[0] != self.geometry.batch_size or q.device != self.geometry.w2cs.device:
            raise ShapeMismatch("Q/K/V batch or device differs from prepared geometry")
        expected_tokens = self.geometry.camera_count * int(self.prepare_request.metadata["patches_x"]) * int(self.prepare_request.metadata["patches_y"])
        if q.shape[2] != expected_tokens:
            raise ShapeMismatch("token count does not match camera grid", expected=expected_tokens, actual=q.shape[2])
        native = PropeDotProductAttention(
            head_dim=int(q.shape[-1]), patches_x=int(self.prepare_request.metadata["patches_x"]),
            patches_y=int(self.prepare_request.metadata["patches_y"]), image_width=self.provider.image_width,
            image_height=self.provider.image_height, freq_base=self.provider.freq_base,
            freq_scale=self.provider.freq_scale,
        ).to(device=q.device)
        native._precompute_and_cache_apply_fns(self.geometry.w2cs, self.geometry.intrinsics)
        q_out = native._apply_to_q(q)
        k_out = native._apply_to_kv(k)
        v_out = native._apply_to_kv(v)
        continuation = self._issue_continuation(operation=OP_TRANSFORM_INPUTS, request=request,
                                                output_layout_id=CANONICAL_LAYOUT_ID)
        self._pending[continuation.token] = (q_out, native)
        return TransformResult(
            query=q_out, key=k_out, value=v_out, continuation=continuation,
            tensor_descriptors={"query": _descriptor(q_out, role="query"), "key": _descriptor(k_out, role="key"),
                                "value": _descriptor(v_out, role="value")},
            metadata={"execution_mode": "transformed_operands", "consumer_attention": "scaled_dot_product_attention",
                      "value_policy": "native_apply_kv", "message_shape": list(q_out.shape)},
        )

    def _restore_output_impl(self, attention_message: Any, *, continuation: OpaqueContinuation, request: Any = None) -> TransformResult:
        del request
        try:
            q_out, native = self._pending[continuation.token]
        except KeyError as exc:
            raise InvalidContinuation("unknown or already restored PRoPE continuation") from exc
        message = _tensor(attention_message, "attention_message")
        if tuple(message.shape) != tuple(q_out.shape) or message.device != q_out.device or message.dtype != q_out.dtype:
            raise ShapeMismatch("attention message must match transformed query", expected=tuple(q_out.shape), actual=tuple(message.shape))
        restored = native._apply_to_o(message)
        del self._pending[continuation.token]
        return TransformResult(output=restored, tensor_descriptors={"output": _descriptor(restored, role="output")},
                               metadata={"execution_mode": "restored_output", "output_shape": list(restored.shape)})


class ProPEProvider(RopeProvider):
    """Provider façade around PRoPE's unchanged Torch implementation."""

    def __init__(self, *, patches_x: int, patches_y: int, image_width: int, image_height: int,
                 head_dim: int = 128, freq_base: float = 100.0, freq_scale: float = 1.0,
                 provider_id: str = PROVIDER_ID) -> None:
        if patches_x <= 0 or patches_y <= 0 or image_width <= 0 or image_height <= 0:
            raise ShapeMismatch("patch grid and image dimensions must be positive")
        if head_dim <= 0 or head_dim % 4:
            raise ShapeMismatch("PRoPE head_dim must be a positive multiple of four", actual=head_dim)
        self.patches_x, self.patches_y = int(patches_x), int(patches_y)
        self.image_width, self.image_height = int(image_width), int(image_height)
        self.head_dim = int(head_dim)
        self.freq_base, self.freq_scale = float(freq_base), float(freq_scale)
        config = {"head_dim": self.head_dim, "patches_x": self.patches_x, "patches_y": self.patches_y,
                  "image_width": self.image_width, "image_height": self.image_height,
                  "freq_base": self.freq_base, "freq_scale": self.freq_scale}
        manifest = CapabilityManifest(
            provider_id=provider_id, provider_version="prope-nvs",
            profiles=(_profile(head_dim=self.head_dim, config=config),), adapter_id=ADAPTER_ID,
            adapter_version=ADAPTER_VERSION,
            metadata={"native_entrypoint": "prope.torch.PropeDotProductAttention"},
        )
        super().__init__(manifest=manifest)

    def _geometry(self, request: PrepareRequest) -> ProPEGeometry:
        payload = request.payload
        w2cs = _select(payload, ("w2cs", "w2c"), name="w2cs")
        intrinsics = _select(payload, ("intrinsics", "Ks", "K"), name="intrinsics")
        if w2cs is None or intrinsics is None:
            raise ValidationError("PrepareRequest payload requires w2cs and intrinsics")
        w2cs, intrinsics = _tensor(w2cs, "w2cs"), _tensor(intrinsics, "intrinsics")
        for role, tensor in (("w2cs", w2cs), ("intrinsics", intrinsics)):
            descriptor = request.tensor_descriptors.get(role)
            if descriptor is not None:
                descriptor.validate(expected_shape=tuple(tensor.shape), expected_layout=CANONICAL_LAYOUT_ID,
                                    expected_dtype=str(tensor.dtype), expected_device=str(tensor.device),
                                    expected_role="attention_message", require_token_order=True)
        metadata = {**dict(payload.get("geometry_metadata", {})), **dict(request.metadata)}
        if int(metadata.get("patches_x", self.patches_x)) != self.patches_x or int(metadata.get("patches_y", self.patches_y)) != self.patches_y:
            raise ShapeMismatch("prepare patch grid differs from provider configuration")
        expected = int(w2cs.shape[1]) * self.patches_x * self.patches_y
        if int(metadata.get("token_count", expected)) != expected:
            raise ShapeMismatch("prepare token_count conflicts with native patch grid", expected=expected, actual=metadata.get("token_count"))
        metadata["token_count"] = expected
        return ProPEGeometry(w2cs=w2cs, intrinsics=intrinsics, batch_id=request.batch_id, metadata=metadata)

    def _create_session(self, request: PrepareRequest, profile: ProviderProfile) -> RopeSession:
        return ProPESession(provider=self, profile=profile, prepare_request=request, geometry=self._geometry(request))

    def open_session_for_geometry(self, geometry: ProPEGeometry, *, profile_id: str = "prope_v1",
                                  consumer_id: Optional[str] = None, consumer_version: Optional[str] = None) -> ProPESession:
        request = make_prepare_request(geometry, patches_x=self.patches_x, patches_y=self.patches_y,
                                       profile_id=profile_id, provider_id=self.provider_id,
                                       consumer_id=consumer_id, consumer_version=consumer_version)
        return self.open_session(request)  # type: ignore[return-value]

    def legacy_native(self, *, device: Optional[torch.device] = None) -> PropeDotProductAttention:
        """Construct the original PRoPE callable for explicit rollback."""
        native = PropeDotProductAttention(
            self.head_dim, patches_x=self.patches_x, patches_y=self.patches_y,
            image_width=self.image_width, image_height=self.image_height,
            freq_base=self.freq_base, freq_scale=self.freq_scale,
        )
        return native if device is None else native.to(device=device)


__all__ = [
    "ProPEGeometry", "ProPEProvider", "ProPESession",
    "make_prepare_request", "make_transform_request",
]
