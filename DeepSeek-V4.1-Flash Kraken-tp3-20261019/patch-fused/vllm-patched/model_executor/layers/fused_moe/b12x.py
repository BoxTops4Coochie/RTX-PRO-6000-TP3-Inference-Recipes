# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""b12x modular tensor-parallel fused MoE backend."""

import functools
import inspect
import os
import weakref
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config import get_current_vllm_config_or_none
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
    kMxfp8Dynamic,
    kMxfp8Static,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.b12x import (
    B12xPreparationUnit,
    B12xWorkload,
    PreparationResourceUnavailableError,
    get_b12x_a16_max_tokens,
    get_b12x_fused_moe,
    is_b12x_a4_prefill_enabled,
    reuse_packed_weight_storage,
    set_b12x_preparation_provider,
)
from vllm.model_executor.layers.fused_moe.expert_residency import (
    BANK_BUDGET_ENV,
    BANK_ENABLE_ENV,
    BANK_HISTOGRAM_ENV,
    BANK_SLOTS,
    BankModeRejected,
    DeferredWeightBanks,
    ExpertRouteMonitor,
    _slot_stride_bytes,
    bank_receipt,
    build_expert_banks,
    parse_histogram_env,
    select_hot_experts,
)


logger = init_logger(__name__)

_B12X_MOE_MODES: dict[
    tuple[torch.dtype | str | None, torch.dtype | str | None],
    tuple[str, str, str],
] = {
    ("mxfp4", "mxfp8"): ("w4a8_mx", "fp4_e8m0_k32", "w31"),
    ("mxfp8", "mxfp8"): ("w8a8_mx", "mxfp8_e8m0_k32", "w31"),
    ("mxfp4", None): ("w4a16", "fp4_e8m0_k32", "w31"),
    ("exl3", None): ("w4a16", "exl3", "w31"),
    ("nvfp4", "nvfp4"): ("nvfp4", "modelopt_nvfp4", "w31"),
    ("nvfp4", "mxfp8"): ("w4a8_nvfp4", "modelopt_nvfp4", "w31"),
    ("nvfp4", None): ("w4a16", "modelopt_nvfp4", "w31"),
    ("iq2_xs", None): ("w4a16", "iq2_xs", "w31"),
    ("iq2_xxs", None): ("w4a16", "iq2_xxs", "w31"),
    ("q8_0", None): ("w4a16", "q8_0", "w31"),
}


def _install_codec_aware_tuning():
    """When B12X_SCALE_CODEC=1, pin every component's tuning to its heuristic
    default.  After weights, compact scale planes, and full expert promotion
    land, only ~1.5 GiB of headroom remains per rank, which is not enough to
    hold even a single measured trial's scratch for the attention components.
    The pin goes through configure()'s override slot, so each obligation
    settles through the same pinned path the heuristic mode uses and never
    enumerates candidates."""
    if (
        os.environ.get("B12X_SCALE_CODEC", "0") != "1"
        and os.environ.get("DS41_B12X_CSF", "0") != "1"
    ):
        return
    # Online CSF (DS41_B12X_CSF=1) needs the pin too: each TP rank compresses
    # different expert scales, so measured MoE tuning keys differ per rank and
    # the startup coordinator rejects them as incompatible boundaries.
    # B12X_CODEC_PIN_TUNING=0 keeps measured tuning with the codec; the pin
    # exists only for placements without startup headroom for tuning trials.
    if os.environ.get("B12X_CODEC_PIN_TUNING", "1") == "0":
        return
    from b12x.preparation import PreparationSession, TuningContract

    _stock_configure = TuningContract.configure
    # Comma-separated component-ID prefixes to pin, or "all". MoE must stay
    # pinned: its measured candidates include layouts the compact scales reject.
    pinned = os.environ.get("B12X_CODEC_PIN_COMPONENTS", "all")
    prefixes = None if pinned == "all" else tuple(p for p in pinned.split(",") if p)

    def _codec_configure(self, query, *, device, override=None, search=True):
        if override is None and (
            prefixes is None or self.component_id.startswith(prefixes)
        ):
            override = self.default_config(query, device)
        return _stock_configure(
            self, query, device=device, override=override, search=search
        )

    TuningContract.configure = _codec_configure

    # Narrow the measured race: every resident trial carries its own scratch,
    # and after the weights, compact scale planes, and full expert promotion
    # land together, roughly 1.6 GiB of headroom remains per rank. Keep only
    # a champion plus one challenger resident and cap concurrent scratch so
    # the race fits that headroom.
    _stock_init = PreparationSession.__init__

    def _narrow_init(self, *args, **kwargs):
        kwargs.setdefault("race_batch", 2)
        kwargs.setdefault("race_budget", 128 * 1024 * 1024)
        _stock_init(self, *args, **kwargs)

    PreparationSession.__init__ = _narrow_init


_install_codec_aware_tuning()

def promote_pending_inplace() -> None:
    """Run in-place promotion now for every armed, still-pending VMM layer."""
    for backend in list(_PENDING_INPLACE_LAYERS):
        backend._promote_inplace()


# Deferred banks not yet promoted. With B12X_DEFERRED_BANK_EVEN_SPLIT=1 each
# promotion takes an equal share of the remaining budget instead of the first
# layer filling up and starving the later ones; with route-ordered promotion
# that keeps each layer's most-routed experts resident.
_PENDING_DEFERRED_BANKS: "weakref.WeakSet[DeferredWeightBanks]" = weakref.WeakSet()
# Bank-free in-place mode (B12X_DEFERRED_INPLACE=1, B12X_DEFERRED_WEIGHT_BANKS=0):
# VMM-backed expert layers awaiting promotion of their canonical regions.
_PENDING_INPLACE_LAYERS: "weakref.WeakSet[Any]" = weakref.WeakSet()


def _require_b12x_fused_moe() -> Any:
    fused_moe = get_b12x_fused_moe()
    assert fused_moe is not None
    return fused_moe


def _b12x_activation_name(activation: MoEActivation) -> str:
    if activation == MoEActivation.SILU:
        return "silu"
    if activation in (MoEActivation.RELU2, MoEActivation.RELU2_NO_MUL):
        return "relu2"
    return activation.value


@dataclass(frozen=True)
class _PreparedMoECall:
    """Priming tensors for one exact prepared MoE variant.

    These are trial-only bindings over the real loaded expert representation;
    serving always binds the layer-held prepared plan to caller tensors.
    """

    state: Any
    tokens: int
    topk: int
    prepared: Any
    output_dtype: torch.dtype

    def make(self, tensors):
        from b12x.preparation import PreparedCall

        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
            for spec in self.state.scratch.scratch_specs()
        )
        (
            hidden,
            activation_source,
            output,
            route_ids,
            route_weights,
            ids,
            weights,
        ) = tensors

        def reset() -> None:
            output.zero_()
            for buffer in scratch:
                buffer.zero_()

        # Keep route_ids alive through the producer for weakref-based trial
        # reuse, without exporting trial buffers into the serving plan.

        def produce(pattern: int = 0) -> None:
            hidden.copy_(activation_source)
            ids.copy_(route_ids[pattern])
            weights.copy_(route_weights)

        def restore() -> None:
            reset()
            produce()

        binding = self.state.bind(
            scratch=scratch,
            a=hidden,
            experts=self.prepared,
            topk_weights=weights,
            topk_ids=ids,
            output=output,
            input_scales_static=True,
        )
        return PreparedCall(
            run=binding.run,
            output=output,
            produce=produce,
            reset=reset,
            restore=restore,
            capture_safe=False,
            benchmark_producers=tuple(
                partial(produce, pattern) for pattern in range(route_ids.shape[0])
            ),
        )


def _prepared_moe_call_factory(
    *, tokens: int, topk: int, prepared: Any, output_dtype: torch.dtype
):
    shared = None

    def factory(state: Any):
        from b12x.moe.fused_moe.workloads import make_tuning_routes

        nonlocal shared
        tensors = None if shared is None else tuple(ref() for ref in shared)
        if tensors is None or any(tensor is None for tensor in tensors):
            device = prepared.device
            hidden = torch.empty(
                (tokens, int(prepared.hidden_size)),
                dtype=prepared.plan.activation.io_dtype,
                device=device,
            )
            generator = torch.Generator(device=device).manual_seed(42)
            activation_source = torch.empty_like(hidden).normal_(
                mean=0.0, std=0.125, generator=generator
            )
            output = torch.empty(hidden.shape, dtype=output_dtype, device=device)
            route_rows = torch.arange(
                tokens, dtype=torch.int32, device=device
            ).unsqueeze(1)
            route_columns = torch.arange(
                topk, dtype=torch.int32, device=device
            ).unsqueeze(0)
            route_ids = make_tuning_routes(
                tokens, topk, int(prepared.num_experts), device=device
            )
            route_logits = (
                route_rows.to(dtype=torch.float32) * 0.03125
                + route_columns.to(dtype=torch.float32) * 0.125
            )
            route_weights = torch.softmax(route_logits, dim=-1).contiguous()
            ids = torch.empty_like(route_ids[0])
            weights = torch.empty_like(route_weights)
            tensors = (
                hidden,
                activation_source,
                output,
                route_ids,
                route_weights,
                ids,
                weights,
            )
            shared = tuple(weakref.ref(tensor) for tensor in tensors)
        return _PreparedMoECall(
            state=state,
            tokens=tokens,
            topk=topk,
            prepared=prepared,
            output_dtype=output_dtype,
        ).make(tensors)

    return factory


def _is_current_stream_capturing() -> bool:
    is_capturing = getattr(torch.cuda, "is_current_stream_capturing", None)
    return bool(is_capturing is not None and is_capturing())


def _normalize_topk_weights(topk_weights: torch.Tensor) -> torch.Tensor:
    if topk_weights.dtype == torch.float32 and topk_weights.is_contiguous():
        return topk_weights
    if _is_current_stream_capturing():
        raise RuntimeError(
            "b12x MoE topk_weights normalization would allocate during CUDA capture"
        )
    return topk_weights.to(dtype=torch.float32).contiguous()


def _a4_prefill_parts(tokens: int) -> tuple[tuple[int, int, bool], ...]:
    if _is_current_stream_capturing() or not is_forward_context_available():
        return ((0, tokens, False),)
    metadata = get_forward_context().moe_prefill_metadata
    if metadata is None:
        return ((0, tokens, False),)
    if metadata.num_tokens != tokens:
        raise ValueError(
            "B12X hybrid prefill requires the target forward's padded token ordering: "
            f"expected {metadata.num_tokens} rows, got {tokens}"
        )
    parts = []
    previous = 0
    for start, end in metadata.prefill_ranges:
        if not previous <= start < end <= tokens:
            raise ValueError("B12X hybrid prefill ranges must be ordered and disjoint")
        if previous < start:
            parts.append((previous, start, False))
        parts.append((start, end, True))
        previous = end
    if previous < tokens:
        parts.append((previous, tokens, False))
    return tuple(parts)


def _replace_parameter_with_empty(
    layer: torch.nn.Module,
    name: str,
) -> torch.Tensor | None:
    parameter = getattr(layer, name, None)
    if not isinstance(parameter, torch.Tensor):
        return None
    empty = torch.empty((0,), dtype=parameter.dtype, device=parameter.device)
    replace_parameter(layer, name, empty)
    return getattr(layer, name)


def _normalize_expert_scale(scale: torch.Tensor) -> torch.Tensor:
    if scale.ndim == 2:
        if scale.shape[1] not in (1, 2):
            raise ValueError(
                "expected an expert scale with one or two columns, got "
                f"{tuple(scale.shape)}"
            )
        scale = scale[:, 0]
    return scale.to(dtype=torch.float32).contiguous()


@functools.cache
def _b12x_has_mxfp8_moe() -> bool:
    """Whether the installed b12x prepares MXFP8 (w8a8_mx) MoE experts."""
    try:
        from b12x.moe.fused_moe import PackedSourceFormat
    except ImportError:
        return False
    return "mxfp8_e8m0_k32" in {fmt.value for fmt in PackedSourceFormat}



def _online_csf_enabled(source_format: str, quant_mode: str) -> bool:
    """DS41_B12X_CSF=1 compresses expert block scales while preparing.

    b12x plans online CSF for native MXFP4 (fp4_e8m0_k32, A8/A16) and — PATCH(ds41-nvfp4) —
    for ModelOpt NVFP4 sources (modelopt_nvfp4, A4/A16): see flashinfer b12x planning.py
    constraints. Without it the NVFP4 E4M3 per-16 scales stay on-device uncompressed
    (~9 GB/GPU on DS41), which halves expert promotion depth.
    """
    if os.environ.get("DS41_B12X_CSF", "0") != "1":
        return False
    if source_format == "fp4_e8m0_k32":
        return quant_mode in ("w4a8_mx", "w4a16")
    if source_format == "modelopt_nvfp4" and os.environ.get("DS41_NVFP4_CSF", "1") == "1":
        return quant_mode in ("nvfp4", "w4a16")
    return False


_CSF_SCALE_SCRATCH: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}


def _shared_csf_scale_scratch(
    w1_scale: torch.Tensor, w2_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return this device's CSF expansion buffers, sized by the first layer.

    The first prepared layer is a full target layer, the largest MoE on the
    device; a later, larger layer would need buffers its predecessors do not
    reference, so it is rejected instead of silently reallocating.
    """
    device = w1_scale.device
    need = (w1_scale.numel() * w1_scale.element_size(),
            w2_scale.numel() * w2_scale.element_size())
    scratch = _CSF_SCALE_SCRATCH.get(device)
    if scratch is None:
        scratch = tuple(
            torch.empty(n, dtype=torch.uint8, device=device) for n in need
        )
        _CSF_SCALE_SCRATCH[device] = scratch
        logger.info("b12x online CSF scale scratch: %d + %d bytes on %s", *need, device)
    if any(n > s.numel() for n, s in zip(need, scratch)):
        raise ValueError("b12x online CSF: a later MoE layer outgrows the shared scratch")
    return scratch

class B12xExperts(mk.FusedMoEExpertsModular):
    """Packed MoE experts backed by the b12x SM12x planned API."""

    # Set by the NVFP4-CSF method while the shared scale scratch holds this
    # layer's scales, expanded ahead of the call (b12x expand_scales()).
    scales_expanded = False

    def __init__(
        self,
        moe_config: mk.FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        """Bind the b12x quant mode, packed source format and W13 layout.

        Raises ValueError when no b12x kernel covers the configured weight
        dtype or the (weight, activation) scheme; no weights are prepared.
        """
        super().__init__(moe_config, quant_config)
        if quant_config.weight_quant_dtype not in (
            "mxfp4",
            "mxfp8",
            "nvfp4",
            "exl3",
            "iq2_xs",
            "iq2_xxs",
            "q8_0",
        ):
            raise ValueError(
                "b12x MoE requires MXFP4, MXFP8, NVFP4, EXL3, IQ2_XS, IQ2_XXS "
                "or Q8_0 weights, got "
                f"{quant_config.weight_quant_dtype}"
            )
        scheme = (
            quant_config.weight_quant_dtype,
            quant_config.quant_dtype,
        )
        try:
            self._quant_mode, self._source_format, self._w13_layout = _B12X_MOE_MODES[
                scheme
            ]
        except KeyError as exc:
            raise ValueError(
                f"unsupported b12x MoE quantization scheme {scheme}"
            ) from exc
        self._prepared_experts: Any | None = None
        self._a4_prefill_enabled = (
            self._quant_mode == "w4a16"
            and self._source_format == "modelopt_nvfp4"
            and is_b12x_a4_prefill_enabled()
        )
        self._a16_max_tokens = (
            get_b12x_a16_max_tokens()
            if self._source_format == "modelopt_nvfp4" and not self._a4_prefill_enabled
            else 0
        )
        if self._a4_prefill_enabled:
            parallel = moe_config.moe_parallel_config
            config = get_current_vllm_config_or_none()
            if (
                parallel.dp_size != 1
                or parallel.pcp_size != 1
                or parallel.is_sequence_parallel
                or parallel.use_ep
                or parallel.ep_size != 1
                or (config is not None and config.parallel_config.use_ubatching)
            ):
                raise NotImplementedError(
                    "B12X hybrid A4 prefill supports TP/DCP without DP, PCP, "
                    "EP, sequence parallelism or ubatching"
                )
            if (
                config is not None
                and config.speculative_config is not None
                and config.speculative_config.enable_adaptive_verification
            ):
                raise NotImplementedError(
                    "B12X hybrid A4 prefill requires host-visible token ranges; "
                    "adaptive verification changes token ranges on the GPU"
                )
            fused_moe = _require_b12x_fused_moe()
            binding_type = getattr(fused_moe, "Binding", None)
            scale_query = getattr(fused_moe, "uses_expanded_nvfp4_scales", None)
            if (
                not callable(scale_query)
                or "scales_expanded" not in inspect.signature(scale_query).parameters
                or "a4_prefill_launches"
                not in getattr(binding_type, "__dataclass_fields__", {})
            ):
                raise RuntimeError(
                    "B12X hybrid A4 prefill requires a compatible FlashInfer build "
                    "with semantic A4 selection and the "
                    "uses_expanded_nvfp4_scales(scales_expanded=...) API"
                )
        self._source_parameters_released = False
        self._unit_scales: dict[torch.device, torch.Tensor] = {}
        self._apply_router_weight_on_input = False
        self._plan: Any | None = None
        self._plan_key: tuple | None = None
        self._plan_activation: MoEActivation | None = None
        self._plan_route_on_input: bool | None = None
        self._deferred_bank_lock = threading.Lock()
        self._deferred_bank_promoted = False
        self._route_monitor: ExpertRouteMonitor | None = None

    def _unit_scale(self, device: torch.device, num_experts: int) -> torch.Tensor:
        scale = self._unit_scales.get(device)
        if scale is None or scale.numel() != num_experts:
            scale = torch.ones(num_experts, dtype=torch.float32, device=device)
            self._unit_scales[device] = scale
        return scale

    def _weight_global_scale(
        self,
        device: torch.device,
        num_experts: int,
        scale: torch.Tensor | None,
        name: str,
    ) -> torch.Tensor:
        if self._source_format != "modelopt_nvfp4":
            return self._unit_scale(device, num_experts)
        if scale is None:
            raise ValueError(f"b12x NVFP4 MoE requires {name}")
        scale = _normalize_expert_scale(scale)
        if scale.numel() != num_experts:
            raise ValueError(
                f"b12x NVFP4 MoE expected {num_experts} {name} values, "
                f"got {scale.numel()}"
            )
        return scale.to(device=device)

    def _swiglu_params(
        self,
        activation: MoEActivation,
    ) -> tuple[float | None, float | None, float | None]:
        if activation in (
            MoEActivation.SITU,
            MoEActivation.RELU2,
            MoEActivation.RELU2_NO_MUL,
        ):
            return None, None, None

        limit = self.quant_config.gemm1_clamp_limit
        if limit is None:
            limit = self.moe_config.swiglu_limit
        if activation != MoEActivation.SWIGLUOAI_UNINTERLEAVE:
            return limit, None, None

        alpha = self.quant_config.gemm1_alpha
        if alpha is None:
            alpha = self.moe_config.swiglu_alpha
        beta = self.quant_config.gemm1_beta
        if beta is None:
            beta = self.moe_config.swiglu_beta
        return limit, alpha, beta

    def _prepare_experts(
        self,
        *,
        w1: torch.Tensor,
        w2: torch.Tensor,
        activation: MoEActivation,
        params_dtype: torch.dtype,
    ) -> Any:
        """Build the b12x prepared-experts representation for these weights.

        Block-quantized sources are planned straight from the packed extents
        with a ``BlockQuantWeights`` binding; every other mode requires the
        w1/w2 block scales, recovers the logical intermediate size from the
        packed extent (FP4 holds two values per byte) and binds the weight and
        activation global scales. Raises RuntimeError under CUDA graph capture.
        """
        quant_mode = self._quant_mode
        if _is_current_stream_capturing():
            raise RuntimeError(
                "b12x MoE weights must be prepared before CUDA graph capture"
            )
        if self._source_format in ("iq2_xs", "iq2_xxs", "q8_0"):
            fused_moe = _require_b12x_fused_moe()
            weight_plan = fused_moe.plan_weights(
                source=fused_moe.PackedSource(
                    format=self._source_format, w13_layout="w31"
                ),
                activation=fused_moe.ActivationSpec(
                    mode="a16",
                    nonlinearity=_b12x_activation_name(activation),
                    io_dtype=params_dtype,
                ),
                geometry=fused_moe.MoEGeometry(
                    num_experts=int(w1.shape[0]),
                    hidden_size=int(w2.shape[1]),
                    intermediate_size=int(w2.shape[2])
                    * (32 if self._source_format == "q8_0" else 256),
                ),
            )
            return fused_moe.prepare_weights(
                plan=weight_plan,
                weights=fused_moe.BlockQuantWeights(
                    w13=w1, w2=w2, codec=self._source_format
                ),
            )
        if self.w1_scale is None or self.w2_scale is None:
            raise ValueError("b12x MoE requires w1 and w2 block scales")

        fused_moe = _require_b12x_fused_moe()

        num_experts = int(w1.shape[0])
        hidden_size = int(w2.shape[1])
        # FP4 weights pack two values per byte, so the loaded w2 last dim is
        # half the logical channel count; MXFP8 stores one byte per value and
        # the loaded extent is already logical.
        w1_scale, w2_scale = self.w1_scale, self.w2_scale
        if quant_mode in ("w8a8_mx",):
            intermediate_size = int(w2.shape[2])
        else:
            intermediate_size = int(w2.shape[2]) * 2
        unit_scale = self._unit_scale(w1.device, num_experts)
        w1_global_scale = self._weight_global_scale(
            w1.device, num_experts, self.g1_alphas, "w1 global scales"
        )
        w2_global_scale = self._weight_global_scale(
            w2.device, num_experts, self.g2_alphas, "w2 global scales"
        )

        if quant_mode in ("nvfp4", "w4a8_nvfp4"):
            if self.a1_gscale is None or self.a2_gscale is None:
                raise ValueError("b12x NVFP4 MoE requires activation global scales")
            a1_gscale = _normalize_expert_scale(self.a1_gscale).to(w1.device)
            a2_gscale = _normalize_expert_scale(self.a2_gscale).to(w2.device)
        elif self._a4_prefill_enabled:
            a1_gscale = a2_gscale = None
            if self.a1_gscale is not None and self.a2_gscale is not None:
                a1 = _normalize_expert_scale(self.a1_gscale).to(w1.device)
                a2 = _normalize_expert_scale(self.a2_gscale).to(w2.device)
                if all(bool((torch.isfinite(t) & (t > 0)).all()) for t in (a1, a2)):
                    a1_gscale = a1.amin().reshape(1)
                    a2_gscale = a2
        else:
            a1_gscale = unit_scale
            a2_gscale = unit_scale

        limit, alpha, beta = self._swiglu_params(activation)
        mode = {
            "w4a16": fused_moe.ActivationMode.A16,
            "w4a8_mx": fused_moe.ActivationMode.A8,
            "w8a8_mx": fused_moe.ActivationMode.A8,
            "w4a8_nvfp4": fused_moe.ActivationMode.A8,
            "nvfp4": fused_moe.ActivationMode.A4,
        }[quant_mode]
        online_csf = _online_csf_enabled(self._source_format, quant_mode)
        # BRRR TP3 stage-3 A/B: DS41_B12X_CSF_LAYERS="17-39" keeps CSF only for those layers; the fused decoder
        # layer reads native scales, so the layers it serves run uncompressed.
        _csf_layers = os.environ.get("DS41_B12X_CSF_LAYERS", "")
        if online_csf and _csf_layers:
            import re as _re
            _m = _re.search(r"layers\.(\d+)\.", getattr(self, "_layer_name", ""))
            _lo, _hi = (int(v) for v in _csf_layers.split("-"))
            if _m is None or not (_lo <= int(_m.group(1)) <= _hi):
                online_csf = False
        if os.environ.get("DS41_FUSED_N64_FOR_NATIVE", "0") == "1":
            # BRRR TP3 stage-3 A/B: the fused kernel addresses the exact-tile (n64) layout; force it only for the
            # native-scale layers it serves, never for CSF layers (b12x's micro path mis-reads tail-less n64).
            os.environ["B12X_FORCE_N64_REPACK"] = "0" if online_csf else "1"
        weight_plan = fused_moe.plan_weights(
            constraints=(
                fused_moe.WeightPlanConstraints(scale_compression="csf")
                if online_csf
                else None
            ),
            source=fused_moe.PackedSource(
                format=fused_moe.PackedSourceFormat(self._source_format),
                w13_layout=fused_moe.W13Layout(self._w13_layout),
            ),
            activation=fused_moe.ActivationSpec(
                mode=mode,
                nonlinearity=_b12x_activation_name(activation),
                io_dtype=params_dtype,
                swiglu_limit=limit,
                swiglu_alpha=alpha,
                swiglu_beta=beta,
                a16_max_tokens=self._a16_max_tokens,
            ),
            geometry=fused_moe.MoEGeometry(
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
            ),
        )
        # Lossless per-row scale compression (FlashInfer b12x
        # scale_compression="csf"): the prepared experts keep compressed scales
        # and expand them into scratch shared by every layer on this device;
        # layers run serially on one stream.
        scale_scratch = (
            _shared_csf_scale_scratch(w1_scale, w2_scale) if online_csf else None
        )
        if scale_scratch is not None and self._source_format == "modelopt_nvfp4":
            # PATCH(ds41-nvfp4): b12x expands NVFP4 scales straight into the caller's
            # buffers and indexes them as [experts, rows, cols] (swizzled layout), so hand
            # it views shaped like the swizzled block-scale tensors, not flat bytes.
            scale_scratch = tuple(
                buf[: t.numel() * t.element_size()].view(t.dtype).view(t.shape)
                for buf, t in zip(scale_scratch, (w1_scale, w2_scale))
            )
        return fused_moe.prepare_weights(
            plan=weight_plan,
            scale_scratch=scale_scratch,
            weights=fused_moe.PackedWeights(
                w13=w1,
                w2=w2,
                w13_block_scales=w1_scale,
                w2_block_scales=w2_scale,
                w13_global_scales=w1_global_scale,
                w2_global_scales=w2_global_scale,
                input_scale=a1_gscale,
                intermediate_scale=a2_gscale,
                # Loaded scales are fixed while prepared plans and their
                # captured CUDA graphs execute.
                immutable_input_scales=True,
            ),
        )

    def _refresh_quant_config(self, layer: torch.nn.Module) -> None:
        if self._source_format in ("iq2_xs", "iq2_xxs", "q8_0"):
            return
        self.quant_config._w1.scale = layer.w13_weight_scale
        self.quant_config._w2.scale = layer.w2_weight_scale
        if self._source_format != "modelopt_nvfp4":
            return

        self.quant_config._w1.alpha_or_gscale = layer.w13_weight_scale_2
        self.quant_config._w2.alpha_or_gscale = layer.w2_weight_scale_2
        if self._quant_mode in ("nvfp4", "w4a8_nvfp4") or (
            self._a4_prefill_enabled
            and getattr(layer, "w13_input_scale", None) is not None
            and getattr(layer, "w2_input_scale", None) is not None
        ):
            self.quant_config._a1.alpha_or_gscale = 1.0 / layer.w13_input_scale
            self.quant_config._a2.alpha_or_gscale = 1.0 / layer.w2_input_scale

    def _release_source_parameters(self, layer: torch.nn.Module) -> None:
        if self._source_parameters_released:
            return
        w1_scale = _replace_parameter_with_empty(layer, "w13_weight_scale")
        w2_scale = _replace_parameter_with_empty(layer, "w2_weight_scale")
        if w1_scale is not None:
            self.quant_config._w1.scale = w1_scale
        if w2_scale is not None:
            self.quant_config._w2.scale = w2_scale
        _replace_parameter_with_empty(layer, "w13_weight")
        _replace_parameter_with_empty(layer, "w2_weight")
        self._source_parameters_released = True

    def _reuse_prepared_storage(self, layer: torch.nn.Module, prepared: Any) -> Any:
        previous = getattr(layer, "_b12x_prepared_experts", None)
        prepared = reuse_packed_weight_storage(previous, prepared)
        if prepared is not previous:
            self._plan = None
            self._plan_key = None
        self._prepared_experts = prepared
        layer._b12x_prepared_experts = prepared
        return prepared

    def _attach_expert_banks(self, layer: torch.nn.Module, prepared: Any) -> None:
        """Publish pointer banks when the placement opts in via env."""

        budget = os.environ.get(BANK_BUDGET_ENV)
        deferred = os.environ.get("B12X_DEFERRED_WEIGHT_BANKS", "0") == "1"
        if not deferred and os.environ.get("B12X_DEFERRED_INPLACE", "0") == "1":
            from vllm.model_executor.offloader.vmm_shim import vmm_owner_for_tensor

            impl = getattr(prepared, "_impl", prepared)
            # PATCH(ds41-nvfp4): use this layer's planned quant mode (w4a8_mx for MXFP4-CSF,
            # nvfp4/w4a8_nvfp4/w4a16 for ModelOpt NVFP4) instead of assuming w4a8_mx.
            payload = impl.representation_for(self._quant_mode)
            if payload is not None and hasattr(payload, "w13_rp"):
                w13_t, w2_t = payload.w13_rp, payload.w2_rp          # repacked (w4a8_mx)
            else:
                # ModelOpt NVFP4: no repack; the kernels read the canonical packed
                # weights, which are the loader's (VMM-wrapped) parameter storage.
                w13_t, w2_t = getattr(impl, "w1_fp4", None), getattr(impl, "w2_fp4", None)
            if w13_t is None or w2_t is None:
                logger.warning(
                    "b12x in-place promotion unavailable for quant mode %s (layer %s)",
                    self._quant_mode, getattr(layer, "layer_name", "?"),
                )
                return
            owners = (
                vmm_owner_for_tensor(w13_t),
                vmm_owner_for_tensor(w2_t),
            )
            if all(owner is not None for owner in owners):
                self._inplace_owners = owners
                _PENDING_INPLACE_LAYERS.add(self)
            else:
                import re as _re
                _m = _re.search(r"layers\.(\d+)\.", str(getattr(layer, "layer_name", "")))
                _lo = int(os.environ.get("DS41_OFFLOAD_FROM_ORDINAL", "20"))
                _hi = int(os.environ.get("DS41_OFFLOAD_TO_ORDINAL", "22"))
                _parked = _m is not None and _lo <= int(_m.group(1)) <= _hi
                (logger.warning if _parked else logger.debug)(
                    "b12x in-place promotion: layer %s expert storage is not VMM-owned "
                    "(mode %s); experts stay resident",
                    getattr(layer, "layer_name", "?"), self._quant_mode,
                )
            return
        if deferred:
            if self._quant_mode != "w4a8_mx":
                raise BankModeRejected("deferred weight banks support the w4a8_mx path only")
            if os.environ.get("B12X_SCALE_CODEC", "0") != "1":
                raise BankModeRejected("deferred weight banks require B12X_SCALE_CODEC=1")
            if budget is not None and os.environ.get(BANK_ENABLE_ENV, "1") != "0":
                raise BankModeRejected(
                    "deferred weight banks replace startup expert banks"
                )
            impl = getattr(prepared, "_impl", prepared)
            payload = impl.representation_for("w4a8_mx")
            from vllm.model_executor.offloader.vmm_shim import vmm_owner_for_tensor

            owners = (
                vmm_owner_for_tensor(payload.w13_rp),
                vmm_owner_for_tensor(payload.w2_rp),
            )
            if any(owner is None for owner in owners):
                return
            mapping = {
                "w13_bank": payload.w13_rp,
                "down_bank": payload.w2_rp,
                "sfbw13_bank": payload.w13_sfb,
                "sfbdown_bank": payload.w2_sfb,
            }
            banks = DeferredWeightBanks(
                mapping,
                num_experts=int(impl.num_experts),
                device=impl.w1_fp4.device,
                owners=owners,
            )
            _PENDING_DEFERRED_BANKS.add(banks)
            from b12x.moe.fused_moe._impl import attach_expert_banks as _attach_banks

            _attach_banks(impl, banks)
            logger.warning(
                "b12x deferred weight banks layer=%s %s",
                getattr(layer, "_b12x_layer_index", "?"),
                banks.receipt(),
            )
            return
        if budget is None or os.environ.get(BANK_ENABLE_ENV, "1") == "0":
            return
        if os.environ.get("B12X_SCALE_CODEC", "0") == "1":
            raise BankModeRejected(
                "B12X_SCALE_CODEC and startup expert pointer banks are mutually exclusive"
            )
        if self._quant_mode != "w4a8_mx":
            raise BankModeRejected("startup expert banks support the w4a8_mx path only")
        impl = getattr(prepared, "_impl", prepared)
        payload = impl.representation_for("w4a8_mx")
        if bool(getattr(payload, "n64_repack", False)):
            raise BankModeRejected(
                "expert banks support the N128 repacked w4a8_mx path only; "
                "the N64 repack variant must keep legacy arithmetic"
            )
        mapping = {
            "w13_bank": payload.w13_rp,
            "down_bank": payload.w2_rp,
            "sfbw13_bank": payload.w13_sfb,
            "sfbdown_bank": payload.w2_sfb,
        }
        num_experts = int(impl.num_experts)
        strides = [
            _slot_stride_bytes(mapping[name], num_experts) for name in BANK_SLOTS
        ]
        histogram = parse_histogram_env(
            os.environ.get(BANK_HISTOGRAM_ENV), num_experts=num_experts
        )
        hot_ids = select_hot_experts(
            histogram,
            num_experts=num_experts,
            expert_bytes=sum(strides),
            budget_bytes=int(budget),
        )
        banks = build_expert_banks(
            mapping,
            num_experts=num_experts,
            hot_ids=hot_ids,
            device=impl.w1_fp4.device,
        )
        from b12x.moe.fused_moe._impl import attach_expert_banks as _attach_banks
        _attach_banks(impl, banks)
        logger.info(
            "b12x expert banks layer=%s %s",
            getattr(layer, "_b12x_layer_index", "?"),
            bank_receipt(banks, expert_bytes=sum(strides)),
        )


    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self._source_format == "exl3":
            raise RuntimeError("EXL3 weights require install_prepared_experts")
        self._apply_router_weight_on_input = layer.apply_router_weight_on_input
        if self._apply_router_weight_on_input and self._quant_mode != "w4a16":
            raise ValueError(
                "b12x MoE supports apply_router_weight_on_input only with W4A16"
            )
        self._source_parameters_released = False
        self._refresh_quant_config(layer)
        self._layer_name = str(getattr(layer, "layer_name", ""))  # BRRR TP3 stage-3 A/B: per-layer CSF gate
        prepared = self._prepare_experts(
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            activation=layer.activation,
            params_dtype=self.moe_config.in_dtype,
        )
        prepared = self._reuse_prepared_storage(layer, prepared)
        if os.environ.get("B12X_SCALE_CODEC", "0") == "1" and self._quant_mode == "w4a8_mx":
            payload = prepared._impl.representation_for("w4a8_mx")
            logger.warning(
                "b12x scale codec layer=%s w13_dtype=%s w2_dtype=%s "
                "encoded_bytes=%d",
                getattr(layer, "_b12x_layer_index", "?"),
                payload.w13_sfb.dtype,
                payload.w2_sfb.dtype,
                payload.w13_sfb.nbytes + payload.w2_sfb.nbytes,
            )
        if prepared.plan._impl.discards_source_parameters:
            self._release_source_parameters(layer)
        self._attach_expert_banks(layer, prepared)
        if not getattr(layer, "b12x_preparation_suppressed", False):
            set_b12x_preparation_provider(layer, self)
        if os.environ.get("B12X_EXPERT_ROUTES_ALL", "0") == "1":
            banks = getattr(getattr(prepared, "_impl", prepared), "banks", None)
            if not isinstance(banks, DeferredWeightBanks):
                import re

                layer_name = getattr(layer, "layer_name", "")
                match = re.search(r"layers\.(\d+)", layer_name)
                if match is None:
                    raise ValueError(
                        "all-layer route monitor cannot parse layer id from "
                        f"{layer_name!r}"
                    )
                self._route_monitor = ExpertRouteMonitor(
                    layer_ordinal=int(match.group(1)),
                    num_experts=int(prepared.num_experts),
                    device=prepared.device,
                )
        _register_b12x_moe_output_collective(
            layer, hidden_size=int(prepared.hidden_size)
        )

    def install_prepared_experts(self, layer: torch.nn.Module, prepared: Any) -> None:
        """Install canonical prepared experts without retaining source parameters."""
        fused_moe = _require_b12x_fused_moe()
        if not isinstance(prepared, fused_moe.PreparedExperts):
            raise TypeError("Installation requires B12X PreparedExperts")
        trellis = self._source_format == "exl3" and isinstance(
            prepared.plan.source, fused_moe.TrellisSource
        )
        packed = (
            self._source_format == "fp4_e8m0_k32"
            and isinstance(prepared.plan.source, fused_moe.PackedSource)
            and prepared.plan.source.format == "fp4_e8m0_k32"
        )
        nvfp4 = (
            self._source_format == "modelopt_nvfp4"
            and self._quant_mode in ("nvfp4", "w4a16")
            and isinstance(prepared.plan.source, fused_moe.PackedSource)
            and prepared.plan.source.format == "modelopt_nvfp4"
            and prepared.plan.source.w13_layout in (self._w13_layout, "w13")
        )
        if not (trellis or packed or nvfp4):
            raise TypeError("Prepared expert encoding does not match the backend")
        if (
            prepared.num_experts != self.moe_config.num_experts
            or prepared.hidden_size != self.moe_config.hidden_dim
            or prepared.intermediate_size
            != self.moe_config.intermediate_size_per_partition
            or prepared.plan.activation.mode
            != (
                "a4"
                if nvfp4 and self._quant_mode == "nvfp4"
                else "a8"
                if packed and self._quant_mode == "w4a8_mx"
                else "a16"
            )
            or (trellis and prepared.plan.activation.rotation_dtype != torch.float16)
            or prepared.plan.activation.io_dtype != self.moe_config.in_dtype
            or prepared.plan.activation.nonlinearity
            != _b12x_activation_name(layer.activation)
            or prepared.plan.activation.swiglu_limit != self.moe_config.swiglu_limit
            or layer.apply_router_weight_on_input
        ):
            raise ValueError(
                "Prepared expert geometry, activation or routing "
                "does not match the layer"
            )
        self._apply_router_weight_on_input = False
        self._reuse_prepared_storage(layer, prepared)
        set_b12x_preparation_provider(layer, self)
        _register_b12x_moe_output_collective(layer, hidden_size=prepared.hidden_size)

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: mk.FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        """Report whether b12x can run this MoE config, with the reason if not.

        Applies the kernel-specific bias, dtype, activation and alignment
        gates, then defers to the base-class checks.
        """
        if moe_config.has_bias:
            return False, "kernel does not support expert biases"
        if moe_config.in_dtype not in (torch.float16, torch.bfloat16):
            return (
                False,
                f"kernel does not support {moe_config.in_dtype} input/output dtype",
            )
        if moe_config.activation == MoEActivation.SITU and (
            moe_config.activation_situ_beta != 4.0
            or moe_config.activation_situ_linear_beta != 25.0
        ):
            return False, "kernel supports only SiTU beta=4 and linear_beta=25"
        if (
            activation_key is not None
            and moe_config.activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE
        ):
            return (
                False,
                "kernel does not support swigluoai_uninterleave with W4A8",
            )
        unpadded_intermediate_size = (
            moe_config.intermediate_size_per_partition_unpadded
            or moe_config.intermediate_size_per_partition
        )
        if weight_key == kMxfp4Static and unpadded_intermediate_size % 32 != 0:
            return (
                False,
                "MXFP4 requires the per-rank intermediate size to be divisible by 32",
            )
        if weight_key == kMxfp4Static and activation_key == kMxfp8Dynamic:
            if moe_config.activation not in (
                MoEActivation.SILU,
                MoEActivation.SITU,
            ):
                return False, "MXFP4 W4A8 supports only SiLU and SiTU"
            if (
                moe_config.hidden_dim % 256 != 0
                or moe_config.intermediate_size_per_partition % 32 != 0
            ):
                return (
                    False,
                    (
                        "MXFP4 W4A8 requires hidden size divisible by 256 and "
                        "per-rank intermediate size divisible by 32"
                    ),
                )
        if weight_key == kMxfp8Static:
            if not _b12x_has_mxfp8_moe():
                return False, "the installed b12x has no MXFP8 W8A8 MoE recipe"
            if moe_config.activation != MoEActivation.SILU:
                return False, "MXFP8 W8A8 supports only SiLU"
            if moe_config.hidden_dim % 128 != 0 or unpadded_intermediate_size % 32 != 0:
                return (
                    False,
                    (
                        "MXFP8 W8A8 requires hidden size divisible by 128 and "
                        "per-rank intermediate size divisible by 32"
                    ),
                )
        return mk.FusedMoEExperts.is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )

    @staticmethod
    def _supports_current_device() -> bool:
        if not (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(120)
        ):
            return False
        fused_moe = get_b12x_fused_moe()
        if fused_moe is None:
            return False
        return fused_moe.is_supported()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        """Check the (weight, activation) pair against the b12x kernel set."""
        return (weight_key, activation_key) in (
            (kMxfp4Static, kMxfp8Dynamic),
            (kMxfp4Static, None),
            (kMxfp8Static, kMxfp8Dynamic),
            (kNvfp4Static, kNvfp4Dynamic),
            (kNvfp4Static, kMxfp8Dynamic),
            (kNvfp4Static, None),
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (
            MoEActivation.SILU,
            MoEActivation.SITU,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            MoEActivation.RELU2_NO_MUL,
        )

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return (
            not moe_parallel_config.use_ep
            and moe_parallel_config.ep_size == 1
            and not moe_parallel_config.use_all2all_kernels
            and not moe_parallel_config.enable_eplb
        )

    def _promote_inplace(self) -> None:
        """Remap this layer's VMM expert regions to device memory once armed.

        Regions are promoted whole (w13, then w2) at their canonical VA, so
        captured graphs and every kernel path read device memory afterwards.
        """
        owners = getattr(self, "_inplace_owners", None)
        if owners is None or self._deferred_bank_promoted:
            return
        arm_file = os.environ.get("B12X_DEFERRED_BANK_ARM_FILE", "")
        if not arm_file or not os.path.exists(arm_file):
            return
        if torch.cuda.is_current_stream_capturing():
            return
        with self._deferred_bank_lock:
            if self._deferred_bank_promoted:
                return
            reserve = int(
                os.environ.get("B12X_DEFERRED_BANK_RESERVE_BYTES", str(1 << 30))
            )
            torch.cuda.empty_cache()
            free_bytes, _ = torch.cuda.mem_get_info()
            budget = max(0, int(free_bytes) - reserve)
            chunked = all(getattr(o, "num_chunks", 0) for o in owners)
            if chunked:
                # Equal share per pending layer; each layer's chunks hold its
                # most-routed experts first (hot-first relabeling at load).
                budget //= max(1, len(_PENDING_INPLACE_LAYERS))
            _PENDING_INPLACE_LAYERS.discard(self)
            promoted = 0
            if chunked:
                progress = True
                while progress:
                    progress = False
                    for owner in owners:
                        index = owner._chunk_promoted
                        if index >= owner.num_chunks:
                            continue
                        size = owner._cold_chunks[index][1]
                        if promoted + size > budget:
                            continue
                        promoted += owner.promote_chunk()
                        progress = True
            else:
                for owner in owners:
                    promoted += owner.promote_up_to(max(0, budget - promoted))
            self._deferred_bank_promoted = True
            logger.warning(
                "b12x in-place VMM promotion free_before=%d reserve=%d "
                "promoted=%d regions_hot=%d/%d chunks_hot=%s",
                free_bytes,
                reserve,
                promoted,
                sum(1 for o in owners if o.hot_bytes),
                len(owners),
                [f"{o._chunk_promoted}/{o.num_chunks}" for o in owners] if chunked else "-",
            )

    def _promote_deferred_banks(self, prepared: Any) -> None:
        banks = getattr(getattr(prepared, "_impl", prepared), "banks", None)
        arm_file = os.environ.get("B12X_DEFERRED_BANK_ARM_FILE", "")
        if not arm_file or not os.path.exists(arm_file):
            return
        if not isinstance(banks, DeferredWeightBanks) or self._deferred_bank_promoted:
            return
        if torch.cuda.is_current_stream_capturing():
            return
        with self._deferred_bank_lock:
            if self._deferred_bank_promoted:
                return
            reserve = int(
                os.environ.get("B12X_DEFERRED_BANK_RESERVE_BYTES", str(1 << 30))
            )
            # mem_get_info cannot see blocks the caching allocator holds but
            # does not use (~1.25 GiB/rank measured); return them first so the
            # budget covers them. Each bank's slab is one large allocation, so
            # those scattered free blocks could not be reused in place anyway.
            torch.cuda.empty_cache()
            free_bytes, _ = torch.cuda.mem_get_info(prepared.device)
            budget = max(0, int(free_bytes) - reserve)
            if os.environ.get("B12X_DEFERRED_BANK_EVEN_SPLIT", "0") == "1":
                budget //= max(1, len(_PENDING_DEFERRED_BANKS))
            _PENDING_DEFERRED_BANKS.discard(banks)
            if os.environ.get("B12X_DEFERRED_INPLACE", "0") == "1":
                # Remap the layer's canonical VMM regions to device memory at
                # the same VA instead of copying experts into separate slabs.
                # Every kernel path, banked or not, then reads device memory,
                # and captured graphs keep valid addresses. Promotion is whole
                # regions (w13, then w2); the cold bank tables already point
                # at those canonical addresses.
                promoted = 0
                for owner in banks.owners:
                    promoted += owner.promote_up_to(max(0, budget - promoted))
                banks.enable_routes()
                self._deferred_bank_promoted = True
                logger.warning(
                    "b12x in-place VMM promotion free_before=%d reserve=%d "
                    "promoted=%d regions_hot=%d/%d",
                    free_bytes,
                    reserve,
                    promoted,
                    sum(1 for o in banks.owners if o.hot_bytes),
                    len(banks.owners),
                )
                return
            promoted = banks.promote_up_to(budget)
            banks.enable_routes()
            self._deferred_bank_promoted = True
            logger.warning(
                "b12x deferred weight promotion free_before=%d reserve=%d "
                "promoted=%d hot=%d cold=%d",
                free_bytes,
                reserve,
                promoted,
                len(banks.hot_ids),
                len(banks.cold_ids),
            )

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def _prepared(self) -> Any:
        if self._prepared_experts is None:
            raise RuntimeError(
                "b12x MoE weights must be prepared by process_weights_after_loading"
            )
        return self._prepared_experts

    def _execution_parts(self, tokens: int) -> tuple[tuple[int, int, bool], ...]:
        prepared = self._prepared()
        if self._a4_prefill_enabled and getattr(
            getattr(prepared, "_impl", None), "a4_prefill_scales", False
        ):
            return _a4_prefill_parts(tokens)
        return ((0, tokens, False),)

    def uses_expanded_nvfp4_scales(
        self, tokens: int, route_ids_dtype: torch.dtype
    ) -> bool:
        """Query whether this layer's semantic spans can use prefetched scales."""
        return any(
            _require_b12x_fused_moe().uses_expanded_nvfp4_scales(
                self._plan,
                num_tokens=end - start,
                a4_prefill=a4_prefill,
                scales_expanded=True,
                route_ids_dtype=route_ids_dtype,
            )
            for start, end, a4_prefill in self._execution_parts(tokens)
        )

    def _prepared_plan(
        self, *, activation: MoEActivation, apply_router_weight_on_input: bool
    ) -> Any:
        plan = self._plan
        if (
            plan is None
            or activation != self._plan_activation
            or bool(apply_router_weight_on_input) != self._plan_route_on_input
        ):
            raise PreparationResourceUnavailableError(
                "b12x MoE has no prepared plan for this activation/routing"
            )
        return plan

    def _plan_for_tokens(
        self,
        tokens: int,
        *,
        activation: MoEActivation,
        apply_router_weight_on_input: bool,
    ) -> Any:
        """Reuse the declared prefill capacity and exact graph variants."""
        plan = self._prepared_plan(
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
        assert self._plan_key is not None
        capacity = max(self._plan_key[0])
        if int(tokens) > capacity:
            raise ValueError(
                f"live MoE token count {tokens} exceeds the configured prefill "
                f"capacity {capacity}"
            )
        return plan

    def get_b12x_preparation_units(
        self, layer: torch.nn.Module, workload: B12xWorkload
    ) -> Sequence[B12xPreparationUnit]:
        from b12x.moe.fused_moe.workloads import TUNING_WORKLOAD_VERSION
        from b12x.preparation import FrozenMapping

        if workload.stage != "weights":
            return ()
        if workload.output_dtype != self.moe_config.in_dtype:
            raise ValueError("b12x MoE output dtype differs from its loaded contract")
        prepared = self._prepared()
        counts = tuple(sorted({workload.max_tokens, *workload.fixed_token_counts}))
        activation = layer.activation
        topk = int(self.moe_config.experts_per_token)
        route_on_input = bool(layer.apply_router_weight_on_input)
        fused_moe = _require_b12x_fused_moe()
        plan_key = (counts, activation, route_on_input, id(prepared))
        plan = self._plan if getattr(self, "_plan_key", None) == plan_key else None
        if plan is None:
            # The layer holds one plan for its serving shapes; a later call with
            # the same workload reuses it so the prepared state stays installed.
            plan = fused_moe.plan_execution(
                experts=prepared,
                capacity=fused_moe.ExecutionCapacity(
                    max_tokens=max(counts),
                    top_k=topk,
                    warmup_token_counts=counts,
                    route_num_experts=0,
                ),
                routing=fused_moe.RoutingSpec(
                    apply_router_weight_on_input=route_on_input,
                ),
                # Tuning choices depend on the routing corpus. Invalidate only
                # MoE choices when its distribution changes, not compiled code
                # or unrelated component selections.
                invocation=FrozenMapping(
                    {
                        "tuning_route_pattern": TUNING_WORKLOAD_VERSION,
                    }
                ),
            )
            self._plan = plan
            self._plan_key = plan_key
            self._plan_activation = activation
            self._plan_route_on_input = route_on_input
        name = f"fused_moe:{id(layer)}"
        if hasattr(plan, "token_counts"):
            calls = {
                count: _prepared_moe_call_factory(
                    tokens=count,
                    topk=topk,
                    prepared=prepared,
                    output_dtype=self.output_dtype,
                )
                for count in plan.token_counts
            }
            benchmark_calls = {
                count: _prepared_moe_call_factory(
                    tokens=count,
                    topk=topk,
                    prepared=prepared,
                    output_dtype=self.output_dtype,
                )
                for count in plan.token_counts
            }
            request = plan.request(
                name=name,
                prepare_calls=calls,
                benchmark_calls=benchmark_calls,
            )
        else:
            prepare_call = _prepared_moe_call_factory(
                tokens=counts[0],
                topk=topk,
                prepared=prepared,
                output_dtype=self.output_dtype,
            )
            benchmark_call = _prepared_moe_call_factory(
                tokens=counts[0],
                topk=topk,
                prepared=prepared,
                output_dtype=self.output_dtype,
            )
            request = plan.request(
                name=name,
                prepare_call=prepare_call,
                benchmark_call=benchmark_call,
            )
        key = (
            self._quant_mode,
            self._a16_max_tokens,
            self._source_format,
            self._w13_layout,
            activation,
            route_on_input,
            counts,
            workload.output_dtype,
        )
        return (
            B12xPreparationUnit(
                name=self._quant_mode.upper(),
                key=key,
                requests=(request,),
                stage="weights",
            ),
        )

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        """Return (experts, tokens, N, K, topk) for the current problem.

        Delegates to the base implementation while the weight tensors still
        hold data; once they are released, the geometry comes from the
        prepared plan, with N = 2 * intermediate (the gate and up rows).
        """
        if w1.numel() and w2.numel():
            return super().moe_problem_size(a1, w1, w2, topk_ids)
        prepared = self._prepared()
        tokens = int(a1.shape[0] if a1.ndim == 2 else a1.shape[1])
        return (
            int(prepared.num_experts),
            tokens,
            int(prepared.intermediate_size) * 2,
            int(a1.shape[-1]),
            int(topk_ids.shape[1]),
        )

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        del N, global_num_experts, local_num_experts, expert_tokens_meta
        plan = self._plan_for_tokens(
            int(M),
            activation=activation,
            apply_router_weight_on_input=self._apply_router_weight_on_input,
        )
        required_nbytes = sum(spec.nbytes for spec in plan.scratch_specs())
        itemsize = self.moe_config.in_dtype.itemsize
        return (0,), (max(1, (required_nbytes + itemsize - 1) // itemsize),), (M, K)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ) -> None:
        del w1, w2, global_num_experts
        del a1q_scale, a2_scale, workspace13, expert_tokens_meta
        if expert_map is not None:
            raise ValueError("b12x TP MoE does not support expert maps")
        if bool(apply_router_weight_on_input) != self._apply_router_weight_on_input:
            raise ValueError(
                "apply_router_weight_on_input does not match the prepared b12x MoE plan"
            )
        prepared = self._prepared()
        self._promote_deferred_banks(prepared)
        self._promote_inplace()
        banks = getattr(getattr(prepared, "_impl", prepared), "banks", None)
        if isinstance(banks, DeferredWeightBanks):
            banks.record_routes(topk_ids)
        # Native routes are specialized for both int32 and int64 identifiers
        # during materialization.  Preserve the caller's representation: a
        # conversion here would allocate during capture and would silently
        # discard the prepared int64 path.
        if (
            topk_ids.dtype not in (torch.int32, torch.int64)
            or not topk_ids.is_contiguous()
        ):
            raise TypeError("b12x MoE topk_ids must be contiguous int32 or int64")
        if self._route_monitor is not None:
            arm_file = os.environ.get("B12X_DEFERRED_BANK_ARM_FILE", "")
            if arm_file and os.path.exists(arm_file):
                self._route_monitor.enable()
            self._route_monitor.record_routes(topk_ids)
        topk_weights = _normalize_topk_weights(topk_weights)
        plan = self._plan_for_tokens(
            int(hidden_states.shape[0]),
            activation=activation,
            apply_router_weight_on_input=bool(apply_router_weight_on_input),
        )
        if workspace2 is None or not workspace2.is_contiguous():
            raise ValueError("b12x MoE requires contiguous caller-owned workspace2")
        scratch = workspace2.view(-1).view(torch.uint8)
        expanded = {"scales_expanded": True} if self.scales_expanded else {}
        if getattr(getattr(prepared, "_impl", None), "a4_prefill_scales", False):
            tokens = int(hidden_states.shape[0])
            for start, end, a4_prefill in self._execution_parts(tokens):
                binding = _require_b12x_fused_moe().bind(
                    plan,
                    scratch=scratch,
                    a=hidden_states[start:end],
                    experts=prepared,
                    topk_weights=topk_weights[start:end],
                    topk_ids=topk_ids[start:end],
                    output=output[start:end],
                    input_scales_static=True,
                    a4_prefill=a4_prefill,
                    **expanded,
                )
                _require_b12x_fused_moe().run(binding=binding)
            return
        binding = _require_b12x_fused_moe().bind(
            plan,
            scratch=scratch,
            a=hidden_states,
            experts=prepared,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            output=output,
            input_scales_static=True,
            **expanded,
        )
        _require_b12x_fused_moe().run(binding=binding)

    def moe_sum(self, input: torch.Tensor, output: torch.Tensor) -> None:
        raise NotImplementedError("LoRA is not supported for B12xExperts")


def _register_b12x_moe_output_collective(
    layer: torch.nn.Module, *, hidden_size: int
) -> None:
    """Describe the rank-local routed-MoE output to the existing TP transport."""
    from vllm.distributed.device_communicators.b12x_pcie_all_reduce import (
        B12xPcieInvocation,
    )
    from vllm.distributed.parallel_state import register_b12x_collective_describer

    def describe(workload):
        prefix = layer.layer_name
        return tuple(
            B12xPcieInvocation(
                name=f"{prefix}.moe_output_all_reduce.m{rows}.lane{workload.lane}",
                operation="all_reduce",
                shape=(rows, hidden_size),
                dtype=workload.output_dtype,
            )
            for rows in workload.token_counts
        )

    register_b12x_collective_describer(layer, describe)
