# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 checkpoint quantization configuration."""

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding


class DeepseekV41FP8Config(Fp8Config):
    is_scale_e8m0 = True
    _nvfp4_config = None

    # PATCH(ds41-nvfp4): ModelOpt mixed-precision checkpoints declare the routed-expert
    # algorithm as quantization_config.moe_quant_algo ("NVFP4" for
    # nvidia/DeepSeek-V4.1-Flash-1017-NVFP4); stock MXFP4 checkpoints omit it.
    @property
    def moe_quant_algo(self) -> str:
        from vllm.config import get_current_vllm_config

        try:
            hf_config = get_current_vllm_config().model_config.hf_config
        except Exception:
            return ""
        quant_cfg = getattr(hf_config, "quantization_config", None) or {}
        return str(quant_cfg.get("moe_quant_algo") or "").upper()

    def _experts_are_nvfp4(self, prefix: str) -> bool:
        """True for routed-expert layers listed in quantization_config.quantized_layers.

        nvidia/DeepSeek-V4.1-Flash-1017-NVFP4 quantizes layers.0..39 only; the MTP/dspark
        draft layers (built as layers.40..42) keep the source MXFP4 experts and must use
        the stock MXFP4 method, otherwise their weight_scale_2/input_scale stay zero.
        """
        import re
        from vllm.config import get_current_vllm_config

        try:
            hf_config = get_current_vllm_config().model_config.hf_config
        except Exception:
            return True
        quant_cfg = getattr(hf_config, "quantization_config", None) or {}
        quantized = quant_cfg.get("quantized_layers") or {}
        if not quantized:
            return True
        m = re.search(r"layers\.(\d+)\.ffn\.experts", prefix or "")
        return m is not None and f"layers.{m.group(1)}.ffn.experts" in quantized

    def _get_nvfp4_config(self):
        if self._nvfp4_config is None:
            from vllm.model_executor.layers.quantization.modelopt import (
                ModelOptNvFp4Config,
            )

            import os

            # DS41_NVFP4_ACTIVATIONS=a16 (default): BF16 activations into the FP4 experts
            # (W4A16, the b12x decode path lil ships for GLM/Qwen NVFP4; pair with
            # B12X_W4A16_A4_PREFILL=1 for FP4 prefill rows). a4: W4A4 everywhere.
            act = os.environ.get("DS41_NVFP4_ACTIVATIONS", "a16").lower()
            if act not in ("a16", "a4"):
                raise ValueError("DS41_NVFP4_ACTIVATIONS must be a16 or a4")
            self._nvfp4_config = ModelOptNvFp4Config(
                quant_method="W4A16_NVFP4" if act == "a16" else "NVFP4",
                is_checkpoint_nvfp4_serialized=True,
                kv_cache_quant_algo=None,
                exclude_modules=[],
                group_size=16,
            )
        return self._nvfp4_config

    @classmethod
    def get_name(cls):
        return "deepseek_v41_fp8"

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        if getattr(hf_config, "model_type", None) in (
            "deepseek_v41",
            "deepseek_v41_text",
        ) and hf_quant_cfg.get("quant_method") in ("fp8", "deepseek_v41_fp8"):
            return cls.get_name()
        return None

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase):
            from .b12x_layers import B12xFP8LinearMethod, B12xLinearMethod

            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
                match_mode=self.ignored_layers_match_mode,
            ):
                return B12xLinearMethod()
            return B12xFP8LinearMethod(self)
        if isinstance(layer, VocabParallelEmbedding):
            from .b12x_layers import B12xEmbeddingMethod

            return B12xEmbeddingMethod()
        if isinstance(layer, RoutedExperts):
            if self.moe_quant_algo == "NVFP4" and self._experts_are_nvfp4(prefix):
                from vllm.model_executor.layers.quantization.modelopt import (
                    ModelOptNvFp4FusedMoE,
                )

                return ModelOptNvFp4FusedMoE(
                    quant_config=self._get_nvfp4_config(),
                    moe_config=layer.moe_config,
                )
            return Mxfp4MoEMethod(layer.moe_config)
        return None
