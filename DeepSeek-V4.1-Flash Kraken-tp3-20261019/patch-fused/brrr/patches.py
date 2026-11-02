from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module
    from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
    from vllm.model_executor.layers.fused_moe.b12x import B12xExperts
    from vllm.models.deepseek_v4_1.b12x_layers import B12xFP8LinearMethod
    from vllm.models.deepseek_v4_1.common.engram import Engram
    from vllm.models.deepseek_v4_1.nvidia.model_state import DeepseekV41ModelState
    from vllm.sampling_params import SamplingParams
    from vllm.utils.b12x import B12xPreparationUnit, B12xWorkload
    from vllm.v1.core.sched.output import GrammarOutput
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.sample.output import SamplerOutput
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
    from vllm.v1.worker.gpu.states import RequestState

type Patch = Callable[[ModuleType], None]


def patch_engram(module: ModuleType) -> None:
    from brrr.engram_nvme import NvmeEngramEmbedding

    module.ParallelEngramEmbedding = NvmeEngramEmbedding
    engram = module.Engram
    base_forward, base_units = engram.forward, engram.get_b12x_preparation_units

    def forward(self: Engram, hidden_states: Tensor, hash_ids: Tensor, token_mask: Tensor | None = None) -> Tensor:
        self.embed_tokens.wait(self.staged_rows, hash_ids.shape[0])
        return base_forward(self, hidden_states, hash_ids, token_mask)

    def get_b12x_preparation_units(self: Engram, layer: Module, workload: B12xWorkload) -> tuple[B12xPreparationUnit, ...]:
        return tuple(unit for unit in base_units(self, layer, workload) if unit.name != "EngramLookup")

    engram.forward = forward
    engram.get_b12x_preparation_units = get_b12x_preparation_units


def patch_communicator(module: ModuleType) -> None:
    from brrr.phased_all_reduce import PhasedAllReduce

    communicator = module.CudaCommunicator
    base_init, base_all_reduce, base_all_gather, base_destroy = communicator.__init__, communicator.all_reduce, communicator.all_gather, communicator.destroy

    def init(self: CudaCommunicator, *args: object, **kwargs: object) -> None:
        base_init(self, *args, **kwargs)
        self.phased_ar_comm = PhasedAllReduce(self.cpu_group, self.device) if self.use_b12x_allreduce and self.world_size > 1 else None

    def all_reduce(self: CudaCommunicator, input_: Tensor) -> Tensor:
        phased, b12x = self.phased_ar_comm, self.b12x_ar_comm
        if phased is not None and phased.should_use(input_) and (b12x is None or b12x.disabled or not b12x.should_custom_ar(input_)):
            return phased.all_reduce(input_)
        return base_all_reduce(self, input_)

    def all_gather(self: CudaCommunicator, input_: Tensor, dim: int = -1) -> Tensor:
        phased = self.phased_ar_comm
        return phased.all_gather(input_, dim) if phased is not None and phased.should_gather(input_, dim) else base_all_gather(self, input_, dim)

    def destroy(self: CudaCommunicator) -> None:
        self.phased_ar_comm = None
        base_destroy(self)

    communicator.__init__ = init
    communicator.all_reduce = all_reduce
    communicator.all_gather = all_gather
    communicator.destroy = destroy


def patch_dspark(module: ModuleType) -> None:
    from brrr import vp_sampling

    speculator = module.DSparkSpeculator
    base_load, base_sample = speculator.load_draft_model, speculator._sample_sequential

    def load_draft_model(self: DSparkSpeculator, target_model: Module, target_attn_layer_names: set[str]) -> Module:
        model = base_load(self, target_model, target_attn_layer_names)
        self.vp = vp_sampling.shard_drafts(self, model)
        return model

    def _sample_sequential(self: DSparkSpeculator, num_reqs: int, head_hidden: Tensor) -> None:
        return base_sample(self, num_reqs, head_hidden) if self.vp is None else vp_sampling.draft(self, num_reqs, head_hidden)

    speculator.load_draft_model = load_draft_model
    speculator._sample_sequential = _sample_sequential


def patch_rejection(module: ModuleType) -> None:
    from brrr import vp_sampling

    sampler = module.RejectionSampler
    base_call = sampler.__call__

    def call(self: RejectionSampler, logits: Tensor, input_batch: InputBatch, draft_logits: Tensor | None = None) -> SamplerOutput:
        return base_call(self, logits, input_batch, vp_sampling.full_draft_logits(draft_logits))

    sampler.__call__ = call


def patch_runner(module: ModuleType) -> None:
    import numpy as np
    from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
    from vllm.v1.worker.gpu.sample.output import SamplerOutput
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS

    from brrr import vp_sampling

    runner = module.GPUModelRunner
    base_sample = runner.sample
    base_add_request = Sampler.add_request

    def add_request(self: Sampler, req_idx: int, prompt_len: int, sampling_params: SamplingParams) -> None:
        base_add_request(self, req_idx, prompt_len, sampling_params)
        if not hasattr(self, "vocab_parallel"):
            self.vocab_parallel = np.zeros_like(self.needs_logits_processing)
        bias, states, budget = self.logit_bias_state, self.sampling_states, self.thinking_budget_state
        self.vocab_parallel[req_idx] = not (
            bias.num_allowed_token_ids.np[req_idx]
            or bias.num_logit_bias.np[req_idx]
            or self.penalties_state.use_penalty[req_idx]
            or self.bad_words_state.num_bad_words.np[req_idx]
            or (budget.enabled and budget.use_thinking_budget[req_idx])
            or states.min_p.np[req_idx] != 0.0
            or states.top_k.np[req_idx] != states.vocab_size
            or states.top_p.np[req_idx] != 1.0
        )

    def sample(self: GPUModelRunner, hidden_states: Tensor, input_batch: InputBatch, grammar_output: GrammarOutput | None) -> tuple[SamplerOutput, Tensor, Tensor]:
        shard, rs, sampler = getattr(self.speculator, "vp", None), self.rejection_sampler, self.sampler
        head = getattr(self.model, "language_model", self.model).lm_head
        if not (
            shard is not None
            and head.shard_indices.org_vocab_start_index == shard.start
            and self.batch_sharder is None
            and grammar_output is None
            and input_batch.num_reqs > 0
            and input_batch.num_draft_tokens > 0
            and type(rs) is module.RejectionSampler
            and rs.use_block_verification
            and rs.synthetic_conditional_rates is None
            and rs.watermark_key is None
            and not sampler.compute_nans
            and not sampler.use_fp64_gumbel
            and not sampler.return_sampling_mask
            and np.all(sampler.vocab_parallel[input_batch.idx_mapping_np])
            and (sampler.trace_replay_state is None or not np.any(sampler.trace_replay_state.trace_len.np[input_batch.idx_mapping_np] > 0))
            and sampler.sampling_states.max_num_logprobs(input_batch.idx_mapping_np) == NO_LOGPROBS
        ):
            return base_sample(self, hidden_states, input_batch, grammar_output)
        states = sampler.sampling_states
        logits = self.model.compute_logits_local(hidden_states[input_batch.logits_indices])
        if np.any(sampler.logit_bias_state.num_stop_token_ids.np[input_batch.idx_mapping_np]):
            vp_sampling.mask_stops(logits, input_batch, sampler.logit_bias_state, shard)
        sampled, num_sampled = vp_sampling.verify(logits, input_batch, self.speculator, states.temperature.gpu, states.seeds.gpu)
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            num_sampled, input_batch.seq_lens, input_batch.cu_num_logits, input_batch.idx_mapping, sampler.req_states.prefill_len.gpu
        )
        output = SamplerOutput(sampled_token_ids=sampled, logprobs_tensors=None, num_nans=None, num_sampled=num_sampled, num_rejected=num_rejected)
        return output, num_sampled, num_rejected

    runner.sample = sample
    Sampler.add_request = add_request


def patch_model_state(module: ModuleType) -> None:
    state_cls = module.DeepseekV41ModelState
    base_prepare_attn, base_prepare_inputs = state_cls.prepare_attn, state_cls.prepare_inputs

    def prepare_attn(self: DeepseekV41ModelState, input_batch: InputBatch, *args: object, **kwargs: object) -> dict[str, object]:
        req_states = getattr(self, "engram_req_states", None)
        if self.disk_engram_models and req_states is not None and self.lookback_token_ids is not None:
            window = self.lookback_token_ids
            all_token_ids = req_states.all_token_ids.gpu
            module._gather_lookback_kernel[(window.shape[0],)](
                window,
                input_batch.idx_mapping,
                req_states.num_computed_tokens.gpu,
                all_token_ids,
                all_token_ids.stride(0),
                input_batch.idx_mapping.shape[0],
                DEPTH=window.shape[1],
                BLOCK_DEPTH=module.triton.next_power_of_2(window.shape[1]),
            )
            for model in self.disk_engram_models:
                model.prepare_disk_engram(input_batch.input_ids, input_batch.query_start_loc, window)
            self.staged_engram_batch = input_batch
        return base_prepare_attn(self, input_batch, *args, **kwargs)

    def prepare_inputs(self: DeepseekV41ModelState, input_batch: InputBatch, req_states: RequestState) -> dict[str, Tensor | None]:
        self.engram_req_states = req_states
        if getattr(self, "staged_engram_batch", None) is not input_batch:
            return base_prepare_inputs(self, input_batch, req_states)
        models, self.disk_engram_models = self.disk_engram_models, ()
        try:
            return base_prepare_inputs(self, input_batch, req_states)
        finally:
            self.disk_engram_models = models
            self.staged_engram_batch = None

    state_cls.prepare_attn = prepare_attn
    state_cls.prepare_inputs = prepare_inputs


def patch_b12x(module: ModuleType) -> None:
    base = module.B12xExperts.process_weights_after_loading

    def process_weights_after_loading(self: B12xExperts, layer: Module) -> None:
        base(self, layer)
        layer.b12x_experts = self

    module.B12xExperts.process_weights_after_loading = process_weights_after_loading


def patch_linear(module: ModuleType) -> None:
    from brrr import skinny

    method = module.B12xFP8LinearMethod
    base_process, base_apply = method.process_weights_after_loading, method.apply

    def process_weights_after_loading(self: B12xFP8LinearMethod, layer: Module) -> None:
        base_process(self, layer)
        if getattr(layer, "skinny_target", False):
            skinny.prepare(layer)

    def apply(self: B12xFP8LinearMethod, layer: Module, x: Tensor, bias: Tensor | None = None) -> Tensor:
        return skinny.project(layer, x) if bias is None and skinny.eligible(layer, x) else base_apply(self, layer, x, bias)

    method.process_weights_after_loading = process_weights_after_loading
    method.apply = apply


def patch_indexer(module: ModuleType) -> None:
    from brrr import candidates

    module.run_row_topk = candidates.row_topk(module.run_row_topk)
    module._launch = candidates.sorted_launch(module._launch)


PATCHES: dict[str, Patch] = {
    # TP3 STAGE 1: rank-agnostic kernels only (skinny FP8 GEMM <=8 rows, sparse decode attention H=24,
    # radix indexer top-k). Excluded for now: NVMe Engram, phased all-reduce / vocab-split sampling (need
    # RANKS=3 validation), fused decoder (model.py wiring + CSF scales).
    # STAGE 2a: phased PCIe all-reduce / all-gather for the 16 KB..1 MB sizes b12x one-shot declines (RANKS=3 build)
    "vllm.distributed.device_communicators.cuda_communicator": patch_communicator,
    # STAGE 2b: vocab-split draft sampling + verification (padding-aware for TP3)
    "vllm.v1.worker.gpu.spec_decode.dspark.speculator": patch_dspark,
    "vllm.v1.worker.gpu.spec_decode.rejection_sampler": patch_rejection,
    "vllm.v1.worker.gpu.model_runner": patch_runner,
    "vllm.model_executor.layers.fused_moe.b12x": patch_b12x,
    "vllm.models.deepseek_v4_1.b12x_layers": patch_linear,
    "b12x.attention.dsa_indexer.mxfp4": patch_indexer,
}
