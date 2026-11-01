# azallaza/ds41-kraken-tp3:20261019.7 = 20261019.5 + BRRR decode kernels ported to TP3 (stages 1, 2a, 2b).
# From fraserprice/DeepSeek-4.1-Flash-1017-BRRR (Apache-2.0; LICENSE-brrr + NOTICE-brrr kept): skinny FP8 GEMM for
# <=8 decode rows, radix indexer top-k, phased PCIe all-reduce/all-gather rebuilt for 3 ranks, vocab-split DSpark
# draft sampling + verification made padding-aware for the TP3 vocab (129,280 -> 129,408) with rank lanes padded to a
# power of two, plus BRRR's renderer fix (tool-result-only messages don't start a new turn; parallel tool results
# sorted by call order). Sparse decode attention and the fused decoder are NOT included (24-head tiling / CSF scales).
# The vocab-split path needs the DSpark markov head sharded across TP (ENV below; off = stock replicated head and
# the split sampler disables itself). Build from the release dir (kernels compile for sm_120a, ~3 min):
#   docker build -f patch-brrr/image/Dockerfile -t azallaza/ds41-kraken-tp3:20261019.7 .
FROM azallaza/ds41-kraken-tp3:20261019.5
ARG SP=/opt/venv/lib/python3.12/site-packages
COPY patch-brrr/brrr     $SP/brrr
COPY patch-brrr/brrr.pth $SP/brrr.pth
COPY patch-brrr/LICENSE-brrr patch-brrr/NOTICE-brrr /opt/brrr/
COPY patch-brrr/vllm-patched/models/deepseek_v4_1/attention.py   $SP/vllm/models/deepseek_v4_1/attention.py
COPY patch-brrr/vllm-patched/renderers/deepseek_v4.py            $SP/vllm/renderers/deepseek_v4.py
COPY patch-brrr/vllm-patched/tokenizers/deepseek_v41_encoding.py $SP/vllm/tokenizers/deepseek_v41_encoding.py
ENV TORCH_EXTENSIONS_DIR=/opt/brrr-ext \
    VLLM_DSPARK_SHARD_MARKOV_HEAD=1 \
    VLLM_DSPARK_REPLICATE_MARKOV_W1=1
RUN mkdir -p /opt/brrr-ext && cd /tmp && MAX_JOBS=16 /opt/venv/bin/python -c "from brrr import cuext; [cuext.extension(n) for n in ('skinny','candidates','phased_all_reduce','sparse_decode')]" \
    && find $SP/vllm/models/deepseek_v4_1/__pycache__ $SP/vllm/renderers/__pycache__ $SP/vllm/tokenizers/__pycache__ -name "*.pyc" -delete 2>/dev/null || true
LABEL org.opencontainers.image.description="DS41 kraken TP3 20261019.5 + BRRR TP3 kernels (skinny GEMM, radix top-k, phased all-reduce, vocab-split sampling) + renderer tool-loop fix"
