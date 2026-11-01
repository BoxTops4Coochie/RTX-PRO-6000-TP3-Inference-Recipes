# azallaza/ds41-kraken-tp3:20261019.6 = 20261019.5 + BRRR stage 1/2a kernels ported to TP3.
# From fraserprice/DeepSeek-4.1-Flash-1017-BRRR (Apache-2.0; LICENSE-brrr + NOTICE-brrr kept): skinny FP8 GEMM for
# <=8 decode rows, radix indexer top-k, phased PCIe all-reduce/all-gather (rebuilt for 3 ranks) for the 16 KB..1 MB
# messages b12x's one-shot declines, plus BRRR's renderer fix (tool-result-only messages don't start a new turn;
# parallel tool results sorted by call order). Sparse decode attention and the fused decoder are NOT included
# (24-head tiling / CSF scales; see README). Measured: c1 83 -> 86.5 steps/s, coding mean 336-341 -> 365 tok/s.
# Build from the release dir (kernels compile for sm_120a during the build, ~3 min):
#   docker build -f patch-brrr/image/Dockerfile -t azallaza/ds41-kraken-tp3:20261019.6 .
FROM azallaza/ds41-kraken-tp3:20261019.5
ARG SP=/opt/venv/lib/python3.12/site-packages
COPY patch-brrr/brrr     $SP/brrr
COPY patch-brrr/brrr.pth $SP/brrr.pth
COPY patch-brrr/LICENSE-brrr patch-brrr/NOTICE-brrr /opt/brrr/
COPY patch-brrr/vllm-patched/models/deepseek_v4_1/attention.py   $SP/vllm/models/deepseek_v4_1/attention.py
COPY patch-brrr/vllm-patched/renderers/deepseek_v4.py            $SP/vllm/renderers/deepseek_v4.py
COPY patch-brrr/vllm-patched/tokenizers/deepseek_v41_encoding.py $SP/vllm/tokenizers/deepseek_v41_encoding.py
ENV TORCH_EXTENSIONS_DIR=/opt/brrr-ext
# prebuild the CUDA extensions (no GPU needed to compile; sm_120a)
RUN mkdir -p /opt/brrr-ext && cd /tmp && MAX_JOBS=16 /opt/venv/bin/python -c "from brrr import cuext; [cuext.extension(n) for n in ('skinny','candidates','phased_all_reduce','sparse_decode')]" \
    && find $SP/vllm/models/deepseek_v4_1/__pycache__ $SP/vllm/renderers/__pycache__ $SP/vllm/tokenizers/__pycache__ -name "*.pyc" -delete 2>/dev/null || true
LABEL org.opencontainers.image.description="DS41 kraken TP3 20261019.5 + BRRR TP3 kernels (skinny GEMM, radix top-k, phased all-reduce) + renderer tool-loop fix"
