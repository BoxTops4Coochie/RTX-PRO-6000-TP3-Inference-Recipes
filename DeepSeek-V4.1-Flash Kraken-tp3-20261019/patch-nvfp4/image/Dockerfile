# azallaza/ds41-kraken-tp3:20261019.4 = 20261019.3 + MXFP4-packed Engram tables + NVFP4 routed-expert
# support (online CSF scales, in-place VMM promotion), baked instead of mounted. The patched files are
# inert for the stock checkpoint (same code paths when engram_embed_quant is absent and experts are MXFP4).
# Sources: ../work/*.py in the release repo (patch-nvfp4/). Build from the release dir:
#   docker build -f patch-nvfp4/image/Dockerfile -t azallaza/ds41-kraken-tp3:20261019.4 .
FROM azallaza/ds41-kraken-tp3:20261019.3
ARG SP=/opt/venv/lib/python3.12/site-packages
COPY patch-nvfp4/work/engram_native__kernels.py     $SP/flashinfer/experimental/b12x/sequence/engram/_kernels.py
COPY patch-nvfp4/work/engram_native__impl.py        $SP/flashinfer/experimental/b12x/sequence/engram/_impl.py
COPY patch-nvfp4/work/engram_native__disk.py        $SP/flashinfer/experimental/b12x/sequence/engram/_disk.py
COPY patch-nvfp4/work/engram_native__preparation.py $SP/flashinfer/experimental/b12x/sequence/engram/_preparation.py
COPY patch-nvfp4/work/engram_native__tuning.py      $SP/flashinfer/experimental/b12x/sequence/engram/_tuning.py
COPY patch-nvfp4/work/engram.py                     $SP/vllm/models/deepseek_v4_1/common/engram.py
COPY patch-nvfp4/work/quant_config.py               $SP/vllm/models/deepseek_v4_1/quant_config.py
COPY patch-nvfp4/work/b12x.py                       $SP/vllm/model_executor/layers/fused_moe/b12x.py
# drop stale bytecode so the patched sources are what runs
RUN find $SP/flashinfer/experimental/b12x/sequence/engram/__pycache__ $SP/vllm/models/deepseek_v4_1/__pycache__ \
         $SP/vllm/models/deepseek_v4_1/common/__pycache__ $SP/vllm/model_executor/layers/fused_moe/__pycache__ \
         -name "*.pyc" -delete 2>/dev/null || true
LABEL org.opencontainers.image.description="DS41 kraken TP3 20261019.3 + MXFP4 Engram tables + NVFP4 experts (online CSF)"
