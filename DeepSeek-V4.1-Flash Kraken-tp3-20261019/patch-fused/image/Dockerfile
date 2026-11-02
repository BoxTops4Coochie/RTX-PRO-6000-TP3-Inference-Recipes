# azallaza/ds41-kraken-tp3:20261019.9 = 20261019.7 + BRRR's fused decoder layer ported to TP3, on all 40 layers, reading
# vLLM's own dense weights (no staged copies: ~1.6 GiB/GPU back to expert promotion; .8 = with staged copies).
# From fraserprice/DeepSeek-4.1-Flash-1017-BRRR (Apache-2.0; LICENSE-brrr + NOTICE-brrr kept). TP3 port: 3 ranks,
# 72 padded heads (24/rank) in the fused attention prologue, padded o-proj (OA 3072), INTER 768 on b12x's natural W4A8
# layout, routed spans of 16 rows to fit sm_120 shared memory, and an in-kernel reader for b12x's compressed (CSF)
# MXFP4 expert scales, so the fused layers keep CSF and the .7 parking layout (20-33). Default: DS41_FUSED_LAYERS=40.
# DS41_FUSED_LAYERS=0 is the plain .7 path (no fused engine is built). Build from the release dir (~5 min):
#   docker build -f patch-fused/image/Dockerfile -t azallaza/ds41-kraken-tp3:20261019.9 .
FROM azallaza/ds41-kraken-tp3:20261019.7
ARG SP=/opt/venv/lib/python3.12/site-packages
COPY patch-fused/brrr $SP/brrr
COPY patch-fused/LICENSE-brrr patch-fused/NOTICE-brrr /opt/brrr/
COPY patch-fused/vllm-patched/models/deepseek_v4_1/nvidia/model.py   $SP/vllm/models/deepseek_v4_1/nvidia/model.py
COPY patch-fused/vllm-patched/model_executor/layers/fused_moe/b12x.py $SP/vllm/model_executor/layers/fused_moe/b12x.py
ENV DS41_FUSED_LAYERS=40
RUN rm -rf /opt/brrr-ext && mkdir -p /opt/brrr-ext && cd /tmp && MAX_JOBS=16 /opt/venv/bin/python -c "from brrr import cuext; [cuext.extension(n) for n in ('skinny','candidates','phased_all_reduce','sparse_decode','decoder')]" \
    && find /opt/brrr-ext -name lock -delete \
    && find $SP/vllm/models/deepseek_v4_1 $SP/vllm/model_executor/layers/fused_moe -name "*.pyc" -delete 2>/dev/null || true
LABEL org.opencontainers.image.description="DS41 kraken TP3 20261019.7 + BRRR fused decoder layer at TP3 (all 40 layers, in-kernel CSF scales, native dense weights)"
