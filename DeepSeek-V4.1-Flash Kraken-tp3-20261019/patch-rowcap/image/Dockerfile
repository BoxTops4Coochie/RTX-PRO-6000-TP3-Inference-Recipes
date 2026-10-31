# azallaza/ds41-kraken-tp3:20261019.5 = 20261019.4 + "cap-only" adaptive verification for DSpark.
# One patched file (patch-rowcap/adaptive_verification.py): with DS41_VERIFY_ROW_CAP=N (and the launcher's
# adaptive-verification: true), every draft is verified while the decoding requests' total target rows fit under N;
# above N drafts are trimmed, confidence-ranked across requests. DS41_VERIFY_ROW_CAP unset/0 = stock adaptive.
# Build from the release dir:
#   docker build -f patch-rowcap/image/Dockerfile -t azallaza/ds41-kraken-tp3:20261019.5 .
FROM azallaza/ds41-kraken-tp3:20261019.4
ARG SP=/opt/venv/lib/python3.12/site-packages
COPY patch-rowcap/adaptive_verification.py $SP/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py
RUN find $SP/vllm/v1/worker/gpu/spec_decode/__pycache__ -name "adaptive_verification*.pyc" -delete 2>/dev/null || true
LABEL org.opencontainers.image.description="DS41 kraken TP3 20261019.4 + cap-only adaptive verification (DS41_VERIFY_ROW_CAP)"
