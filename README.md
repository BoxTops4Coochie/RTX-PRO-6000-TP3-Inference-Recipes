#  Native TP3

Inference recipes for 3x RTX PRO 6000 Blackwell (TP3) with vLLM. Each folder is one build: image,
compose file, measurements and the source changes.

| Folder | Model | Image |
| --- | --- | --- |
| [DeepSeek-V4.1-Flash Kraken-tp3-20261019](DeepSeek-V4.1-Flash%20Kraken-tp3-20261019/README.md) | DeepSeek-V4.1-Flash-1017 (original weights + NVIDIA 4-bit Engram tables), 1M context, 8 slots | `azallaza/ds41-kraken-tp3:20261019.7` |
| [GLM-5.3-Flash Kraken-tp3-CSF](GLM-5.3-Flash%20Kraken-tp3-CSF/README.md) | GLM-5.3-Flash NVFP4-MXFP8 CSF-QAD | `azallaza/glm53-kraken-tp3:20261015` |
| [GLM-5.3-Flash Kraken-tp3-20260927](GLM-5.3-Flash%20Kraken-tp3-20260927/README.md) | GLM-5.3-Flash | `azallaza/glm53-kraken-tp3:20260927` |
| [GLM-5.3-Flash Kraken-tp3](GLM-5.3-Flash%20Kraken-tp3/glm-5.3-flash-tp3-kraken.md) | GLM-5.3-Flash | see folder |
| [GLM-5.3-Flash Kraken tp3 experimental](GLM-5.3-Flash%20Kraken%20tp3%20experimental/glm-5.3-flash-tp3-kraken.md) | GLM-5.3-Flash (experiments) | see folder |
| GLM-5.3-Flash Jovian r34, r34-v2, r34-v3 | GLM-5.3-Flash on the Jovian image | see folders |
