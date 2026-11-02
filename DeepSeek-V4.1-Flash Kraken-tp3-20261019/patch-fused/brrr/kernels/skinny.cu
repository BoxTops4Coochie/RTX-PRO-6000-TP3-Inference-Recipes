#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "fp8_rows.cuh"
#include "host.cuh"

namespace {

constexpr int MAXM = 8;
constexpr int WARPS = 8;
constexpr int THREADS = WARPS * 32;

struct Args {
  const void* x;
  long x_ld;
  const uint8_t* xs;
  const uint8_t* w;
  const uint8_t* ws;
  __nv_bfloat16* y;
  long y_ld;
  int m;
};

template <int K>
__device__ __forceinline__ void stage_bf16(const Args& a, uint8_t* xq, uint8_t* xs) {
  constexpr int BLOCKS = K / 32, VECS = MAXM * K / 8, PER = VECS / THREADS;
  static_assert(VECS % THREADS == 0);
  uint4 raw[PER];
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    const int v = threadIdx.x + i * THREADS, tok = v / (K / 8), col = (v % (K / 8)) * 8;
    raw[i] = tok < a.m ? __ldg(reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(a.x) + tok * a.x_ld + col)) : make_uint4(0, 0, 0, 0);
  }
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    const int v = threadIdx.x + i * THREADS, tok = v / (K / 8), col = (v % (K / 8)) * 8;
    const unsigned u[4] = {raw[i].x, raw[i].y, raw[i].z, raw[i].w};
    float f[8];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const float2 h = unpack_bf16(u[q]);
      f[2 * q] = h.x, f[2 * q + 1] = h.y;
    }
    const int e = block_exp(f, MX_AMAX_FLOOR);
    *reinterpret_cast<uint2*>(xq + tok * (K + 16) + col) = quant8(f, e);
    if ((v & 3) == 0) xs[tok * BLOCKS + col / 32] = uint8_t(e + 127);
  }
}

template <int K>
__device__ __forceinline__ void stage_fp8(const Args& a, uint8_t* xq, uint8_t* xs) {
  constexpr int BLOCKS = K / 32, VECS = MAXM * K / 16;
  for (int v = threadIdx.x; v < VECS; v += THREADS) {
    const int tok = v / (K / 16), col = (v % (K / 16)) * 16;
    *reinterpret_cast<uint4*>(xq + tok * (K + 16) + col) =
        tok < a.m ? __ldg(reinterpret_cast<const uint4*>(static_cast<const uint8_t*>(a.x) + tok * a.x_ld + col)) : make_uint4(0, 0, 0, 0);
  }
  for (int i = threadIdx.x; i < MAXM * BLOCKS; i += THREADS) {
    const int tok = i / BLOCKS, grp = i % BLOCKS;
    xs[i] = tok < a.m ? a.xs[scale_index(grp, tok)] : uint8_t(127);
  }
}

template <int K, int T, bool PREQ>
__global__ void __launch_bounds__(THREADS) skinny_kernel(Args a) {
  constexpr int BLOCKS = K / 32, PAIRS = BLOCKS / 2, STRIDE = WARPS / T, UNITS = PAIRS / STRIDE;
  static_assert(WARPS % T == 0 && PAIRS % STRIDE == 0);
  extern __shared__ __align__(16) uint8_t smem[];
  uint8_t* xq = smem;
  uint8_t* xs = smem + MAXM * (K + 16);
  float* red = reinterpret_cast<float*>(xs + MAXM * BLOCKS);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int tile = blockIdx.x * T + warp % T;
  Rows<K, UNITS, STRIDE> rows;
  rows.load(a.w, a.ws, tile, warp / T, lane);
  if constexpr (PREQ) stage_fp8<K>(a, xq, xs);
  else stage_bf16<K>(a, xq, xs);
  __syncthreads();
  rows.ready(lane);
  float acc[4] = {};
  rows.template accumulate<MAXM>(xq, K + 16, xs, BLOCKS, warp / T, 0, lane, acc);
  *reinterpret_cast<float2*>(red + (warp * MAXM + g) * 8 + 2 * t) = make_float2(acc[0], acc[1]);
  __syncthreads();
  for (int idx = threadIdx.x; idx < T * MAXM * 8; idx += THREADS) {
    const int i = idx / (MAXM * 8), tok = idx / 8 % MAXM, row = idx % 8;
    float v = 0.0f;
#pragma unroll
    for (int w = i; w < WARPS; w += T) v += red[(w * MAXM + tok) * 8 + row];
    if (tok < a.m) a.y[tok * a.y_ld + (blockIdx.x * T + i) * 8 + row] = __float2bfloat16(v);
  }
}

template <int K>
constexpr int smem_bytes() {
  return MAXM * (K + 16) + MAXM * (K / 32) + WARPS * MAXM * 8 * 4;
}

template <int K, int T, bool PREQ>
void launch(const Args& a, int n) {
  opt_in_smem<skinny_kernel<K, T, PREQ>>(smem_bytes<K>());
  skinny_kernel<K, T, PREQ><<<n / (8 * T), THREADS, smem_bytes<K>(), c10::cuda::getCurrentCUDAStream()>>>(a);
}

template <bool PREQ>
void dispatch(const Args& a, int k, int n) {
  if (k == 5120 && n % 8 == 0) return launch<5120, 1, PREQ>(a, n);
  if (k == 1280 && n % 32 == 0 && n >= 8192) return launch<1280, 4, PREQ>(a, n);
  if (k == 1280 && n % 16 == 0) return launch<1280, 2, PREQ>(a, n);
  TORCH_CHECK(false, "skinny gemm: unsupported shape k=", k, " n=", n);
}

__global__ void __launch_bounds__(THREADS) split_norm_kernel(const __nv_bfloat16* x, int width, int split, const __nv_bfloat16* wa, const __nv_bfloat16* wb,
                                                             __nv_bfloat16* a, __nv_bfloat16* b, uint8_t* aq, uint8_t* as, float eps) {
  __shared__ float partial[WARPS];
  const int row = blockIdx.x >> 1, part = blockIdx.x & 1, lo = part ? split : 0, n = part ? width - split : split;
  const __nv_bfloat16* src = x + long(row) * width + lo;
  float sq = 0.0f;
  for (int i = threadIdx.x; i < n; i += THREADS) {
    const float v = __bfloat162float(src[i]);
    sq += v * v;
  }
  sq = warp_sum(sq);
  if ((threadIdx.x & 31) == 0) partial[threadIdx.x >> 5] = sq;
  __syncthreads();
  float total = 0.0f;
#pragma unroll
  for (int w = 0; w < WARPS; ++w) total += partial[w];
  const float scale = rsqrtf(total / n + eps);
  const __nv_bfloat16* weight = part ? wb : wa;
  __nv_bfloat16* dst = (part ? b : a) + long(row) * n;
  for (int i = threadIdx.x; i < n; i += THREADS) {
    const __nv_bfloat16 y = __float2bfloat16(__bfloat162float(src[i]) * scale * __bfloat162float(weight[i]));
    dst[i] = y;
    if (aq && !part) {
      const float v = __bfloat162float(y);
      const int e = quant_exp(warp_max(fabsf(v)), MX_AMAX_FLOOR);
      aq[long(row) * n + i] = quant_byte(v, e);
      if ((threadIdx.x & 31) == 0) as[scale_index(i / 32, row)] = uint8_t(e + 127);
    }
  }
}

}

std::vector<torch::Tensor> split_norm(torch::Tensor x, int64_t split, torch::Tensor wa, torch::Tensor wb, double eps) {
  TORCH_CHECK(x.dim() == 2 && x.is_contiguous() && x.scalar_type() == at::kBFloat16 && split > 0 && split < x.size(1));
  TORCH_CHECK(wa.is_contiguous() && wb.is_contiguous() && wa.numel() == split && wb.numel() == x.size(1) - split);
  TORCH_CHECK(wa.scalar_type() == at::kBFloat16 && wb.scalar_type() == at::kBFloat16);
  const c10::cuda::CUDAGuard guard(x.device());
  auto a = torch::empty({x.size(0), split}, x.options());
  auto b = torch::empty({x.size(0), x.size(1) - split}, x.options());
  const bool quantize = x.size(0) <= MAXM && split % THREADS == 0;
  const auto u8 = x.options().dtype(at::kByte);
  auto aq = torch::empty({quantize ? x.size(0) : 0, split}, u8);
  auto as = torch::empty({quantize ? (split / 32 + 3) / 4 * 512 : 0}, u8);
  if (x.size(0))
    split_norm_kernel<<<x.size(0) * 2, THREADS, 0, c10::cuda::getCurrentCUDAStream()>>>(
        data_as<const __nv_bfloat16>(x), int(x.size(1)), int(split), data_as<const __nv_bfloat16>(wa), data_as<const __nv_bfloat16>(wb),
        data_as<__nv_bfloat16>(a), data_as<__nv_bfloat16>(b), quantize ? aq.data_ptr<uint8_t>() : nullptr, quantize ? as.data_ptr<uint8_t>() : nullptr,
        float(eps));
  return {a, b, aq, as};
}

torch::Tensor gemm(torch::Tensor x, std::optional<torch::Tensor> xs, torch::Tensor w, torch::Tensor ws) {
  const long m = x.size(0), k = x.size(1), n = w.size(0);
  TORCH_CHECK(x.dim() == 2 && m <= MAXM && x.stride(1) == 1 && w.is_contiguous() && w.element_size() == 1 && w.size(1) == k);
  TORCH_CHECK(ws.is_contiguous() && ws.element_size() == 1 && ws.numel() == n * k / 32);
  TORCH_CHECK(xs.has_value() == (x.element_size() == 1) && (xs.has_value() || x.scalar_type() == at::kBFloat16));
  const c10::cuda::CUDAGuard guard(x.device());
  auto y = torch::empty({m, n}, x.options().dtype(at::kBFloat16));
  const Args a{x.data_ptr(), x.stride(0), xs ? data_as<const uint8_t>(*xs) : nullptr, data_as<const uint8_t>(w), data_as<const uint8_t>(ws), data_as<__nv_bfloat16>(y), y.stride(0), int(m)};
  if (xs) dispatch<true>(a, int(k), int(n));
  else dispatch<false>(a, int(k), int(n));
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.attr("MAXM") = MAXM;
  m.def("gemm", &gemm);
  m.def("split_norm", &split_norm);
}
