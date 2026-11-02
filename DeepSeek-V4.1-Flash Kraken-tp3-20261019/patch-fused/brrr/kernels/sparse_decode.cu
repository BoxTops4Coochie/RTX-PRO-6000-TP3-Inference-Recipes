#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "common.cuh"
#include "host.cuh"

namespace {

constexpr int H = 24;  // TP3: 72 padded heads / 3
constexpr int D = 512;
constexpr int CL = 8;
constexpr int MAXK = 64;  // TP3: H=24 raised Smem past the sm_120 99 KB opt-in limit at MAXK=80; 64 keeps swa-only (192) decode eligible
constexpr int WARPS = 8;
constexpr int THREADS = WARPS * 32;
constexpr int PS = MAXK + 8;
constexpr int SWA_BYTES = 528;
constexpr int IDX_BYTES = 288;
constexpr float LOG2E = 1.4426950408889634f;

struct Args {
  const __nv_bfloat16* q;
  __nv_bfloat16* out;
  const uint8_t* swa;
  long swa_page_stride;
  int swa_page;
  const long* positions;
  const int* reqs;
  const int* swa_table;
  long swa_table_stride;
  const int* starts;
  const long* request_positions;
  const long* replay;
  int window;
  int draft;
  int swa_width;
  const uint8_t* idx;
  long idx_page_stride;
  int idx_page;
  long idx_pages;
  const int* idx_logical;
  const int* visible;
  const int* idx_reqs;
  const int* idx_table;
  long idx_table_stride;
  int idx_table_width;
  int idx_width;
  int per_cta;
  const float* sink;
  float scale;
  float* partial;
  float2* stats;
};

struct Smem {
  __nv_bfloat16 q[H * D];
  __nv_bfloat16 k[MAXK * D];
  float red[WARPS][H];
  union {
    float extra[2][H][8];
    float w[CL][H];
  };
  float m[H];
  float l[H];
  int slots[MAXK];
  long offsets[MAXK];
  long window_start;
  int window_length;
  int window_req;
  int top_length;
  int idx_req;
};

__device__ __forceinline__ int swz(int row, int col) { return row * D + ((((col >> 3) ^ (row & 7))) << 3) + (col & 7); }

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}

__device__ __forceinline__ void ldsm_x2(uint32_t (&r)[2], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(smem_u32(p)));
}

__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}

__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}

__device__ __forceinline__ float2 half2_float2(uint32_t h) { return __half22float2(*reinterpret_cast<const __half2*>(&h)); }

__device__ __forceinline__ float2 e2m1x2(uint8_t v) {
  uint32_t h;
  asm("{\n .reg .b8 t;\n mov.b16 {t, _}, %1;\n cvt.rn.f16x2.e2m1x2 %0, t;\n}" : "=r"(h) : "h"(static_cast<uint16_t>(v)));
  return half2_float2(h);
}

__device__ __forceinline__ void row_window(const Args& a, Smem& sm, int row) {
  const int req = a.reqs[row];
  const long pos = a.positions[row];
  const bool valid = req >= 0 && pos >= 0;
  long start = 0, length = 0;
  if (valid) {
    if (a.draft) {
      const long prefix = a.request_positions[req];
      start = max(prefix - a.window, 0L);
      length = prefix + a.starts[req + 1] - a.starts[req] - start;
    } else {
      start = max(pos + 1 - a.window, 0L);
      if (a.replay) start = max(start, a.replay[req]);
      length = pos + 1 - start;
    }
  }
  sm.window_req = req;
  sm.window_start = start;
  sm.window_length = valid ? int(min(max(length, 0L), long(a.swa_width))) : 0;
  sm.top_length = valid && a.idx ? min(max(a.visible[row], 0), 512) : 0;
  sm.idx_req = a.idx ? a.idx_reqs[row] : -1;
}

__device__ __forceinline__ int slot_of(const Args& a, const Smem& sm, int row, int k) {
  if (k < a.swa_width) {
    if (k >= sm.window_length) return -1;
    const long logical = sm.window_start + k;
    const long page = a.swa_table[long(sm.window_req) * a.swa_table_stride + logical / a.swa_page];
    return page > 0 ? int(page * a.swa_page + logical % a.swa_page) : -1;
  }
  k -= a.swa_width;
  if (k >= a.idx_width || k >= sm.top_length || sm.idx_req < 0) return -1;
  const long logical = a.idx_logical[long(row) * a.idx_width + k];
  if (logical < 0) return -1;
  const long page = logical / a.idx_page;
  if (page >= a.idx_table_width) return -1;
  const long pid = a.idx_table[long(sm.idx_req) * a.idx_table_stride + page];
  if (pid <= 0 || pid >= a.idx_pages) return -1;
  return int(pid * a.idx_page + logical % a.idx_page);
}

__device__ __forceinline__ void decode_keys(const Args& a, Smem& sm, int first, int warp, int lane) {
  constexpr int PER = MAXK / WARPS;
  const int per = a.per_cta / WARPS;
  uint4 raw[PER];
  uint8_t scale[PER];
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    const int key = warp * per + i, slot = i < per ? sm.slots[key] : -1;
    const bool swa = first + key < a.swa_width;
    const uint8_t* rec = (swa ? a.swa : a.idx) + sm.offsets[key];
    raw[i] = make_uint4(0, 0, 0, 0);
    scale[i] = 0;
    if (slot >= 0) {
      if (swa) {
        raw[i] = __ldg(reinterpret_cast<const uint4*>(rec + 16 * lane));
      } else {
        const uint2 r = __ldg(reinterpret_cast<const uint2*>(rec + 8 * lane));
        raw[i] = make_uint4(r.x, r.y, 0, 0);
      }
      scale[i] = __ldg(rec + (swa ? D + lane / 2 : D / 2 + lane));
    }
  }
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    if (i >= per) break;
    const int key = warp * per + i;
    uint32_t o[8];
    if (first + key < a.swa_width) {
      const float s = e8m0(scale[i]);
      const uint16_t* v = reinterpret_cast<const uint16_t*>(&raw[i]);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float2 f = half2_float2(e4m3x2_f16x2(v[j]));
        o[j] = pack_bf16(f.x * s, f.y * s);
      }
    } else {
      const float s = half2_float2(e4m3x2_f16x2(scale[i])).x;
      const uint8_t* v = reinterpret_cast<const uint8_t*>(&raw[i]);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float2 f = e2m1x2(v[j]);
        o[j] = pack_bf16(f.x * s, f.y * s);
      }
    }
    *reinterpret_cast<uint4*>(sm.k + swz(key, lane * 16)) = make_uint4(o[0], o[1], o[2], o[3]);
    *reinterpret_cast<uint4*>(sm.k + swz(key, lane * 16 + 8)) = make_uint4(o[4], o[5], o[6], o[7]);
  }
}

template <class Op>
__device__ __forceinline__ float across_warps(Smem& sm, int warp, int lane, int tid, float lo, float hi, float init, Op op) {
  const int g = lane / 4;
  if (lane % 4 == 0) sm.red[warp][g] = lo, sm.red[warp][g + 8] = hi;
  __syncthreads();
  float v = init;
  if (tid < H)
#pragma unroll
    for (int w = 0; w < WARPS; ++w) v = op(v, sm.red[w][tid]);
  return v;
}

__global__ void __cluster_dims__(CL, 1, 1) __launch_bounds__(THREADS, 1) sparse_decode(Args a) {
  extern __shared__ __align__(128) uint8_t raw_smem[];
  Smem& sm = *reinterpret_cast<Smem*>(raw_smem);
  const int rank = blockIdx.x, row = blockIdx.y, tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int keys = a.per_cta, first = rank * keys, tiles = keys / 8;
  pdl_launch_dependents();

  if (tid == 0) row_window(a, sm, row);
  pdl_wait();
  for (int i = tid; i < H * D / 8; i += THREADS) {
    const int h = i / (D / 8), c = (i % (D / 8)) * 8;
    cp_async16(sm.q + swz(h, c), a.q + (long(row) * H + h) * D + c);
  }
  cp_async_commit();
  __syncthreads();
  if (tid < keys) {
    const int slot = slot_of(a, sm, row, first + tid);
    const bool swa = first + tid < a.swa_width;
    const int page = swa ? a.swa_page : a.idx_page;
    sm.slots[tid] = slot;
    sm.offsets[tid] = slot < 0 ? 0 : long(slot / page) * (swa ? a.swa_page_stride : a.idx_page_stride) + long(slot % page) * (swa ? SWA_BYTES : IDX_BYTES);
  }
  reinterpret_cast<float*>(sm.extra)[tid] = 0.0f;
  __syncthreads();
  decode_keys(a, sm, first, warp, lane);
  cp_async_wait_all();
  __syncthreads();

  float s[2][4] = {};
  const auto score = [&](float (&acc)[4], int tile, int k0, int k1) {
    float odd[4] = {};
#pragma unroll 4
    for (int kk = k0; kk < k1; kk += 2) {
      uint32_t qa[4], kb[2], qb[4], kc[2];
      ldsm_x4(qa, sm.q + swz(lane % 16, kk * 16 + (lane / 16) * 8));
      ldsm_x2(kb, sm.k + swz(tile * 8 + lane % 8, kk * 16 + ((lane / 8) % 2) * 8));
      ldsm_x4(qb, sm.q + swz(lane % 16, (kk + 1) * 16 + (lane / 16) * 8));
      ldsm_x2(kc, sm.k + swz(tile * 8 + lane % 8, (kk + 1) * 16 + ((lane / 8) % 2) * 8));
      mma_bf16(acc, qa[0], qa[1], qa[2], qa[3], kb[0], kb[1]);
      mma_bf16(odd, qb[0], qb[1], qb[2], qb[3], kc[0], kc[1]);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[i] += odd[i];
  };
  if (warp < tiles) score(s[0], warp, 0, D / 16);
  const int spill = tiles - WARPS, share = WARPS / 2, quarter = D / 16 / share;
  const int col = 2 * (lane % 4);
  if (spill > 0 && WARPS + warp / share < tiles) {
    float part[4] = {};
    const int t = warp / share;
    score(part, WARPS + t, (warp % share) * quarter, (warp % share + 1) * quarter);
    atomicAdd(&sm.extra[t][lane / 4][col], part[0]);
    atomicAdd(&sm.extra[t][lane / 4][col + 1], part[1]);
    atomicAdd(&sm.extra[t][lane / 4 + 8][col], part[2]);
    atomicAdd(&sm.extra[t][lane / 4 + 8][col + 1], part[3]);
  }
  __syncthreads();
  if (warp < spill) {
    s[1][0] = sm.extra[warp][lane / 4][col];
    s[1][1] = sm.extra[warp][lane / 4][col + 1];
    s[1][2] = sm.extra[warp][lane / 4 + 8][col];
    s[1][3] = sm.extra[warp][lane / 4 + 8][col + 1];
  }
  const int g = lane / 4;
  const float scale = a.scale * LOG2E;
  float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
  for (int t = 0; t < 2; ++t) {
    const int tile = warp + t * WARPS, key = tile * 8 + 2 * (lane % 4);
    const bool live = tile < tiles && (t == 0 || warp < spill);
    const bool v0 = live && sm.slots[key] >= 0, v1 = live && sm.slots[key + 1] >= 0;
    s[t][0] = v0 ? s[t][0] * scale : -INFINITY;
    s[t][1] = v1 ? s[t][1] * scale : -INFINITY;
    s[t][2] = v0 ? s[t][2] * scale : -INFINITY;
    s[t][3] = v1 ? s[t][3] * scale : -INFINITY;
    mx0 = fmaxf(mx0, fmaxf(s[t][0], s[t][1]));
    mx1 = fmaxf(mx1, fmaxf(s[t][2], s[t][3]));
  }
  const float head_max = across_warps(sm, warp, lane, tid, quad_max(mx0), quad_max(mx1), -INFINITY, [](float a, float b) { return fmaxf(a, b); });
  if (tid < H) sm.m[tid] = head_max;
  __syncthreads();
  __nv_bfloat16* p = sm.q;
  const float m0 = sm.m[g], m1 = sm.m[g + 8];
  float l0 = 0.0f, l1 = 0.0f;
#pragma unroll
  for (int t = 0; t < 2; ++t) {
    const int tile = warp + t * WARPS, key = tile * 8 + 2 * (lane % 4);
    if (tile >= tiles) break;
    const auto e = [](float v, float m) { return v == -INFINITY ? 0.0f : exp2f(v - m); };
    const uint32_t pb0 = pack_bf16(e(s[t][0], m0), e(s[t][1], m0)), pb1 = pack_bf16(e(s[t][2], m1), e(s[t][3], m1));
    *reinterpret_cast<uint32_t*>(p + g * PS + key) = pb0;
    *reinterpret_cast<uint32_t*>(p + (g + 8) * PS + key) = pb1;
    const __nv_bfloat162 r0 = *reinterpret_cast<const __nv_bfloat162*>(&pb0), r1 = *reinterpret_cast<const __nv_bfloat162*>(&pb1);
    l0 += __low2float(r0) + __high2float(r0);
    l1 += __low2float(r1) + __high2float(r1);
  }
  __syncthreads();
  const float head_sum = across_warps(sm, warp, lane, tid, quad_sum(l0), quad_sum(l1), 0.0f, [](float a, float b) { return a + b; });
  if (tid < H) sm.l[tid] = head_sum;

  float o[8][4] = {};
  for (int ks = 0; ks < keys / 16; ++ks) {
    uint32_t pa[4];
    ldsm_x4(pa, p + (lane % 16) * PS + ks * 16 + (lane / 16) * 8);
#pragma unroll
    for (int nt = 0; nt < 8; nt += 2) {
      uint32_t vb[4];
      ldsm_x4_t(vb, sm.k + swz(ks * 16 + (lane % 8) + ((lane / 8) % 2) * 8, warp * 64 + nt * 8 + (lane / 16) * 8));
      mma_bf16(o[nt], pa[0], pa[1], pa[2], pa[3], vb[0], vb[1]);
      mma_bf16(o[nt + 1], pa[0], pa[1], pa[2], pa[3], vb[2], vb[3]);
    }
  }
  float* po = a.partial + (long(row) * CL + rank) * H * D;
#pragma unroll
  for (int nt = 0; nt < 8; ++nt) {
    const int d = warp * 64 + nt * 8 + 2 * (lane % 4);
    *reinterpret_cast<float2*>(po + g * D + d) = make_float2(o[nt][0], o[nt][1]);
    *reinterpret_cast<float2*>(po + (g + 8) * D + d) = make_float2(o[nt][2], o[nt][3]);
  }
  if (tid < H) a.stats[(long(row) * CL + rank) * H + tid] = make_float2(sm.m[tid], sm.l[tid]);
  cluster_sync();

  if (tid < H) {
    float2 ml[CL];
#pragma unroll
    for (int c = 0; c < CL; ++c) ml[c] = __ldcg(a.stats + (long(row) * CL + c) * H + tid);
    float ms[CL], ls[CL];
    float mx = a.sink[tid] * LOG2E;
    const float sink = mx;
#pragma unroll
    for (int c = 0; c < CL; ++c) {
      ms[c] = ml[c].x;
      ls[c] = ml[c].y;
      mx = fmaxf(mx, ms[c]);
    }
    float total = mx == -INFINITY ? 0.0f : exp2f(sink - mx);
#pragma unroll
    for (int c = 0; c < CL; ++c) {
      const float w = ms[c] == -INFINITY ? 0.0f : exp2f(ms[c] - mx);
      sm.w[c][tid] = w;
      total += w * ls[c];
    }
    sm.red[0][tid] = total > 0.0f ? 1.0f / total : 0.0f;
  }
  __syncthreads();
  constexpr int SLICE = D / CL / 4;
  const int h = tid / SLICE, d4 = rank * SLICE + tid % SLICE;
  float4 v[CL];
#pragma unroll
  for (int c = 0; c < CL; ++c) v[c] = __ldcg(reinterpret_cast<const float4*>(a.partial + ((long(row) * CL + c) * H + h) * D) + d4);
  float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
  for (int c = 0; c < CL; ++c) {
    const float w = sm.w[c][h];
    acc.x += w * v[c].x, acc.y += w * v[c].y, acc.z += w * v[c].z, acc.w += w * v[c].w;
  }
  const float inv = sm.red[0][h];
  *reinterpret_cast<uint2*>(a.out + (long(row) * H + h) * D + d4 * 4) = make_uint2(pack_bf16(acc.x * inv, acc.y * inv), pack_bf16(acc.z * inv, acc.w * inv));
  cluster_sync();
}

}

void decode(torch::Tensor q, torch::Tensor out, torch::Tensor swa_cache, int64_t swa_page, torch::Tensor positions, torch::Tensor reqs, torch::Tensor swa_table,
            torch::Tensor starts, torch::Tensor request_positions, c10::optional<torch::Tensor> replay, int64_t window, int64_t swa_width, bool draft,
            c10::optional<torch::Tensor> idx_cache, int64_t idx_page, c10::optional<torch::Tensor> idx_logical, c10::optional<torch::Tensor> visible,
            c10::optional<torch::Tensor> idx_reqs, c10::optional<torch::Tensor> idx_table, torch::Tensor sink, double sm_scale, torch::Tensor partial,
            torch::Tensor stats) {
  const int rows = q.size(0);
  TORCH_CHECK(q.is_contiguous() && out.is_contiguous() && q.size(1) == H && q.size(2) == D && q.scalar_type() == at::kBFloat16 && sink.scalar_type() == at::kFloat);
  TORCH_CHECK(swa_table.stride(1) == 1 && positions.numel() >= rows && reqs.numel() >= rows);
  Args a{};
  a.q = data_as<const __nv_bfloat16>(q);
  a.out = data_as<__nv_bfloat16>(out);
  a.swa = swa_cache.data_ptr<uint8_t>();
  a.swa_page_stride = swa_cache.stride(0);
  a.swa_page = swa_page;
  a.positions = typed<long>(positions, at::kLong);
  a.reqs = typed<int>(reqs, at::kInt);
  a.swa_table = typed<int>(swa_table, at::kInt);
  a.swa_table_stride = swa_table.stride(0);
  a.starts = typed<int>(starts, at::kInt);
  a.request_positions = typed<long>(request_positions, at::kLong);
  a.replay = replay.has_value() ? typed<long>(*replay, at::kLong) : nullptr;
  a.window = window;
  a.draft = draft;
  a.swa_width = swa_width;
  if (idx_cache.has_value()) {
    TORCH_CHECK(idx_logical->is_contiguous() && idx_table->stride(1) == 1);
    a.idx = idx_cache->data_ptr<uint8_t>();
    a.idx_page_stride = idx_cache->stride(0);
    a.idx_page = idx_page;
    a.idx_pages = idx_cache->size(0);
    a.idx_logical = typed<int>(*idx_logical, at::kInt);
    a.visible = typed<int>(*visible, at::kInt);
    a.idx_reqs = typed<int>(*idx_reqs, at::kInt);
    a.idx_table = typed<int>(*idx_table, at::kInt);
    a.idx_table_stride = idx_table->stride(0);
    a.idx_table_width = idx_table->size(1);
    a.idx_width = idx_logical->size(1);
  }
  const int total = a.swa_width + a.idx_width;
  TORCH_CHECK(total % (CL * 16) == 0 && total / CL <= MAXK);
  a.per_cta = total / CL;
  TORCH_CHECK(partial.numel() >= long(rows) * CL * H * D && stats.numel() >= long(rows) * CL * H * 2);
  a.partial = partial.data_ptr<float>();
  a.stats = reinterpret_cast<float2*>(stats.data_ptr<float>());
  a.sink = sink.data_ptr<float>();
  a.scale = float(sm_scale);
  const c10::cuda::OptionalCUDAGuard guard(q.device());
  opt_in_smem<sparse_decode>(sizeof(Smem));
  launch_pdl("sparse decode", sparse_decode, dim3(CL, rows), dim3(THREADS), sizeof(Smem), a);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.attr("H") = H;
  m.attr("D") = D;
  m.attr("CL") = CL;
  m.attr("MAXK") = MAXK;
  m.def("decode", &decode);
}
