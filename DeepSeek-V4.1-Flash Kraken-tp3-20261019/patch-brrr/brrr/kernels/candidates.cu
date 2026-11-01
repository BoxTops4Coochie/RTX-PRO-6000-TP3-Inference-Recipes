#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cub/block/block_scan.cuh>
#include <torch/extension.h>

#include "common.cuh"

namespace {

constexpr int THREADS = 1024;

struct Args {
  const float* logits;
  long stride_row;
  long stride_col;
  const int* ends;
  long stride_end;
  int width;
  int k;
  int* out;
  long out_row;
  long out_col;
  int out_k;
  float* values;
  long values_row;
};

__device__ __forceinline__ unsigned order(float v) {
  return v != v ? 0xffffffffu : float_order(v);
}

__device__ __forceinline__ void pick_bin(const unsigned* hist, int remaining, unsigned* chosen, int tid) {
  __shared__ int warp_total[8];
  int count = 0, inclusive = 0;
  if (tid < 256) {
    count = int(hist[255 - tid]);
    inclusive = count;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int up = __shfl_up_sync(0xffffffffu, inclusive, o);
      if ((tid & 31) >= o) inclusive += up;
    }
    if ((tid & 31) == 31) warp_total[tid >> 5] = inclusive;
  }
  __syncthreads();
  if (tid < 256) {
    for (int w = 0; w < (tid >> 5); ++w) inclusive += warp_total[w];
    const int before = inclusive - count;
    if ((before < remaining && inclusive >= remaining) || (tid == 255 && inclusive < remaining)) {
      chosen[0] = unsigned(255 - tid);
      chosen[1] = unsigned(remaining - before);
    }
  }
}

struct Threshold {
  unsigned key;
  int ties;
};

template <class Visit>
__device__ __forceinline__ Threshold radix_select(unsigned* hist, unsigned* chosen, int k, int tid, Visit visit) {
  unsigned prefix = 0, mask = 0;
  int remaining = k;
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int i = tid; i < 256; i += THREADS) hist[i] = 0;
    __syncthreads();
    visit([&](unsigned key) {
      if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255], 1u);
    });
    __syncthreads();
    pick_bin(hist, remaining, chosen, tid);
    __syncthreads();
    prefix |= chosen[0] << shift;
    mask |= 255u << shift;
    remaining = int(chosen[1]);
    __syncthreads();
  }
  return {prefix, remaining};
}

template <int PER>
__global__ void __launch_bounds__(THREADS) candidates_kernel(Args a) {
  using Scan = cub::BlockScan<int, THREADS>;
  __shared__ typename Scan::TempStorage scan;
  __shared__ unsigned hist[256];
  __shared__ unsigned chosen[2];
  const long row = blockIdx.x;
  const int tid = threadIdx.x;
  const int end = a.ends[row * a.stride_end];
  const int pinned = end > 0 ? end - 1 : -1;
  float score[PER];
  unsigned key[PER];
  bool live[PER];
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    const int col = i * THREADS + tid;
    live[i] = col < a.width;
    const float v = live[i] && col < end ? a.logits[row * a.stride_row + long(col) * a.stride_col] : -INFINITY;
    score[i] = col == pinned ? INFINITY : v;
    key[i] = live[i] ? order(score[i]) : 0u;
  }
  const int take = min(a.k, a.width);
  const unsigned floor_key = order(-INFINITY);
  int above = 0;
#pragma unroll
  for (int i = 0; i < PER; ++i) above += live[i] && key[i] > floor_key;
  int total_above;
  Scan(scan).ExclusiveSum(above, above, total_above);
  __syncthreads();
  unsigned threshold = floor_key;
  int ties = 0, rank[PER] = {};
  if (total_above > take) {
    const Threshold th = radix_select(hist, chosen, take, tid, [&](auto count) {
#pragma unroll
      for (int i = 0; i < PER; ++i)
        if (live[i]) count(key[i]);
    });
    threshold = th.key;
    ties = th.ties;
    int base = 0;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      int before, round;
      Scan(scan).ExclusiveSum(int(live[i] && key[i] == threshold), before, round);
      __syncthreads();
      rank[i] = base + before;
      base += round;
    }
  }
  bool keep[PER];
  int kept = 0;
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    const bool selected = live[i] && (key[i] > threshold || (total_above > take && key[i] == threshold && rank[i] < ties));
    keep[i] = selected && score[i] > -INFINITY && score[i] == score[i];
    kept += keep[i];
  }
  int offset, count;
  Scan(scan).ExclusiveSum(kept, offset, count);
  int* out = a.out + row * a.out_row;
  float* values = a.values + row * a.values_row;
#pragma unroll
  for (int i = 0; i < PER; ++i)
    if (keep[i]) {
      values[offset] = score[i];
      out[long(offset++) * a.out_col] = i * THREADS + tid;
    }
  for (int c = count + tid; c < a.out_k; c += THREADS) {
    values[c] = -INFINITY;
    out[long(c) * a.out_col] = -1;
  }
}

template <int PER>
void launch(const Args& a, int rows) {
  candidates_kernel<PER><<<rows, THREADS, 0, c10::cuda::getCurrentCUDAStream()>>>(a);
}

struct SortedArgs {
  const float* logits;
  long stride_row;
  int width;
  const int* lengths;
  const int* visible;
  const int* active;
  int k;
  int* out;
  long out_row;
  float* values;
  long values_row;
  int* out_lengths;
  const int* table;
};

template <bool EXPAND>
__global__ void __launch_bounds__(THREADS) sorted_kernel(SortedArgs a) {
  using Scan = cub::BlockScan<int, THREADS>;
  __shared__ typename Scan::TempStorage scan;
  __shared__ unsigned hist[256];
  __shared__ unsigned chosen[2];
  const long row = blockIdx.x;
  const int tid = threadIdx.x;
  const float* logits = a.logits + row * a.stride_row;
  const int n = min(max(a.lengths[row], 0), a.width);
  const int visible = min(max(a.visible[row], 0), max(a.active[0], 0));
  const int chunk = cdiv(n, THREADS), lo = tid * chunk, hi = min(n, lo + chunk);
  auto key_at = [&](int i) { return order(logits[i]); };
  auto valid = [](float v) { return v > -INFINITY; };
  int mine = 0;
  for (int i = lo; i < hi; ++i) mine += valid(logits[i]);
  int before, total;
  Scan(scan).ExclusiveSum(mine, before, total);
  __syncthreads();
  unsigned threshold = 0;
  int ties = THREADS * 64, tie_base = 0;
  if (total > a.k) {
    const Threshold th = radix_select(hist, chosen, a.k, tid, [&](auto count) {
      for (int i = lo; i < hi; ++i)
        if (valid(logits[i])) count(key_at(i));
    });
    threshold = th.key;
    ties = th.ties;
    int equal = 0;
    for (int i = lo; i < hi; ++i) equal += valid(logits[i]) && key_at(i) == threshold;
    int unused;
    Scan(scan).ExclusiveSum(equal, tie_base, unused);
    __syncthreads();
  }
  auto selected = [&](int i, int& rank) {
    const float v = logits[i];
    if (!valid(v)) return false;
    const unsigned key = key_at(i);
    if (key > threshold) return true;
    if (key != threshold) return false;
    return rank++ < ties;
  };
  const int scale = EXPAND ? 8 : 1;
  const int* table = a.table ? a.table + row * a.width : nullptr;
  auto position = [&](int i) { return table ? table[i] : i; };
  auto shown = [&](int idx) { return idx >= 0 && idx * scale < visible; };
  int kept = 0, covered = 0;
  {
    int rank = tie_base;
    for (int i = lo; i < hi; ++i) {
      const int idx = position(i);
      if (selected(i, rank) && shown(idx)) ++kept, covered += min(scale, visible - idx * scale);
    }
  }
  int offset, count;
  Scan(scan).ExclusiveSum(kept, offset, count);
  int* out = a.out + row * a.out_row;
  float* values = EXPAND ? nullptr : a.values + row * a.values_row;
  {
    int rank = tie_base;
    for (int i = lo; i < hi; ++i) {
      const int idx = position(i);
      if (selected(i, rank) && shown(idx)) {
        if constexpr (EXPAND) {
#pragma unroll
          for (int o = 0; o < 8; ++o) out[offset * 8 + o] = idx * 8 + o < visible ? idx * 8 + o : -1;
        } else {
          values[offset] = logits[i];
          out[offset] = idx;
        }
        ++offset;
      }
    }
  }
  for (int c = count + tid; c < a.k; c += THREADS) {
    if constexpr (EXPAND) {
#pragma unroll
      for (int o = 0; o < 8; ++o) out[c * 8 + o] = -1;
    } else {
      values[c] = -INFINITY;
      out[c] = -1;
    }
  }
  if constexpr (EXPAND) {
    int total_covered;
    __syncthreads();
    Scan(scan).ExclusiveSum(covered, covered, total_covered);
    if (tid == 0) a.out_lengths[row] = total_covered;
  }
}

}

void select_sorted(torch::Tensor logits, torch::Tensor lengths, torch::Tensor visible, torch::Tensor active, int64_t k, torch::Tensor out, std::optional<torch::Tensor> values,
                   std::optional<torch::Tensor> out_lengths, std::optional<torch::Tensor> table) {
  const bool expand = out_lengths.has_value();
  TORCH_CHECK(!table || (table->scalar_type() == at::kInt && table->is_contiguous() && table->sizes() == logits.sizes()));
  TORCH_CHECK(logits.scalar_type() == at::kFloat && logits.stride(1) == 1 && lengths.scalar_type() == at::kInt && visible.scalar_type() == at::kInt && active.scalar_type() == at::kInt);
  TORCH_CHECK(lengths.is_contiguous() && visible.is_contiguous() && out.scalar_type() == at::kInt && out.is_contiguous() && out.numel() >= logits.size(0) * k * (expand ? 8 : 1));
  TORCH_CHECK(expand != values.has_value());
  TORCH_CHECK(expand ? out_lengths->scalar_type() == at::kInt && out_lengths->is_contiguous()
                     : values->scalar_type() == at::kFloat && values->stride(1) == 1 && values->size(1) == k && values->size(0) >= logits.size(0));
  const c10::cuda::CUDAGuard guard(logits.device());
  const SortedArgs a{logits.data_ptr<float>(), logits.stride(0), int(logits.size(1)), lengths.data_ptr<int>(), visible.data_ptr<int>(), active.data_ptr<int>(), int(k),
                     out.data_ptr<int>(), k * (expand ? 8 : 1), expand ? nullptr : values->data_ptr<float>(), expand ? 0 : values->stride(0),
                     expand ? out_lengths->data_ptr<int>() : nullptr, table ? table->data_ptr<int>() : nullptr};
  if (!logits.size(0)) return;
  if (expand) sorted_kernel<true><<<logits.size(0), THREADS, 0, c10::cuda::getCurrentCUDAStream()>>>(a);
  else sorted_kernel<false><<<logits.size(0), THREADS, 0, c10::cuda::getCurrentCUDAStream()>>>(a);
}

bool select_blocks(torch::Tensor logits, torch::Tensor ends, int64_t k, torch::Tensor out, torch::Tensor values) {
  const int rows = int(logits.size(0)), width = int(logits.size(1));
  if (logits.scalar_type() != at::kFloat || ends.scalar_type() != at::kInt || out.scalar_type() != at::kInt || width > 32 * THREADS ||
      values.scalar_type() != at::kFloat || values.stride(1) != 1 || values.size(1) != out.size(1))
    return false;
  const c10::cuda::CUDAGuard guard(logits.device());
  const Args a{logits.data_ptr<float>(), logits.stride(0), logits.stride(1), ends.data_ptr<int>(), ends.stride(0), width, int(k), out.data_ptr<int>(),
               out.stride(0), out.stride(1), int(out.size(1)), values.data_ptr<float>(), values.stride(0)};
  const int per = cdiv(width, THREADS);
  if (per <= 1) launch<1>(a, rows);
  else if (per <= 2) launch<2>(a, rows);
  else if (per <= 4) launch<4>(a, rows);
  else if (per <= 8) launch<8>(a, rows);
  else if (per <= 16) launch<16>(a, rows);
  else launch<32>(a, rows);
  return true;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("select_blocks", &select_blocks);
  m.def("select_sorted", &select_sorted);
}
