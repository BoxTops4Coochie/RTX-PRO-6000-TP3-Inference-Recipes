#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "common.cuh"
#include "host.cuh"

#include <vector>

namespace {

constexpr int MAX_CTAS = 64;
constexpr int THREADS = 512;
constexpr int FLAG_WORDS = RANKS * MAX_CTAS;
constexpr int LL_WORDS = 4096;

enum : int { L_EPOCH, L_ARRIVED, L_BASE, L_BARRIER, LOCAL_WORDS };

struct Peers {
  uint4* inbox[RANKS];
  unsigned* flags[RANKS];
};

struct State {
  Peers peers;
  unsigned* local;
  int rank;
  long slot_vectors;
};

std::vector<State> states;

struct Ll {
  unsigned long long* recv[RANKS];
  unsigned* epoch;
  int rank;
};

std::vector<Ll> lls;

struct Phase {
  unsigned base;
  unsigned epoch;
  int parity;
  int lo;
  int hi;
};

__device__ __forceinline__ Phase begin(const unsigned* local, int count) {
  const unsigned e = ld_volatile(local + L_EPOCH);
  const int per = (count + gridDim.x - 1) / gridDim.x;
  const int lo = blockIdx.x * per;
  return {ld_volatile(local + L_BASE), e + 1, int(e & 1), lo, min(count, lo + per)};
}

template <class T>
__device__ __forceinline__ const T* exchange(const Peers& peers, unsigned* local, const T* in, long slot, int rank, const Phase& ph) {
  unsigned* barrier = local + L_BARRIER;
  for (int step = 1; step < RANKS; ++step) {
    const int target = (rank + step) % RANKS;
    T* dst = reinterpret_cast<T*>(peers.inbox[target]) + (long(ph.parity) * RANKS + rank) * slot;
    for (int i = ph.lo + threadIdx.x; i < ph.hi; i += THREADS) dst[i] = in[i];
    __syncthreads();
    if (threadIdx.x == 0) {
      __threadfence_system();
      st_release_sys(peers.flags[target] + rank * MAX_CTAS + blockIdx.x, ph.epoch);
      if (step < RANKS - 1) {
        atomicAdd(barrier, 1);
        while (before(ld_volatile(barrier), ph.base + gridDim.x * step)) {}
      }
    }
    __syncthreads();
  }
  if (threadIdx.x < RANKS && threadIdx.x != rank)
    while (before(ld_acquire_sys(peers.flags[rank] + threadIdx.x * MAX_CTAS + blockIdx.x), ph.epoch)) {}
  __syncthreads();
  return reinterpret_cast<const T*>(peers.inbox[rank]) + long(ph.parity) * RANKS * slot;
}

__device__ __forceinline__ void finish(unsigned* local, const Phase& ph) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(local + L_ARRIVED, 1) == gridDim.x - 1) {
      local[L_ARRIVED] = 0;
      local[L_BASE] = ph.base + gridDim.x * (RANKS - 2);
      __threadfence();
      atomicExch(local + L_EPOCH, ph.epoch);
    }
  }
}

__global__ void __launch_bounds__(THREADS)
    phased_all_reduce(Peers peers, unsigned* local, const uint4* __restrict__ in, uint4* __restrict__ out, int vectors, long slot_vectors, int rank) {
  const Phase ph = begin(local, vectors);
  const uint4* boxes = exchange(peers, local, in, slot_vectors, rank, ph);
  for (int i = ph.lo + threadIdx.x; i < ph.hi; i += THREADS) {
    float acc[8] = {};
#pragma unroll
    for (int r = 0; r < RANKS; ++r) add8(acc, r == rank ? in[i] : __ldcg(boxes + r * slot_vectors + i));
    out[i] = pack8(acc);
  }
  finish(local, ph);
}

__global__ void __launch_bounds__(THREADS)
    phased_all_gather(Peers peers, unsigned* local, const uint4* __restrict__ in, unsigned* __restrict__ out, int words, int inner, long slot_vectors, int rank) {
  const Phase ph = begin(local, cdiv(words, 4));
  const unsigned* boxes = reinterpret_cast<const unsigned*>(exchange(peers, local, in, slot_vectors, rank, ph));
  const unsigned* mine = reinterpret_cast<const unsigned*>(in);
  const long slot_words = slot_vectors * 4;
  for (int i = ph.lo * 4 + threadIdx.x; i < min(words, ph.hi * 4); i += THREADS) {
    unsigned* dst = out + long(i / inner) * RANKS * inner + i % inner;
#pragma unroll
    for (int r = 0; r < RANKS; ++r) dst[long(r) * inner] = r == rank ? mine[i] : __ldcg(boxes + r * slot_words + i);
  }
  finish(local, ph);
}

__global__ void __launch_bounds__(THREADS) ll_gather(Ll s, const unsigned* __restrict__ in, unsigned* __restrict__ out, int words) {
  const unsigned epoch = ld_volatile(s.epoch) + 1;
  const long parity = long(epoch & 1) * RANKS * LL_WORDS;
  for (int w = threadIdx.x; w < words; w += THREADS) {
    const unsigned v = in[w];
    const unsigned long long packed = (static_cast<unsigned long long>(epoch) << 32) | v;
    for (int step = 1; step < RANKS; ++step) st_relaxed_sys64(s.recv[(s.rank + step) % RANKS] + parity + s.rank * LL_WORDS + w, packed);
    out[long(s.rank) * words + w] = v;
  }
  const unsigned long long* mine = s.recv[s.rank] + parity;
  for (int step = 1; step < RANKS; ++step) {
    const int peer = (s.rank + step) % RANKS;
    for (int w = threadIdx.x; w < words; w += THREADS) {
      unsigned long long got = ld_volatile64(mine + peer * LL_WORDS + w);
      while (unsigned(got >> 32) != epoch) got = ld_volatile64(mine + peer * LL_WORDS + w);
      out[long(peer) * words + w] = unsigned(got);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) *s.epoch = epoch;
}

}

int64_t flag_bytes() { return FLAG_WORDS * sizeof(unsigned); }

int64_t create(std::vector<int64_t> buffers, int64_t rank, int64_t slot_bytes) {
  TORCH_CHECK(buffers.size() == RANKS && slot_bytes % 16 == 0);
  State s{};
  for (int p = 0; p < RANKS; ++p) {
    s.peers.flags[p] = reinterpret_cast<unsigned*>(buffers[p]);
    s.peers.inbox[p] = reinterpret_cast<uint4*>(buffers[p] + flag_bytes());
  }
  s.rank = rank;
  s.slot_vectors = slot_bytes / 16;
  s.local = zeroed<unsigned>(LOCAL_WORDS);
  TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  states.push_back(s);
  return states.size() - 1;
}

int64_t ll_bytes() { return 2L * RANKS * LL_WORDS * sizeof(unsigned long long); }

int64_t create_ll(std::vector<int64_t> buffers, int64_t rank) {
  TORCH_CHECK(buffers.size() == RANKS);
  Ll s{};
  for (int p = 0; p < RANKS; ++p) s.recv[p] = reinterpret_cast<unsigned long long*>(buffers[p]);
  s.rank = rank;
  s.epoch = zeroed<unsigned>(1);
  TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  lls.push_back(s);
  return lls.size() - 1;
}

void gather_words(int64_t handle, torch::Tensor in, torch::Tensor out) {
  const auto& s = lls.at(handle);
  TORCH_CHECK(in.is_contiguous() && out.is_contiguous() && in.element_size() == 4 && out.nbytes() == RANKS * in.nbytes() && in.numel() <= LL_WORDS);
  const c10::cuda::OptionalCUDAGuard guard(in.device());
  ll_gather<<<1, THREADS, 0, c10::cuda::getCurrentCUDAStream()>>>(s, data_as<const unsigned>(in), data_as<unsigned>(out), in.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void all_reduce(int64_t handle, torch::Tensor in, torch::Tensor out, int64_t ctas) {
  const auto& s = states.at(handle);
  TORCH_CHECK(in.is_contiguous() && out.is_contiguous() && in.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16);
  const long bytes = in.numel() * 2;
  TORCH_CHECK(bytes % 16 == 0 && bytes / 16 <= s.slot_vectors && ctas >= 1 && ctas <= MAX_CTAS);
  const c10::cuda::OptionalCUDAGuard guard(in.device());
  phased_all_reduce<<<ctas, THREADS, 0, c10::cuda::getCurrentCUDAStream()>>>(
      s.peers, s.local, data_as<const uint4>(in), data_as<uint4>(out), bytes / 16, s.slot_vectors, s.rank);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void all_gather(int64_t handle, torch::Tensor in, torch::Tensor out, int64_t inner_bytes, int64_t ctas) {
  const auto& s = states.at(handle);
  TORCH_CHECK(in.is_contiguous() && out.is_contiguous() && out.nbytes() == RANKS * in.nbytes());
  const long bytes = in.nbytes();
  TORCH_CHECK(bytes % 4 == 0 && inner_bytes % 4 == 0 && bytes % inner_bytes == 0 && (bytes + 15) / 16 <= s.slot_vectors && ctas >= 1 && ctas <= MAX_CTAS);
  TORCH_CHECK(reinterpret_cast<uintptr_t>(in.data_ptr()) % 16 == 0);
  const c10::cuda::OptionalCUDAGuard guard(in.device());
  phased_all_gather<<<ctas, THREADS, 0, c10::cuda::getCurrentCUDAStream()>>>(
      s.peers, s.local, data_as<const uint4>(in), data_as<unsigned>(out), bytes / 4, inner_bytes / 4, s.slot_vectors, s.rank);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.attr("RANKS") = RANKS;
  m.def("flag_bytes", &flag_bytes);
  m.def("alloc_ipc", &alloc_ipc);
  m.def("open_ipc", &open_ipc);
  m.def("create", &create);
  m.def("all_reduce", &all_reduce);
  m.def("all_gather", &all_gather);
  m.def("ll_bytes", &ll_bytes);
  m.def("create_ll", &create_ll);
  m.def("gather_words", &gather_words);
}
