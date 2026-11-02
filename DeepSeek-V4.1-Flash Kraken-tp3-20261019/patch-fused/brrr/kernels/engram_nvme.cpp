#include <torch/extension.h>
#include <fcntl.h>
#include <immintrin.h>
#include <liburing.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>

constexpr unsigned DEPTH = 1024;
constexpr int64_t ROW = 256;
constexpr int64_t SECTOR = 512;
constexpr unsigned OFFSET_BITS = 10;
constexpr int64_t SLOT = 2 * SECTOR;
constexpr int64_t DIRECT_ROWS = 512;
constexpr auto SPIN_WINDOW = std::chrono::milliseconds(50);
constexpr auto IDLE_POLL = std::chrono::microseconds(500);

struct Store {
    int64_t base;
    const int64_t* ids;
    uint8_t* rows;
    uint8_t* staging;
    std::vector<unsigned> free;
    int64_t* ids_ready;
    int64_t* rows_ready;
    io_uring ring;
};

static std::vector<std::unique_ptr<Store>> stores;

static void fail(const char* what, int error) {
    std::fprintf(stderr, "engram row read failed: %s (%d)\n", what, error);
    std::abort();
}

static void read_rows(Store* store, int64_t count) {
    const bool direct = count <= DIRECT_ROWS;
    int64_t next = 0, done = 0;
    while (done < count) {
        while (next < count && !store->free.empty()) {
            io_uring_sqe* sqe = io_uring_get_sqe(&store->ring);
            if (!sqe) break;
            const unsigned slot = store->free.back();
            store->free.pop_back();
            const int64_t offset = store->base + std::max<int64_t>(store->ids[next], 0) * ROW;
            const int64_t first = direct ? offset & ~(SECTOR - 1) : offset, last = direct ? (offset + ROW + SECTOR - 1) & ~(SECTOR - 1) : offset + ROW;
            io_uring_prep_read(sqe, direct ? 0 : 1, store->staging + slot * SLOT, unsigned(last - first), first);
            sqe->flags |= IOSQE_FIXED_FILE;
            io_uring_sqe_set_data64(sqe, uint64_t(next) << 32 | uint64_t(slot) << OFFSET_BITS | uint64_t(offset - first));
            ++next;
        }
        const int submitted = io_uring_submit(&store->ring);
        if (submitted < 0) fail("submit", -submitted);
        unsigned head, seen = 0;
        io_uring_cqe* cqe;
        while (!seen) {
            io_uring_for_each_cqe(&store->ring, head, cqe) {
                if (cqe->res < ROW) fail("short read", cqe->res);
                const uint64_t data = io_uring_cqe_get_data64(cqe);
                const unsigned slot = unsigned(data >> OFFSET_BITS) & (DEPTH - 1);
                std::memcpy(store->rows + int64_t(data >> 32) * ROW, store->staging + slot * SLOT + (data & ((1u << OFFSET_BITS) - 1)), ROW);
                store->free.push_back(slot);
                ++seen;
            }
            if (!seen) _mm_pause();
        }
        io_uring_cq_advance(&store->ring, seen);
        done += seen;
    }
}

static void poller(Store* store) {
    int64_t seen = 0;
    auto last = std::chrono::steady_clock::now();
    for (;;) {
        const int64_t value = __atomic_load_n(store->ids_ready, __ATOMIC_ACQUIRE);
        if (value == seen) {
            if (std::chrono::steady_clock::now() - last > SPIN_WINDOW) std::this_thread::sleep_for(IDLE_POLL);
            else _mm_pause();
            continue;
        }
        seen = value;
        read_rows(store, value & 0xffffffff);
        __atomic_store_n(store->rows_ready, value >> 32, __ATOMIC_RELEASE);
        last = std::chrono::steady_clock::now();
    }
}

int64_t create_store(std::string path, int64_t base, torch::Tensor ids, torch::Tensor rows,
                     torch::Tensor ids_ready, torch::Tensor rows_ready) {
    int fds[2] = {open(path.c_str(), O_RDONLY | O_DIRECT), open(path.c_str(), O_RDONLY)};
    TORCH_CHECK(fds[0] >= 0 && fds[1] >= 0, "cannot open ", path);
    posix_fadvise(fds[1], 0, 0, POSIX_FADV_RANDOM);
    auto store = std::make_unique<Store>();
    TORCH_CHECK(posix_memalign(reinterpret_cast<void**>(&store->staging), 4096, DEPTH * SLOT) == 0);
    for (unsigned slot = DEPTH; slot-- > 0;) store->free.push_back(slot);
    store->base = base;
    store->ids = ids.data_ptr<int64_t>();
    store->rows = rows.data_ptr<uint8_t>();
    store->ids_ready = ids_ready.data_ptr<int64_t>();
    store->rows_ready = rows_ready.data_ptr<int64_t>();
    int error = io_uring_queue_init(DEPTH, &store->ring, 0);
    TORCH_CHECK(error == 0, "io_uring_queue_init: ", strerror(-error));
    error = io_uring_register_files(&store->ring, fds, 2);
    TORCH_CHECK(error == 0, "io_uring_register_files: ", strerror(-error));
    std::thread(poller, store.get()).detach();
    stores.push_back(std::move(store));
    return static_cast<int64_t>(stores.size() - 1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.attr("ROW") = ROW;
    m.def("create_store", &create_store);
}
