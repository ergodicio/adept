// Narrow C ABI for Python orchestration. Physics is shared with the benchmark.
#include "core/solver.hpp"
#include <atomic>
#include <cstring>
#include <memory>
#include <sstream>

using namespace adept_native;
namespace {
thread_local std::string last_error;
bool own_mpi = false;
std::atomic<int> live_engines{0};
struct Engine {
    Options options;
    std::unique_ptr<Backend> backend;
    std::unique_ptr<Stepper> stepper;
    int local_rank;
    Engine(Options opts, int local) : options(std::move(opts)), local_rank(local) {
        backend = std::make_unique<Backend>(options.grid, local);
        stepper = std::make_unique<Stepper>(*backend, options);
        if (!options.coordinates.empty()) backend->set_coordinates(read_binary(options.coordinates, options.grid.nv));
        if (!options.background.empty()) backend->set_background(read_binary(options.background, options.grid.nx));
        initialize(*backend, options);
        ++live_engines;
    }
    ~Engine() { --live_engines; }
};
using Handle = std::shared_ptr<Engine>;
template<class F> int guarded(F action) {
    try { action(); return 0; }
    catch (const std::exception& error) { last_error = error.what(); return -1; }
}
// Legacy DLPack 0.8 ABI, exported as a "dltensor" capsule by Python.
// https://dmlc.github.io/dlpack/latest/c_api.html
struct DLDevice { int32_t device_type, device_id; };
struct DLDataType { uint8_t code, bits; uint16_t lanes; };
struct DLTensor {
    void* data; DLDevice device; int32_t ndim; DLDataType dtype;
    int64_t* shape; int64_t* strides; uint64_t byte_offset;
};
struct DLManagedTensor {
    DLTensor dl_tensor; void* manager_ctx; void (*deleter)(DLManagedTensor*);
};
struct Export {
    DLManagedTensor managed;
    Handle owner;
    int64_t shape[2], strides[2];
};
void release_tensor(DLManagedTensor* tensor) { delete static_cast<Export*>(tensor->manager_ctx); }
int device_type() {
#ifdef VLASOV_CUDA
    return 2; // kDLCUDA
#else
    return 1; // kDLCPU
#endif
}
}
extern "C" {
const char* av_error() { return last_error.c_str(); }
void* av_create(int argc, const char** argv) {
    void* result = nullptr;
    if (guarded([&] {
        prepare_device_from_environment();
        int initialized;
        mpi_check(MPI_Initialized(&initialized));
        if (!initialized) { mpi_check(MPI_Init(nullptr, nullptr)); own_mpi = true; }
        MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
        int rank;
        mpi_check(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
        Options opts = parse(argc, const_cast<char**>(argv), rank);
        Grid& g = opts.grid;
        g.rank = rank;
        mpi_check(MPI_Comm_size(MPI_COMM_WORLD, &g.ranks));
        g.collisions = opts.nu > 0;
        if (g.nx < 8 || g.nv < 4 || g.nx % g.ranks || g.nv % g.ranks ||
            g.count() > std::size_t(std::numeric_limits<int>::max()))
            throw std::runtime_error("native grid must divide across ranks and fit MPI int counts");
        if (!g.local_cubic || !g.halos || !g.cell_centred || opts.dt <= 0 || opts.nu < 0 ||
            g.length <= 0 || g.vmax <= 0 || g.halo_capacity < 2 ||
            std::size_t(g.halo_capacity) * g.nx > std::size_t(std::numeric_limits<int>::max()) ||
            (opts.integrator != "sixth" && opts.integrator != "strang"))
            throw std::runtime_error("unsupported native solver options");
        MPI_Comm local;
        mpi_check(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local));
        int local_rank;
        mpi_check(MPI_Comm_rank(local, &local_rank));
        mpi_check(MPI_Comm_free(&local));
        result = new Handle(std::make_shared<Engine>(std::move(opts), local_rank));
    })) return nullptr;
    return result;
}
int av_info(void* handle, int* output) {
    return guarded([&] {
        auto& e = **static_cast<Handle*>(handle);
        const Grid& g = e.options.grid;
        const int values[] = {g.rank, g.ranks, e.local_rank, device_type(), g.nx, g.nv, g.local_v()};
        std::copy_n(values, 7, output);
    });
}
int av_broadcast(char* value, int capacity) {
    return guarded([&] { mpi_check(MPI_Bcast(value, capacity, MPI_CHAR, 0, MPI_COMM_WORLD)); });
}
int av_barrier() { return guarded([] { mpi_check(MPI_Barrier(MPI_COMM_WORLD)); }); }
int av_step(void* handle, int count) {
    return guarded([&] {
        if (count < 0) throw std::runtime_error("negative native step count");
        auto& e = **static_cast<Handle*>(handle);
        for (int i = 0; i < count; ++i) e.stepper->step();
        e.backend->synchronize();
    });
}
int av_set_field(void* handle, const char* path) {
    return guarded([&] {
        auto& e = **static_cast<Handle*>(handle);
        e.backend->set_field(read_binary(path, e.options.grid.nx));
    });
}
void* av_tensor(void* handle, int field) {
    void* result = nullptr;
    if (guarded([&] {
        auto owner = *static_cast<Handle*>(handle);
        auto& e = *owner;
        e.backend->synchronize(); // Producer completion before a consumer stream reads.
        auto* tensor = new Export{};
        tensor->owner = std::move(owner);
        tensor->shape[0] = e.options.grid.nx;
        tensor->shape[1] = e.options.grid.local_v();
        tensor->strides[0] = 1;
        tensor->strides[1] = e.options.grid.nx;
        tensor->managed = {{field ? e.backend->electric_buffer() : e.backend->velocity_buffer(),
            {device_type(), device_type() == 2 ? e.local_rank : 0}, field ? 1 : 2,
            {2, 64, 1}, tensor->shape, tensor->strides, 0}, tensor, release_tensor};
        result = &tensor->managed;
    })) return nullptr;
    return result;
}
int av_stats(void* handle, double* output) {
    return guarded([&] {
        auto& e = **static_cast<Handle*>(handle);
        mpi_check(MPI_Allreduce(e.stepper->times.data(), output, StageCount, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
    });
}
int av_close(void* handle) {
    return guarded([&] {
        auto* owner = static_cast<Handle*>(handle);
        if (owner->use_count() != 1) throw std::runtime_error("native buffers still have live DLPack views");
        delete owner;
    });
}
int av_finalize() {
    return guarded([] {
        if (live_engines) throw std::runtime_error("native engines must close before MPI finalization");
        if (own_mpi) { mpi_check(MPI_Finalize()); own_mpi = false; }
    });
}
void av_abort(int code) { MPI_Abort(MPI_COMM_WORLD, code); }
}
