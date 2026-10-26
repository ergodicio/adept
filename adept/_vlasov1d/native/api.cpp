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
struct SpeciesState {
    Options options;
    std::unique_ptr<Backend> backend;
    std::unique_ptr<Stepper> stepper;
    int local_rank;
    SpeciesState(Options opts, int local) : options(std::move(opts)), local_rank(local) {
        backend = std::make_unique<Backend>(options.grid, local);
        stepper = std::make_unique<Stepper>(*backend, options);
        if (!options.coordinates.empty()) backend->set_coordinates(read_binary(options.coordinates, options.grid.nv));
        if (!options.background.empty()) backend->set_background(read_binary(options.background, options.grid.nx));
        initialize(*backend, options);
    }
};
struct Engine : SpeciesState {
    std::vector<std::unique_ptr<SpeciesState>> additional;
    Engine(Options opts, int local) : SpeciesState(std::move(opts), local) { ++live_engines; }
    ~Engine() { --live_engines; }
    SpeciesState& species(int index) {
        if (index < 0 || index > int(additional.size())) throw std::runtime_error("invalid species index");
        return index == 0 ? *this : *additional[index - 1];
    }
    template<class F> void each(F action) {
        action(*this);
        for (auto& state : additional) action(*state);
    }
    void solve_field() {
        backend->density(options.grid.charge);
        for (auto& state : additional) {
            state->backend->density(state->options.grid.charge);
            backend->add_density(*state->backend);
        }
        backend->synchronize();
        if (options.host_mpi) {
            auto local = backend->download_density();
            std::vector<double> global(local.size());
            mpi_check(MPI_Allreduce(local.data(), global.data(), options.grid.nx, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
            backend->upload_density(global);
        } else {
            mpi_check(MPI_Allreduce(backend->density_buffer(), backend->reduced_density_buffer(),
                options.grid.nx, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
        }
        backend->field(true);
        for (auto& state : additional) state->backend->copy_field(*backend);
    }
    void step() {
        if (additional.empty()) { stepper->step(); return; }
        backend->synchronize();
        double started = MPI_Wtime();
        const SixthCoefficients coefficients(options.dt);
        for (int k = 0; k < 6; ++k) {
            stepper->timed(Field, [&] { solve_field(); });
            each([&](SpeciesState& state) { state.stepper->velocity_step(coefficients.d[k] * options.dt, true); });
            if (k < 5) each([&](SpeciesState& state) {
                state.stepper->timed(Stream, [&] { state.backend->stream(coefficients.a[k] * options.dt); });
            });
        }
        each([](SpeciesState& state) { state.stepper->collision_step(); });
        backend->synchronize();
        stepper->times[Total] += MPI_Wtime() - started;
    }
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
void validate_options(Options& opts) {
    Grid& g = opts.grid;
    mpi_check(MPI_Comm_rank(MPI_COMM_WORLD, &g.rank));
    mpi_check(MPI_Comm_size(MPI_COMM_WORLD, &g.ranks));
    g.collisions = opts.nu > 0;
    if (g.nx < 8 || g.nv < 4 || g.nx % g.ranks || g.nv % g.ranks ||
        g.count() > std::size_t(std::numeric_limits<int>::max()))
        throw std::runtime_error("native grids must divide across ranks and fit MPI int counts");
    if (!g.local_cubic || !g.halos || !g.cell_centred || opts.dt <= 0 || opts.nu < 0 ||
        g.mass <= 0 || !std::isfinite(g.mass) || !std::isfinite(g.charge) ||
        g.length <= 0 || g.vmax <= 0 || g.halo_capacity < 2 ||
        std::size_t(g.halo_capacity) * g.nx > std::size_t(std::numeric_limits<int>::max()) ||
        (opts.integrator != "sixth" && opts.integrator != "strang"))
        throw std::runtime_error("unsupported native solver options");
}
}
extern "C" {
int av_abi_version() { return 2; }
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
        validate_options(opts);
        MPI_Comm local;
        mpi_check(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local));
        int local_rank;
        mpi_check(MPI_Comm_rank(local, &local_rank));
        mpi_check(MPI_Comm_free(&local));
        result = new Handle(std::make_shared<Engine>(std::move(opts), local_rank));
    })) return nullptr;
    return result;
}
int av_add_species(void* handle, int argc, const char** argv) {
    return guarded([&] {
        auto& e = **static_cast<Handle*>(handle);
        Options opts = parse(argc, const_cast<char**>(argv), e.options.grid.rank);
        validate_options(opts);
        if (opts.grid.nx != e.options.grid.nx || opts.grid.length != e.options.grid.length ||
            opts.dt != e.options.dt || opts.integrator != "sixth" || e.options.integrator != "sixth")
            throw std::runtime_error("coupled species require the same x grid, timestep and sixth-order integrator");
        e.additional.push_back(std::make_unique<SpeciesState>(std::move(opts), e.local_rank));
    });
}
int av_species_info(void* handle, int index, int* output) {
    return guarded([&] {
        auto& s = (**static_cast<Handle*>(handle)).species(index);
        const Grid& g = s.options.grid;
        const int values[] = {g.rank, g.ranks, s.local_rank, device_type(), g.nx, g.nv, g.local_v()};
        std::copy_n(values, 7, output);
    });
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
        for (int i = 0; i < count; ++i) e.step();
        e.backend->synchronize();
    });
}
int av_set_field(void* handle, const char* path) {
    return guarded([&] {
        auto& e = **static_cast<Handle*>(handle);
        e.backend->set_field(read_binary(path, e.options.grid.nx));
    });
}
void* av_species_tensor(void* handle, int species, int field) {
    void* result = nullptr;
    if (guarded([&] {
        auto owner = *static_cast<Handle*>(handle);
        auto& e = owner->species(species);
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
void* av_tensor(void* handle, int field) { return av_species_tensor(handle, 0, field); }
int av_stats(void* handle, double* output) {
    return guarded([&] {
        auto& e = **static_cast<Handle*>(handle);
        std::array<double, StageCount> times{};
        e.each([&](SpeciesState& state) {
            for (int i = 0; i < StageCount; ++i) times[i] += state.stepper->times[i];
        });
        mpi_check(MPI_Allreduce(times.data(), output, StageCount, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
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
void av_abort(int code) {
    int initialized = 0, finalized = 0;
    MPI_Initialized(&initialized);
    if (initialized) MPI_Finalized(&finalized);
    if (initialized && !finalized) MPI_Abort(MPI_COMM_WORLD, code);
    std::_Exit(code);
}
}
