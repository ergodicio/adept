#include "../../adept/_vlasov1d/native/core/solver.hpp"

using namespace adept_native;

int main(int argc, char** argv) {
    try {
        prepare_device_from_environment();
    } catch (const std::exception& error) {
        std::cerr << "device binding before MPI_Init: " << error.what() << '\n';
        return 1;
    }
    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    try {
        Options o = parse(argc, argv, rank);
        if (o.mode == "help") { MPI_Finalize(); return 0; }
        Grid& g = o.grid;
        g.collisions = o.nu > 0;
        g.rank = rank;
        mpi_check(MPI_Comm_size(MPI_COMM_WORLD, &g.ranks));
        if (g.nx < 8 || g.nv < 4 || g.nx % g.ranks || g.nv % g.ranks)
            throw std::runtime_error("nx >= 8 and nv >= 4 must both be divisible by MPI ranks");
        if (g.count() > std::size_t(std::numeric_limits<int>::max()))
            throw std::runtime_error("local state exceeds this prototype's MPI int count limit");
        if (o.steps < 1 || o.warmup < 0 || o.dt <= 0 || g.vmax <= 0 || g.length <= 0 || std::abs(g.amplitude) >= 1)
            throw std::runtime_error("require steps>0, warmup>=0, dt/vmax/length>0 and |amplitude|<1");
        if (o.mode != "landau" && o.mode != "stream" && o.mode != "accelerate" && o.mode != "transpose" && o.mode != "collide")
            throw std::runtime_error("unknown mode: " + o.mode);
        if (o.integrator != "strang" && o.integrator != "sixth") throw std::runtime_error("unknown integrator");
        if (!o.edges.empty() && (!g.conservative || !o.coordinates.empty()))
            throw std::runtime_error("--v-edges requires conservative splines and cannot be combined with --v-grid");
        if (g.halos && (!g.local_cubic || g.halo_capacity < 2))
            throw std::runtime_error("halos require --velocity cubic and capacity >= 2");
        if (g.halos && o.mode == "transpose")
            throw std::runtime_error("transpose-only mode requires --layout transpose");
        if (o.nu < 0 || (o.nu > 0 && !g.cell_centred))
            throw std::runtime_error("collisions require nonnegative --nu and --cells");
        if (std::size_t(g.halo_capacity) * g.nx > std::size_t(std::numeric_limits<int>::max()))
            throw std::runtime_error("halo exceeds MPI int count limit");
        MPI_Comm local;
        mpi_check(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local));
        int local_rank;
        mpi_check(MPI_Comm_rank(local, &local_rank));
        int is_head = local_rank == 0, nodes;
        mpi_check(MPI_Allreduce(&is_head, &nodes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
        char hostname[MPI_MAX_PROCESSOR_NAME];
        int host_length;
        mpi_check(MPI_Get_processor_name(hostname, &host_length));
        {
            Backend backend(g, local_rank);
            std::cerr << "rank=" << rank << " host=" << std::string(hostname, host_length)
                      << " local_rank=" << local_rank << " device=" << backend.device_name() << '\n';
            Stepper stepper(backend, o);
            if (!o.coordinates.empty()) backend.set_coordinates(read_binary(o.coordinates, g.nv));
            if (!o.edges.empty()) backend.set_velocity_edges(read_binary(o.edges, g.nv + 1));
            if (!o.background.empty()) backend.set_background(read_binary(o.background, g.nx));
            initialize(backend, o);
            for (int i = 0; i < o.warmup; ++i) stepper.step();
            // Warmup is excluded from both timings and reported physical time.
            initialize(backend, o);
            auto initial = diagnostics(backend, stepper, g);
            backend.reset_boundary_diagnostics();
            stepper.times.fill(0.0);
            stepper.halo_sent_bytes = 0;
            stepper.slab_sent_bytes = 0;
            stepper.max_halo_width = 0;
            backend.synchronize();
            mpi_check(MPI_Barrier(MPI_COMM_WORLD));
            for (int i = 0; i < o.steps; ++i) stepper.step();
            std::array<double, StageCount> times;
            mpi_check(MPI_Reduce(stepper.times.data(), times.data(), StageCount,
                MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
            auto final = diagnostics(backend, stepper, g);
            const auto boundary = boundary_diagnostics(backend);
            if (!initial.finite || !final.finite) throw std::runtime_error("nonfinite distribution");
            if (!o.dump.empty()) dump(backend, g, o.dump);
            std::uint64_t allocated = backend.allocated_bytes(), max_allocated;
            std::uint64_t max_halo_bytes;
            mpi_check(MPI_Reduce(&stepper.halo_sent_bytes, &max_halo_bytes, 1, MPI_UINT64_T, MPI_MAX, 0, MPI_COMM_WORLD));
            mpi_check(MPI_Reduce(&allocated, &max_allocated, 1, MPI_UINT64_T, MPI_MAX, 0, MPI_COMM_WORLD));
            if (rank == 0) {
                const char* names[] = {"stream_ms", "spline_ms", "pack_ms", "mpi_ms", "field_ms", "collision_ms", "step_ms"};
                std::cout << std::setprecision(16) << "{\"backend\":"
#ifdef VLASOV_CUDA
                          << "\"cuda\""
#else
                          << "\"fftw\""
#endif
                          << ",\"device\":" << quoted(backend.device_name())
                          << ",\"precision\":\"float64\",\"mode\":" << quoted(o.mode)
                          << ",\"integrator\":" << quoted(o.integrator)
                          << ",\"nu\":" << o.nu
                          << ",\"velocity\":" << quoted(g.conservative ? "conservative" : (g.local_cubic ? "cubic" : "natural"))
                          << ",\"layout\":" << quoted(g.halos ? "halos" : "transpose")
                          << ",\"cell_centred\":" << (g.cell_centred ? "true" : "false")
                          << ",\"max_halo_width\":" << stepper.max_halo_width
                          << ",\"rank0_halo_sent_bytes\":" << stepper.halo_sent_bytes
                          << ",\"halo_sent_bytes_per_rank_max\":" << max_halo_bytes
                          << ",\"slab_sent_bytes_per_rank\":" << stepper.slab_sent_bytes
                          << ",\"input\":" << quoted(o.input)
                          << ",\"nx\":" << g.nx << ",\"nv\":" << g.nv << ",\"ranks\":" << g.ranks
                          << ",\"nodes\":" << nodes << ",\"steps\":" << o.steps << ",\"warmup\":" << o.warmup
                          << ",\"dt\":" << o.dt << ",\"length\":" << g.length << ",\"vmax\":" << g.vmax
                          << ",\"amplitude\":" << g.amplitude << ",\"acceleration\":" << o.acceleration
                          << ",\"host_mpi\":" << (o.host_mpi ? "true" : "false")
                          << ",\"state_bytes\":" << std::uint64_t(g.nx) * g.nv * sizeof(double)
                          << ",\"max_allocated_bytes_per_rank\":" << max_allocated;
                for (int i = 0; i < StageCount; ++i)
                    std::cout << ",\"" << names[i] << "\":" << times[i] * 1000.0 / o.steps;
                std::cout << ",\"initial_mass\":" << initial.mass << ",\"final_mass\":" << final.mass
                          << ",\"mass_relative_change\":" << (final.mass - initial.mass) / initial.mass
                          << ",\"velocity_boundary\":"
                          << quoted(g.conservative ? (g.periodic_velocity ? "periodic" : "zero-flux") : "not-applicable")
                          << ",\"max_tail_fraction\":" << boundary[0]
                          << ",\"max_attempted_escape_fraction\":" << boundary[1]
                          << ",\"attempted_mass_left\":" << boundary[2]
                          << ",\"attempted_mass_right\":" << boundary[3]
                          << ",\"initial_kinetic_energy\":" << initial.kinetic << ",\"final_kinetic_energy\":" << final.kinetic
                          << ",\"initial_field_energy\":" << initial.field << ",\"final_field_energy\":" << final.field
                          << ",\"minimum_f\":" << final.minimum << ",\"finite\":true}\n";
            }
        } // Destroy CUDA resources before MPI finalization.
        mpi_check(MPI_Comm_free(&local));
        MPI_Finalize();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "rank " << rank << ": " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
