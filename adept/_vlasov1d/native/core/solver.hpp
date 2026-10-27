#pragma once
#include "backend.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace adept_native {
void mpi_check(int error) {
    if (error == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(error, message, &length);
    throw std::runtime_error(std::string(message, length));
}
struct Options {
    Grid grid;
    int steps = 20, warmup = 3;
    double dt = 0.1, acceleration = 0.2, nu = 0;
    bool host_mpi = false;
    std::string mode = "landau", dump, integrator = "strang", input, coordinates, background;
};
int integer(const std::string& text) {
    std::size_t end;
    int result = std::stoi(text, &end);
    if (end != text.size()) throw std::runtime_error("invalid integer: " + text);
    return result;
}
double real(const std::string& text) {
    std::size_t end;
    double result = std::stod(text, &end);
    if (end != text.size() || !std::isfinite(result)) throw std::runtime_error("invalid real: " + text);
    return result;
}
Options parse(int argc, char** argv, int rank) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--host-mpi") { o.host_mpi = true; continue; }
        if (key == "--cells") { o.grid.cell_centred = true; continue; }
        if (key == "--help") {
            if (rank == 0) std::cout
                << "vlasov_bench [--nx N] [--nv N] [--steps N] [--warmup N] [--dt D]\n"
                << "  [--mode landau|stream|accelerate|transpose] [--acceleration A]\n"
                << "  [--length L] [--vmax V] [--amplitude A] [--host-mpi] [--dump FILE]\n";
            if (rank == 0) std::cout << "  [--velocity natural|cubic] [--layout transpose|halos] [--halo-capacity N]\n"
                << "  [--integrator strang|sixth] [--cells] [--input FILE] [--v-grid FILE] [--background FILE]\n";
            if (rank == 0) std::cout << "  [--nu D] (Chang-Cooper Dougherty; requires --cells)\n";
            // All ranks take the same path; finalize normally in main.
            o.mode = "help";
            return o;
        }
        if (i + 1 == argc) throw std::runtime_error("missing value for " + key);
        std::string value = argv[++i];
        if (key == "--nx") o.grid.nx = integer(value);
        else if (key == "--nv") o.grid.nv = integer(value);
        else if (key == "--steps") o.steps = integer(value);
        else if (key == "--warmup") o.warmup = integer(value);
        else if (key == "--dt") o.dt = real(value);
        else if (key == "--acceleration") o.acceleration = real(value);
        else if (key == "--nu") o.nu = real(value);
        else if (key == "--charge") o.grid.charge = real(value);
        else if (key == "--mass") o.grid.mass = real(value);
        else if (key == "--length") o.grid.length = real(value);
        else if (key == "--vmax") o.grid.vmax = real(value);
        else if (key == "--amplitude") o.grid.amplitude = real(value);
        else if (key == "--mode") o.mode = value;
        else if (key == "--integrator") o.integrator = value;
        else if (key == "--input") o.input = value;
        else if (key == "--v-grid") o.coordinates = value;
        else if (key == "--background") o.background = value;
        else if (key == "--halo-capacity") o.grid.halo_capacity = integer(value);
        else if (key == "--velocity") {
            if (value != "natural" && value != "cubic") throw std::runtime_error("unknown velocity method");
            o.grid.local_cubic = value == "cubic";
        } else if (key == "--layout") {
            if (value != "transpose" && value != "halos") throw std::runtime_error("unknown layout");
            o.grid.halos = value == "halos";
        }
        else if (key == "--dump") o.dump = value;
        else throw std::runtime_error("unknown option: " + key);
    }
    return o;
}
// Times include explicit device completion, with no computation/communication
// overlap. Reduce each rank's accumulated component times with MPI_MAX later.
enum Stage { Stream, Spline, Pack, Exchange, Field, Collision, Total, StageCount };
struct SixthCoefficients {
    double a[5] = {0.168735950563437422448196, 0.377851589220928303880766,
        -0.093175079568731452657924, 0.377851589220928303880766, 0.168735950563437422448196};
    double d[6];
    explicit SixthCoefficients(double dt) {
        double dt2 = dt * dt, dt4 = dt2 * dt2, dt6 = dt4 * dt2;
        d[0] = d[5] = 0.049086460976116245491441 - 2 * 0.000069728715055305084099 * dt2;
        d[1] = d[4] = 0.264177609888976700200146 - 2 * 0.000625704827430047189169 * dt2
            - 4 * 2.916600457689847816445691e-6 * dt4;
        d[2] = d[3] = 0.186735929134907054308413 - 2 * 0.002213085124045325561636 * dt2
            + 4 * 3.048480261700038788680723e-5 * dt4 - 8 * 4.985549387875068121593988e-7 * dt6;
    }
};
class Stepper {
public:
    Stepper(Backend& backend, const Options& opts) : b(backend), o(opts) {}
    void collision_step() {
        if (o.nu == 0) return;
        // One transpose pair remains for the implicit collision solve; the
        // six Vlasov kicks use halos. Count its communication in MPI/pack times.
        redistribute(true);
        timed(Collision, [&] { b.collide(o.dt, o.nu); });
        redistribute(false);
    }
    void velocity_step(double dt, bool self_consistent) {
        if (!o.grid.halos) {
            redistribute(true);
            timed(Spline, [&] { b.velocity(dt, o.acceleration, self_consistent); });
            redistribute(false);
            return;
        }
        timed(Exchange, [&] {
            double maximum = self_consistent ? std::abs(o.grid.charge / o.grid.mass) * b.max_field() : std::abs(o.acceleration);
            double distance = maximum * std::abs(dt) / o.grid.dv();
            if (!std::isfinite(distance) || distance > o.grid.halo_capacity - 2)
                throw std::runtime_error("departure point exceeds halo capacity; increase --halo-capacity");
            int width = int(std::ceil(distance)) + 2, global_width;
            mpi_check(MPI_Allreduce(&width, &global_width, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
            width = global_width;
            max_halo_width = std::max(max_halo_width, width);
            exchange_halos(width);
        });
        timed(Spline, [&] { b.local_velocity(dt, o.acceleration, self_consistent); });
    }
    void exchange_halos(int width) {
        const Grid& g = o.grid;
        if (g.ranks == 1) return;
        const int start = g.rank * g.local_v(), stop = start + g.local_v();
        std::vector<MPI_Request> requests;
        std::vector<double> host_send, host_recv;
        double* send = b.velocity_buffer();
        double* recv = b.halo_buffer();
        if (o.host_mpi) {
            host_send = b.download();
            host_recv.resize(2 * std::size_t(g.halo_capacity) * g.nx);
            send = host_send.data(); recv = host_recv.data();
        }
        b.synchronize();
        auto post = [&](bool receiving, int peer, int count, double* ptr) {
            requests.push_back(MPI_REQUEST_NULL);
            if (receiving) mpi_check(MPI_Irecv(ptr, count, MPI_DOUBLE, peer, 713, MPI_COMM_WORLD, &requests.back()));
            else mpi_check(MPI_Isend(ptr, count, MPI_DOUBLE, peer, 713, MPI_COMM_WORLD, &requests.back()));
        };
        for (int peer = 0; peer < g.ranks; ++peer) {
            if (peer == g.rank) continue;
            int ps = peer * g.local_v(), pe = ps + g.local_v();
            int lo = std::max(ps, peer < g.rank ? start - width : stop);
            int hi = std::min(pe, peer < g.rank ? start : stop + width);
            if (hi > lo) {
                std::size_t offset = std::size_t(g.halo_capacity + lo - (peer < g.rank ? start : stop)) * g.nx;
                post(true, peer, (hi - lo) * g.nx, recv + offset);
            }
            lo = std::max(start, g.rank < peer ? ps - width : pe);
            hi = std::min(stop, g.rank < peer ? ps : pe + width);
            if (hi > lo) {
                int count = (hi - lo) * g.nx;
                post(false, peer, count, send + std::size_t(lo - start) * g.nx);
                halo_sent_bytes += std::uint64_t(count) * sizeof(double);
            }
        }
        mpi_check(MPI_Waitall(int(requests.size()), requests.data(), MPI_STATUSES_IGNORE));
        if (o.host_mpi) b.upload_halos(host_recv);
    }
    template<class F> void timed(Stage stage, F fn) {
        b.synchronize();
        double start = MPI_Wtime();
        fn();
        b.synchronize();
        times[stage] += MPI_Wtime() - start;
    }
    void redistribute(bool to_x) {
        if (o.grid.ranks == 1) { b.local_layout(); return; }
        timed(Pack, [&] { b.pack(to_x); });
        int count = o.grid.count() / o.grid.ranks;
        timed(Exchange, [&] {
            slab_sent_bytes += std::uint64_t(count) * (o.grid.ranks - 1) * sizeof(double);
            if (o.host_mpi) {
                auto send = b.download_send();
                std::vector<double> recv(send.size());
                mpi_check(MPI_Alltoall(send.data(), count, MPI_DOUBLE, recv.data(), count,
                    MPI_DOUBLE, MPI_COMM_WORLD));
                b.upload_received(recv);
            } else {
                mpi_check(MPI_Alltoall(b.send_buffer(), count, MPI_DOUBLE, b.recv_buffer(),
                    count, MPI_DOUBLE, MPI_COMM_WORLD));
            }
        });
        timed(Pack, [&] { b.unpack(to_x); });
    }
    void solve_field() {
        b.density(o.grid.charge);
        b.synchronize();
        if (o.host_mpi) {
            auto density = b.download_density();
            std::vector<double> global(density.size());
            mpi_check(MPI_Allreduce(density.data(), global.data(), o.grid.nx,
                MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
            b.upload_density(global);
        } else {
            mpi_check(MPI_Allreduce(b.density_buffer(), b.reduced_density_buffer(),
                o.grid.nx, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
        }
        b.field(true);
    }
    void step() {
        b.synchronize();
        double start = MPI_Wtime();
        if (o.mode == "stream") timed(Stream, [&] { b.stream(o.dt); });
        else if (o.mode == "transpose") { redistribute(true); redistribute(false); }
        else if (o.mode == "collide") collision_step();
        else {
            if (o.mode == "landau" && o.integrator == "sixth") {
                // Same force-gradient Hamiltonian coefficients and ordering as
                // ADEPT SixthOrderHamIntegrator, including its dt corrections.
                const SixthCoefficients coefficients(o.dt);
                for (int k = 0; k < 6; ++k) {
                    timed(Field, [&] { solve_field(); });
                    velocity_step(coefficients.d[k] * o.dt, true);
                    if (k < 5) timed(Stream, [&] { b.stream(coefficients.a[k] * o.dt); });
                }
                collision_step();
                b.synchronize();
                times[Total] += MPI_Wtime() - start;
                return;
            }
            if (o.mode == "landau") {
                timed(Stream, [&] { b.stream(0.5 * o.dt); });
                timed(Field, [&] { solve_field(); });
            }
            velocity_step(o.dt, o.mode == "landau");
            if (o.mode == "landau") timed(Stream, [&] { b.stream(0.5 * o.dt); });
            collision_step();
        }
        b.synchronize();
        times[Total] += MPI_Wtime() - start;
    }
    std::array<double, StageCount> times{};
    std::uint64_t halo_sent_bytes = 0;
    std::uint64_t slab_sent_bytes = 0;
    int max_halo_width = 0;
private:
    Backend& b;
    const Options& o;
};
struct Diagnostics { double mass, kinetic, field, minimum; bool finite; };
Diagnostics diagnostics(Backend& b, Stepper& stepper, const Grid& g) {
    auto f = b.download();
    double mass = 0, kinetic = 0, minimum = std::numeric_limits<double>::infinity();
    int finite = 1;
    for (std::size_t i = 0; i < f.size(); ++i) {
        int j = i / g.nx + g.rank * g.local_v();
        double v = g.velocity(j);
        double weight = (!g.cell_centred && (j == 0 || j == g.nv - 1) ? 0.5 : 1.0) * g.dv() * g.length / g.nx;
        mass += f[i] * weight;
        kinetic += 0.5 * v * v * f[i] * weight;
        minimum = std::min(minimum, f[i]);
        finite &= std::isfinite(f[i]);
    }
    double local[] = {mass, kinetic}, global[2];
    mpi_check(MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
    double global_min;
    int global_finite;
    mpi_check(MPI_Allreduce(&minimum, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD));
    mpi_check(MPI_Allreduce(&finite, &global_finite, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
    stepper.solve_field();
    auto field = b.download_field();
    double energy = 0;
    for (double e : field) energy += 0.5 * e * e * g.length / g.nx;
    return {global[0], global[1], energy, global_min, bool(global_finite)};
}
void dump(Backend& b, const Grid& g, const std::string& path) {
    auto local = b.download();
    std::vector<double> full(g.rank == 0 ? std::size_t(g.nx) * g.nv : 0);
    mpi_check(MPI_Gather(local.data(), int(local.size()), MPI_DOUBLE, full.data(),
        int(local.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD));
    if (g.rank == 0) {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(full.data()), full.size() * sizeof(double));
        if (!out) throw std::runtime_error("cannot write dump: " + path);
    }
}
std::string quoted(const std::string& text) {
    std::string out = "\"";
    for (char c : text) {
        if (c == '"' || c == '\\') out += '\\';
        if (c == '\n') out += "\\n";
        else out += c;
    }
    return out + '"';
}
std::vector<double> read_binary(const std::string& path, std::size_t count) {
    std::vector<double> values(count);
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in || in.tellg() != std::streamoff(count * sizeof(double)))
        throw std::runtime_error("wrong binary size: " + path);
    in.seekg(0);
    in.read(reinterpret_cast<char*>(values.data()), count * sizeof(double));
    if (!in) throw std::runtime_error("cannot read " + path);
    return values;
}
void initialize(Backend& backend, const Options& o) {
    if (o.input.empty()) { backend.initialize(); return; }
    MPI_File file;
    mpi_check(MPI_File_open(MPI_COMM_WORLD, o.input.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &file));
    MPI_Offset bytes;
    mpi_check(MPI_File_get_size(file, &bytes));
    if (bytes != MPI_Offset(o.grid.nx) * o.grid.nv * sizeof(double))
        throw std::runtime_error("input state must be raw float64 with shape (nv,nx)");
    std::vector<double> values(o.grid.count());
    MPI_Offset offset = MPI_Offset(o.grid.rank) * values.size() * sizeof(double);
    mpi_check(MPI_File_read_at_all(file, offset, values.data(), int(values.size()), MPI_DOUBLE, MPI_STATUS_IGNORE));
    mpi_check(MPI_File_close(&file));
    backend.upload_state(values);
}
} // namespace adept_native
