#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

// Cray MPI initializes its GPU transport during MPI_Init, so bind first.
void prepare_device_from_environment();

struct Grid {
    int nx = 4096, nv = 1024, ranks = 1, rank = 0;
    double length = 12.566370614359172, vmax = 8.0, amplitude = 0.01;
    double charge = -1.0, mass = 1.0;
    bool cell_centred = false, local_cubic = false, conservative = false, halos = false, collisions = false;
    bool periodic_velocity = false;
    int halo_capacity = 64;
    int local_x() const { return nx / ranks; }
    int local_v() const { return nv / ranks; }
    int spline_nodes() const { return nv + (conservative && !periodic_velocity ? 1 : 0); }
    std::size_t count() const { return std::size_t(nx) * local_v(); }
    double dv() const { return 2.0 * vmax / (cell_centred ? nv : nv - 1); }
    double velocity(int j) const { return -vmax + (j + (cell_centred ? 0.5 : 0.0)) * dv(); }
};

// Buffers are GPU-resident in the CUDA build; MPI must support device pointers.
// Both layouts keep the spatial index contiguous, including the spline batches.
class Backend {
public:
    explicit Backend(Grid grid, int local_rank);
    ~Backend();
    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;
    void initialize();
    void synchronize();
    void stream(double dt);
    void density(double charge = 1.0);
    void add_density(const Backend& other);
    void field(bool charge_density = false);
    void copy_field(const Backend& other);
    void velocity(double dt, double acceleration, bool self_consistent);
    void conservative_velocity(double dt, double acceleration, bool self_consistent);
    void zero_flux_velocity(double dt, double acceleration, bool self_consistent);
    // Max tail fraction, max attempted-escape fraction per kick, and cumulative
    // attempted mass through left/right boundaries (integrated over local x).
    std::array<double, 4> boundary_diagnostics();
    void reset_boundary_diagnostics();
    void local_velocity(double dt, double acceleration, bool self_consistent);
    void collide(double dt, double nu);
    double max_field();
    double* velocity_buffer();
    double* electric_buffer();
    void set_field(const std::vector<double>& values);
    double* halo_buffer();
    void upload_halos(const std::vector<double>& values);
    void upload_state(const std::vector<double>& values);
    void set_coordinates(const std::vector<double>& values);
    void set_velocity_edges(const std::vector<double>& values);
    std::vector<double> download_coordinates();
    std::vector<double> download_weights();
    void set_background(const std::vector<double>& values);
    void pack(bool to_x);
    void unpack(bool to_x);
    void local_layout();
    double* send_buffer();
    double* recv_buffer();
    double* density_buffer();
    double* reduced_density_buffer();
    std::vector<double> download();
    std::vector<double> download_field();
    void upload_received(const std::vector<double>& values);
    std::vector<double> download_send();
    std::vector<double> download_density();
    void upload_density(const std::vector<double>& values);
    std::string device_name() const;
    int device_ordinal() const;
    std::size_t allocated_bytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> p;
};
