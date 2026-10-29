#include "backend.hpp"
#include "local_cubic.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <utility>

#ifdef VLASOV_CUDA
#include <cuda_runtime.h>
#include <cufft.h>
#define DEVICE __device__
static void check_cuda(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
static void check_fft(cufftResult error) {
    if (error != CUFFT_SUCCESS) throw std::runtime_error("cuFFT error " + std::to_string(error));
}
template<class F> __global__ void kernel(std::size_t n, F fn) {
    for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
         i < n; i += std::size_t(blockDim.x) * gridDim.x) fn(i);
}
template<class F> void parallel_for(std::size_t n, F fn) {
    kernel<<<std::min<std::size_t>((n + 255) / 256, 65535), 256>>>(n, fn);
    check_cuda(cudaGetLastError());
}
__global__ void maximum_absolute(const double* values, double* output, int n) {
    __shared__ double partial[256];
    double maximum = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        maximum = fmax(maximum, fabs(values[i]));
    partial[threadIdx.x] = maximum;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmax(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicMax(reinterpret_cast<unsigned long long*>(output),
                  static_cast<unsigned long long>(__double_as_longlong(partial[0])));
}
#else
#include <fftw3.h>
#define DEVICE
template<class F> void parallel_for(std::size_t n, F fn) {
    for (std::size_t i = 0; i < n; ++i) fn(i);
}
#endif

namespace {
constexpr double pi = 3.14159265358979323846;
class Buffer {
public:
    explicit Buffer(std::size_t n) : size(n) {
        if (!n) return;
#ifdef VLASOV_CUDA
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(double)));
#else
        ptr = static_cast<double*>(fftw_malloc(n * sizeof(double)));
        if (!ptr) throw std::bad_alloc();
#endif
    }
    ~Buffer() {
#ifdef VLASOV_CUDA
        cudaFree(ptr);
#else
        fftw_free(ptr);
#endif
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    std::vector<double> download() const {
        std::vector<double> result(size);
#ifdef VLASOV_CUDA
        check_cuda(cudaMemcpy(result.data(), ptr, size * sizeof(double), cudaMemcpyDeviceToHost));
#else
        std::copy_n(ptr, size, result.data());
#endif
        return result;
    }
    void upload(const std::vector<double>& values) {
        if (values.size() != size) throw std::runtime_error("buffer size mismatch");
#ifdef VLASOV_CUDA
        check_cuda(cudaMemcpy(ptr, values.data(), size * sizeof(double), cudaMemcpyHostToDevice));
#else
        std::copy(values.begin(), values.end(), ptr);
#endif
    }
    double* ptr = nullptr;
    std::size_t size;
};

// Plans and workspaces persist for the full run. Complex storage is interleaved.
class FFT {
public:
    FFT(int n, int batch, double* real, double* complex) {
#ifdef VLASOV_CUDA
        check_fft(cufftPlan1d(&forward, n, CUFFT_D2Z, batch));
        check_fft(cufftPlan1d(&backward, n, CUFFT_Z2D, batch));
        std::size_t a = 0, b = 0;
        check_fft(cufftGetSize(forward, &a));
        check_fft(cufftGetSize(backward, &b));
        workspace_bytes = a + b;
#else
        int dims[] = {n};
        forward = fftw_plan_many_dft_r2c(1, dims, batch, real, nullptr, 1, n,
            reinterpret_cast<fftw_complex*>(complex), nullptr, 1, n / 2 + 1, FFTW_ESTIMATE);
        backward = fftw_plan_many_dft_c2r(1, dims, batch,
            reinterpret_cast<fftw_complex*>(complex), nullptr, 1, n / 2 + 1,
            real, nullptr, 1, n, FFTW_ESTIMATE);
        if (!forward || !backward) throw std::runtime_error("FFTW plan creation failed");
#endif
    }
    ~FFT() {
#ifdef VLASOV_CUDA
        cufftDestroy(forward);
        cufftDestroy(backward);
#else
        fftw_destroy_plan(forward);
        fftw_destroy_plan(backward);
#endif
    }
    void r2c(double* real, double* complex) {
#ifdef VLASOV_CUDA
        check_fft(cufftExecD2Z(forward, real, reinterpret_cast<cufftDoubleComplex*>(complex)));
#else
        fftw_execute_dft_r2c(forward, real, reinterpret_cast<fftw_complex*>(complex));
#endif
    }
    void c2r(double* complex, double* real) {
#ifdef VLASOV_CUDA
        check_fft(cufftExecZ2D(backward, reinterpret_cast<cufftDoubleComplex*>(complex), real));
#else
        fftw_execute_dft_c2r(backward, reinterpret_cast<fftw_complex*>(complex), real);
#endif
    }
    std::size_t workspace_bytes = 0;
private:
#ifdef VLASOV_CUDA
    cufftHandle forward{}, backward{};
#else
    fftw_plan forward{}, backward{};
#endif
};

std::string select_device(int local_rank) {
#ifdef VLASOV_CUDA
    int count = 0;
    check_cuda(cudaGetDeviceCount(&count));
    if (!count) throw std::runtime_error("no CUDA devices visible");
    // Slurm may expose one device per rank, or all devices on the node.
    if (count != 1 && local_rank >= count)
        throw std::runtime_error("more local ranks than visible GPUs");
    int device = count == 1 ? 0 : local_rank;
    check_cuda(cudaSetDevice(device));
    cudaDeviceProp props{};
    check_cuda(cudaGetDeviceProperties(&props, device));
    return props.name;
#else
    (void)local_rank;
    return "CPU/FFTW (validation only)";
#endif
}
int current_device_ordinal() {
#ifdef VLASOV_CUDA
    int device = 0;
    check_cuda(cudaGetDevice(&device));
    return device;
#else
    return 0;
#endif
}
} // namespace

void prepare_device_from_environment() {
    int local_rank = 0;
    for (const char* name : {"SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID",
                             "PMI_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK"}) {
        if (const char* value = std::getenv(name)) {
            local_rank = std::stoi(value);
            break;
        }
    }
    select_device(local_rank);
}

struct Backend::Impl {
    Grid grid;
    std::string device;
    int ordinal;
    Buffer fv, fx, send, recv, second, rhs, output, inverse;
    Buffer spectrum, density_local, density_global, electric, field_spectrum;
    Buffer coordinates, weights, edges, primitive, spline_factors, background, ghosts, field_maximum;
    FFT streaming_fft, field_fft;
    Impl(Grid g, int local_rank)
        : grid(g), device(select_device(local_rank)), ordinal(current_device_ordinal()),
          fv(g.count()), fx(!g.halos || g.collisions ? g.count() : 0),
          send((!g.halos || g.collisions) && g.ranks > 1 ? g.count() : 0),
          recv((!g.halos || g.collisions) && g.ranks > 1 ? g.count() : 0),
          second(!g.local_cubic || g.collisions ? g.count() : 0),
          rhs(!g.local_cubic || g.collisions ? g.count() : 0),
          output(g.count()), inverse(!g.local_cubic && !g.conservative ? g.nv : 0),
          spectrum(2 * std::size_t(g.nx / 2 + 1) * g.local_v()),
          density_local(g.nx), density_global(g.nx), electric(g.nx),
          field_spectrum(2 * (g.nx / 2 + 1)),
          coordinates(g.nv), weights(g.nv), edges(g.conservative ? g.nv + 1 : 0),
          primitive(g.conservative ? std::size_t(g.nv + 1) * g.local_x() : 0),
          spline_factors(g.conservative ? 3 * g.nv : 0), background(g.nx),
          ghosts(g.halos ? 2 * std::size_t(g.halo_capacity) * g.nx : 0),
          field_maximum(1),
          streaming_fft(g.nx, g.local_v(), fv.ptr, spectrum.ptr),
          field_fft(g.nx, 1, electric.ptr, field_spectrum.ptr) {
        std::vector<double> inv(g.nv, 0.0);
        // LU factors for natural spline: m[i-1] + 4m[i] + m[i+1] = 6*d2f/dv^2.
        for (int i = 1; i < g.nv - 1; ++i) inv[i] = 1.0 / (4.0 - inv[i - 1]);
        if (!g.local_cubic && !g.conservative) inverse.upload(inv);
        std::vector<double> v(g.nv), bg(g.nx, 1.0);
        for (int j = 0; j < g.nv; ++j) v[j] = g.velocity(j);
        coordinates.upload(v);
        std::vector<double> w(g.nv, g.dv());
        if (!g.cell_centred) w.front() = w.back() = 0.5 * g.dv();
        weights.upload(w);
        background.upload(bg);
        electric.upload(std::vector<double>(g.nx, 0.0));
    }
};

Backend::Backend(Grid grid, int local_rank) {
    if (grid.conservative && (!grid.cell_centred || grid.halos || grid.local_cubic || grid.collisions))
        throw std::runtime_error("conservative splines require cell averages, transpose layout, and no collisions");
    p = std::make_unique<Impl>(grid, local_rank);
    if (grid.conservative) {
        std::vector<double> edges(grid.nv + 1);
        for (int j = 0; j <= grid.nv; ++j) edges[j] = -grid.vmax + j * grid.dv();
        set_velocity_edges(edges);
    }
}
Backend::~Backend() = default;
int Backend::device_ordinal() const {
    return p->ordinal;
}
void Backend::synchronize() {
#ifdef VLASOV_CUDA
    check_cuda(cudaSetDevice(p->ordinal));
    check_cuda(cudaDeviceSynchronize());
#endif
}
void Backend::collide(double dt, double nu) {
    const Grid g = p->grid;
    const int nx = g.local_x(), nv = g.nv;
    const double dv = g.dv(), factor = dt * nu;
    const double* f = p->fx.ptr;
    const double* v = p->coordinates.ptr;
    double* upper = p->second.ptr;
    double* rhs = p->rhs.ptr;
    double* out = p->output.ptr;
    parallel_for(nx, [=] DEVICE (std::size_t x) {
        double sum = 0, momentum = 0;
        for (int j = 0; j < nv; ++j) {
            double value = f[std::size_t(j) * nx + x];
            sum += value; momentum += value * v[j];
        }
        double mean = momentum / sum, variance = 0;
        for (int j = 0; j < nv; ++j) {
            double delta = v[j] - mean;
            variance += f[std::size_t(j) * nx + x] * delta * delta * dv;
        }
        double temperature = variance / (sum * dv);
        // Production self_consistent_beta is disabled: beta=1/(2*T).
        double beta = 1.0 / (2.0 * temperature), D = 1.0 / (2.0 * beta);
        double safe_D = D > 1e-30 ? D : 1e-30;
        auto edge = [=] DEVICE (int j, double& alpha, double& beta_edge) {
            if (j < 0 || j >= nv - 1) { alpha = beta_edge = 0; return; }
            double drift = 2.0 * beta * D * (0.5 * (v[j] + v[j + 1]) - mean);
            double w = drift * dv / safe_D;
            double delta = fabs(w) < 1e-8 ? 0.5 - w / 12.0 + w * w * w / 720.0
                                          : 1.0 / w - 1.0 / expm1(w);
            alpha = -drift * delta + safe_D / dv;
            beta_edge = -drift * (1 - delta) - safe_D / dv;
        };
        double previous_upper = 0, previous_rhs = 0;
        for (int j = 0; j < nv; ++j) {
            double al, bl, ar, br;
            edge(j - 1, al, bl); edge(j, ar, br);
            double diag = 1.0 + factor * (ar - bl) / dv;
            double lower = -factor * al / dv, higher = factor * br / dv;
            std::size_t i = std::size_t(j) * nx + x;
            double applied = diag * f[i];
            if (j > 0) applied += lower * f[i - nx];
            if (j < nv - 1) applied += higher * f[i + nx];
            // Match ADEPT's delta formulation rather than solving for f directly.
            double residual = f[i] - applied;
            double denominator = diag - lower * previous_upper;
            previous_upper = higher / denominator;
            previous_rhs = (residual - lower * previous_rhs) / denominator;
            upper[i] = previous_upper; rhs[i] = previous_rhs;
        }
        double next = 0;
        for (int j = nv - 1; j >= 0; --j) {
            std::size_t i = std::size_t(j) * nx + x;
            next = rhs[i] - upper[i] * next;
            out[i] = f[i] + next;
        }
    });
    std::swap(p->fx.ptr, p->output.ptr);
}
void Backend::initialize() {
    const Grid g = p->grid;
    auto velocities = p->coordinates.download(), weights = p->weights.download();
    auto edges = g.conservative ? p->edges.download() : std::vector<double>{};
    std::vector<double> profile(g.nv);
    double norm = 0.0;
    for (int j = 0; j < g.nv; ++j) {
        double v = velocities[j];
        profile[j] = g.conservative
            ? std::sqrt(pi / 2) * (std::erf(edges[j + 1] / std::sqrt(2.0)) - std::erf(edges[j] / std::sqrt(2.0))) / weights[j]
            : std::exp(-0.5 * v * v);
        norm += profile[j] * weights[j];
    }
    Buffer initial_profile(g.nv);
    initial_profile.upload(profile);
    double* f = p->fv.ptr;
    const int nx = g.nx, nv_local = g.local_v(), v0 = g.rank * nv_local;
    const double* values = initial_profile.ptr;
    const double amp = g.amplitude;
    parallel_for(g.count(), [=] DEVICE (std::size_t i) {
        int x = i % nx, vindex = i / nx + v0;
        f[i] = values[vindex] / norm * (1.0 + amp * cos(2.0 * pi * x / nx));
    });
    synchronize(); // The temporary profile must outlive the initialization kernel.
}
void Backend::stream(double dt) {
    const Grid g = p->grid;
    const int nk = g.nx / 2 + 1, nx = g.nx, v0 = g.rank * g.local_v();
    const double k0 = 2 * pi / g.length;
    const double* vgrid = p->coordinates.ptr;
    double* fk = p->spectrum.ptr;
    double* f = p->fv.ptr;
    p->streaming_fft.r2c(f, fk);
    parallel_for(std::size_t(nk) * g.local_v(), [=] DEVICE (std::size_t i) {
        int k = i % nk;
        double v = vgrid[i / nk + v0];
        double angle = -k0 * k * v * dt;
        double c = cos(angle), s = sin(angle), re = fk[2 * i], im = fk[2 * i + 1];
        fk[2 * i] = re * c - im * s;
        fk[2 * i + 1] = (k == 0 || (nx % 2 == 0 && k == nx / 2)) ? 0.0 : re * s + im * c;
    });
    p->streaming_fft.c2r(fk, f);
    parallel_for(g.count(), [=] DEVICE (std::size_t i) { f[i] /= nx; });
}
void Backend::density(double charge) {
    const Grid g = p->grid;
    const int nx = g.nx, nvl = g.local_v(), v0 = g.rank * nvl;
    const double* weights = p->weights.ptr;
    const double* f = p->fv.ptr;
    double* rho = p->density_local.ptr;
    parallel_for(nx, [=] DEVICE (std::size_t x) {
        double sum = 0.0;
        for (int j = 0; j < nvl; ++j)
            sum += f[std::size_t(j) * nx + x] * weights[j + v0];
        rho[x] = charge * sum;
    });
}
void Backend::add_density(const Backend& other) {
    double* target = p->density_local.ptr;
    const double* source = other.p->density_local.ptr;
    parallel_for(p->grid.nx, [=] DEVICE (std::size_t x) { target[x] += source[x]; });
}
void Backend::copy_field(const Backend& other) {
#ifdef VLASOV_CUDA
    check_cuda(cudaMemcpy(p->electric.ptr, other.p->electric.ptr, p->grid.nx * sizeof(double), cudaMemcpyDeviceToDevice));
#else
    std::copy_n(other.p->electric.ptr, p->grid.nx, p->electric.ptr);
#endif
}
void Backend::field(bool charge_density) {
    const Grid g = p->grid;
    const int nx = g.nx, nk = nx / 2 + 1;
    const double k0 = 2 * pi / g.length;
    const double* density = p->density_global.ptr;
    const double* bg = p->background.ptr;
    double* e = p->electric.ptr;
    double* ek = p->field_spectrum.ptr;
    parallel_for(nx, [=] DEVICE (std::size_t i) { e[i] = bg[i] + (charge_density ? density[i] : -density[i]); });
    p->field_fft.r2c(e, ek);
    parallel_for(nk, [=] DEVICE (std::size_t i) {
        if (i == 0 || (nx % 2 == 0 && i == std::size_t(nx / 2))) {
            ek[2 * i] = ek[2 * i + 1] = 0.0;
        } else {
            double re = ek[2 * i], im = ek[2 * i + 1], k = k0 * i;
            ek[2 * i] = im / k;
            ek[2 * i + 1] = -re / k;
        }
    });
    p->field_fft.c2r(ek, e);
    parallel_for(nx, [=] DEVICE (std::size_t i) { e[i] /= nx; });
}
void Backend::velocity(double dt, double acceleration, bool self_consistent) {
    if (p->grid.conservative) { conservative_velocity(dt, acceleration, self_consistent); return; }
    if (p->grid.local_cubic) { local_velocity(dt, acceleration, self_consistent); return; }
    const Grid g = p->grid;
    const int nxl = g.local_x(), nv = g.nv, x0 = g.rank * nxl;
    const double dv = g.dv();
    const double* f = p->fx.ptr;
    const double* inv = p->inverse.ptr;
    double* m = p->second.ptr;
    double* r = p->rhs.ptr;
    // One thread per spatial line. Keeping x contiguous coalesces each sweep
    // across a warp. This baseline has a serial recurrence along each v line.
    parallel_for(nxl, [=] DEVICE (std::size_t x) {
        m[x] = m[std::size_t(nv - 1) * nxl + x] = 0.0;
        double previous = 0.0;
        for (int j = 1; j < nv - 1; ++j) {
            std::size_t i = std::size_t(j) * nxl + x;
            double b = 6.0 * (f[i - nxl] - 2.0 * f[i] + f[i + nxl]) / (dv * dv);
            previous = (b - previous) * inv[j];
            r[i] = previous;
        }
        double next = 0.0;
        for (int j = nv - 2; j >= 1; --j) {
            std::size_t i = std::size_t(j) * nxl + x;
            next = r[i] - inv[j] * next;
            m[i] = next;
        }
    });
    double* out = p->output.ptr;
    const double* e = p->electric.ptr;
    parallel_for(g.count(), [=] DEVICE (std::size_t i) {
        int x = i % nxl, j = i / nxl;
        double a = self_consistent ? (g.charge / g.mass) * e[x0 + x] : acceleration;
        double q = j - a * dt / dv;
        if (q < 0.0 || q > nv - 1) { out[i] = 0.0; return; }
        int lo = int(floor(q));
        if (lo == nv - 1) lo = nv - 2;
        double b = q - lo, a0 = 1.0 - b;
        std::size_t l = std::size_t(lo) * nxl + x, h = l + nxl;
        out[i] = a0 * f[l] + b * f[h]
            + ((a0 * a0 * a0 - a0) * m[l] + (b * b * b - b) * m[h]) * dv * dv / 6.0;
    });
    std::swap(p->fx.ptr, p->output.ptr);
}
void Backend::conservative_velocity(double dt, double acceleration, bool self_consistent) {
    // Afeyan et al., arXiv:1409.2086, section 2.2. Spline the periodic
    // primitive of f - mean(f), then difference its translated edge values.
    // Subtract mass / period (not mass): the primitive must close on any domain.
    const Grid g = p->grid;
    const int nx = g.local_x(), n = g.nv, x0 = g.rank * nx;
    const double* f = p->fx.ptr;
    const double* h = p->weights.ptr;
    const double* edges = p->edges.ptr;
    const double* factors = p->spline_factors.ptr;
    const double* electric = p->electric.ptr;
    double* primitive = p->primitive.ptr;
    double* second = p->second.ptr;
    double* work = p->rhs.ptr;
    double* out = p->output.ptr;
    parallel_for(nx, [=] DEVICE (std::size_t x) {
        const double period = edges[n] - edges[0];
        double mass = 0;
        for (int j = 0; j < n; ++j) mass += f[std::size_t(j) * nx + x] * h[j];
        const double mean = mass / period;
        double cumulative = 0;
        for (int j = 0; j < n; ++j) {
            const std::size_t i = std::size_t(j) * nx + x;
            primitive[i] = cumulative;
            cumulative += (f[i] - mean) * h[j];
        }
        // The endpoint is identical to the first knot, including roundoff.
        primitive[std::size_t(n) * nx + x] = 0;
        double previous = 0;
        for (int j = 0; j < n; ++j) {
            const std::size_t i = std::size_t(j) * nx + x;
            const int jm = (j + n - 1) % n;
            double slope = (primitive[i + nx] - primitive[i]) / h[j];
            double previous_slope = (primitive[i] - primitive[std::size_t(jm) * nx + x]) / h[jm];
            previous = (6 * (slope - previous_slope) - (j ? h[j - 1] * previous : 0)) * factors[j];
            work[i] = previous;
        }
        double next = 0;
        for (int j = n - 1; j >= 0; --j) {
            const std::size_t i = std::size_t(j) * nx + x;
            next = work[i] - factors[n + j] * next;
            second[i] = next;
        }
        const double corner = h[n - 1] / (-2 * (h[n - 1] + h[0]));
        const double correction = (second[x] + corner * second[std::size_t(n - 1) * nx + x])
            / (1 + factors[2 * n] + corner * factors[3 * n - 1]);
        for (int j = 0; j < n; ++j) second[std::size_t(j) * nx + x] -= correction * factors[2 * n + j];
        // Store the mean in the unused endpoint of the primitive workspace.
        // Evaluation treats the actual endpoint value as zero below.
        primitive[std::size_t(n) * nx + x] = mean;
    });
    parallel_for(g.count(), [=] DEVICE (std::size_t i) {
        int x = i % nx, j = i / nx;
        const double period = edges[n] - edges[0];
        double a = self_consistent ? (g.charge / g.mass) * electric[x0 + x] : acceleration;
        // Reduce the shift first to retain accuracy for multi-period transport.
        double q = fmod(edges[j] - edges[0] - fmod(a * dt, period), period);
        if (q < 0) q += period;
        q += edges[0];
        if (q >= edges[n]) q = edges[0];
        int lo = 0, hi = n;
        while (hi - lo > 1) {
            int mid = (lo + hi) / 2;
            if (edges[mid] <= q) lo = mid;
            else hi = mid;
        }
        const double b = (q - edges[lo]) / h[lo], c = 1 - b;
        const std::size_t left = std::size_t(lo) * nx + x;
        const std::size_t right = std::size_t((lo + 1) % n) * nx + x;
        work[i] = c * primitive[left] + b * primitive[right]
            + ((c * c * c - c) * second[left] + (b * b * b - b) * second[right]) * h[lo] * h[lo] / 6;
    });
    parallel_for(g.count(), [=] DEVICE (std::size_t i) {
        int x = i % nx, j = i / nx;
        out[i] = (work[std::size_t((j + 1) % n) * nx + x] - work[i]) / h[j]
            + primitive[std::size_t(n) * nx + x];
    });
    std::swap(p->fx.ptr, p->output.ptr);
}
void Backend::local_velocity(double dt, double acceleration, bool self_consistent) {
    const Grid g = p->grid;
    const bool halo = g.halos;
    const int width = halo ? g.nx : g.local_x();
    const int nv = g.nv, nvl = g.local_v(), v0 = halo ? g.rank * nvl : 0;
    const int x0 = halo ? 0 : g.rank * g.local_x(), cap = g.halo_capacity;
    const double dv = g.dv();
    const double* f = halo ? p->fv.ptr : p->fx.ptr;
    const double* ghosts = p->ghosts.ptr;
    const double* v = p->coordinates.ptr;
    const double* e = p->electric.ptr;
    double* out = p->output.ptr;
    parallel_for(g.count(), [=] DEVICE (std::size_t i) {
        int x = i % width, j = i / width + v0;
        double a = self_consistent ? (g.charge / g.mass) * e[x0 + x] : acceleration;
        double query = v[j] - a * dt;
        int guess = int(floor(j - a * dt / dv));
        auto sample = [=] DEVICE (int index) {
            int local = index - v0;
            if (!halo || (local >= 0 && local < nvl)) return f[std::size_t(local) * width + x];
            int row = local < 0 ? cap + local : cap + local - nvl;
            return ghosts[std::size_t(row) * width + x];
        };
        out[i] = local_cubic_value(query, guess, v, nv, sample, 1e-30);
    });
    if (halo) std::swap(p->fv.ptr, p->output.ptr);
    else std::swap(p->fx.ptr, p->output.ptr);
}
double Backend::max_field() {
#ifdef VLASOV_CUDA
    check_cuda(cudaMemset(p->field_maximum.ptr, 0, sizeof(double)));
    maximum_absolute<<<std::min((p->grid.nx + 255) / 256, 65535), 256>>>(
        p->electric.ptr, p->field_maximum.ptr, p->grid.nx);
    check_cuda(cudaGetLastError());
    return p->field_maximum.download()[0];
#else
    double maximum = 0;
    for (int x = 0; x < p->grid.nx; ++x) maximum = std::max(maximum, std::abs(p->electric.ptr[x]));
    return maximum;
#endif
}
double* Backend::velocity_buffer() { return p->fv.ptr; }
double* Backend::electric_buffer() { return p->electric.ptr; }
void Backend::set_field(const std::vector<double>& values) { p->electric.upload(values); }
double* Backend::halo_buffer() { return p->ghosts.ptr; }
void Backend::upload_halos(const std::vector<double>& values) { p->ghosts.upload(values); }
void Backend::upload_state(const std::vector<double>& values) { p->fv.upload(values); }
void Backend::set_coordinates(const std::vector<double>& values) {
    if (p->grid.conservative) throw std::runtime_error("conservative splines require velocity edges, not centers");
    if (values.size() != std::size_t(p->grid.nv)) throw std::runtime_error("incorrect velocity coordinate count");
    for (std::size_t i = 1; i < values.size(); ++i) {
        double spacing = values[i] - values[i - 1];
        if (!std::isfinite(spacing) || spacing <= 0 || std::abs(spacing / p->grid.dv() - 1) > 1e-10)
            throw std::runtime_error("velocity halos require uniform coordinates matching --nv/--vmax/--cells");
    }
    p->coordinates.upload(values);
}
std::vector<double> Backend::download_coordinates() { return p->coordinates.download(); }
std::vector<double> Backend::download_weights() { return p->weights.download(); }

void Backend::set_velocity_edges(const std::vector<double>& values) {
    const int n = p->grid.nv;
    if (!p->grid.conservative || values.size() != std::size_t(n + 1))
        throw std::runtime_error("conservative velocity grid requires nv+1 edges");
    std::vector<double> h(n), centers(n), factors(3 * n);
    for (int j = 0; j <= n; ++j) {
        if (!std::isfinite(values[j])) throw std::runtime_error("velocity edges must be finite");
        if (j == n) break;
        h[j] = values[j + 1] - values[j];
        if (!std::isfinite(h[j]) || h[j] <= 0)
            throw std::runtime_error("velocity edges must be strictly increasing with finite widths");
        centers[j] = values[j] + 0.5 * h[j];
    }
    if (!std::isfinite(values.back() - values.front())) throw std::runtime_error("velocity period must be finite");
    // Periodic C2 spline: h[j-1] m[j-1] + 2(h[j-1]+h[j]) m[j]
    //                    + h[j] m[j+1] = 6 (slope[j]-slope[j-1]).
    // Factor the cyclic matrix once via Sherman-Morrison. Store inverse
    // pivots, normalized upper diagonal, and z = T^{-1} (gamma,0,...,h[n-1]).
    const double gamma = -2 * (h.back() + h.front());
    for (int j = 0; j < n; ++j) {
        double lower = j ? h[j - 1] : 0;
        double diagonal = 2 * (h[(j + n - 1) % n] + h[j]);
        if (j == 0) diagonal -= gamma;
        if (j == n - 1) diagonal -= h.back() * h.back() / gamma;
        factors[j] = 1 / (diagonal - (j ? lower * factors[n + j - 1] : 0));
        factors[n + j] = j < n - 1 ? h[j] * factors[j] : 0;
        double rhs = j == 0 ? gamma : (j == n - 1 ? h.back() : 0);
        factors[2 * n + j] = (rhs - (j ? lower * factors[2 * n + j - 1] : 0)) * factors[j];
    }
    for (int j = n - 2; j >= 0; --j)
        factors[2 * n + j] -= factors[n + j] * factors[2 * n + j + 1];
    p->edges.upload(values);
    p->coordinates.upload(centers);
    p->weights.upload(h);
    p->spline_factors.upload(factors);
}
void Backend::set_background(const std::vector<double>& values) { p->background.upload(values); }
void Backend::pack(bool to_x) {
    const Grid g = p->grid;
    const int nxl = g.local_x(), nvl = g.local_v(), nx = g.nx;
    const std::size_t tile = std::size_t(nxl) * nvl;
    const double* input = to_x ? p->fv.ptr : p->fx.ptr;
    double* output = p->send.ptr;
    parallel_for(g.count(), [=] DEVICE (std::size_t i) {
        int dst = i / tile;
        std::size_t in_tile = i % tile;
        int v = in_tile / nxl, x = in_tile % nxl;
        output[i] = to_x ? input[std::size_t(v) * nx + dst * nxl + x]
                           : input[std::size_t(dst * nvl + v) * nxl + x];
    });
}
void Backend::unpack(bool to_x) {
    const Grid g = p->grid;
    const int nxl = g.local_x(), nvl = g.local_v(), nx = g.nx;
    const std::size_t tile = std::size_t(nxl) * nvl;
    const double* input = p->recv.ptr;
    double* output = to_x ? p->fx.ptr : p->fv.ptr;
    parallel_for(g.count(), [=] DEVICE (std::size_t i) {
        int src = i / tile;
        std::size_t in_tile = i % tile;
        int v = in_tile / nxl, x = in_tile % nxl;
        if (to_x) output[std::size_t(src * nvl + v) * nxl + x] = input[i];
        else output[std::size_t(v) * nx + src * nxl + x] = input[i];
    });
}
double* Backend::send_buffer() { return p->send.ptr; }
void Backend::local_layout() { std::swap(p->fv.ptr, p->fx.ptr); }
double* Backend::recv_buffer() { return p->recv.ptr; }
double* Backend::density_buffer() { return p->density_local.ptr; }
double* Backend::reduced_density_buffer() { return p->density_global.ptr; }
std::vector<double> Backend::download() { return p->fv.download(); }
std::vector<double> Backend::download_field() { return p->electric.download(); }
std::vector<double> Backend::download_send() { return p->send.download(); }
std::vector<double> Backend::download_density() { return p->density_local.download(); }
void Backend::upload_received(const std::vector<double>& values) { p->recv.upload(values); }
void Backend::upload_density(const std::vector<double>& values) { p->density_global.upload(values); }
std::string Backend::device_name() const { return p->device; }
std::size_t Backend::allocated_bytes() const {
    return (p->fv.size + p->fx.size + p->send.size + p->recv.size + p->second.size + p->rhs.size
        + p->output.size + p->inverse.size + p->spectrum.size
        + 3 * p->grid.nx + p->field_spectrum.size) * sizeof(double)
        + p->streaming_fft.workspace_bytes + p->field_fft.workspace_bytes
        + (p->coordinates.size + p->weights.size + p->edges.size + p->primitive.size + p->spline_factors.size
            + p->background.size + p->ghosts.size + p->field_maximum.size) * sizeof(double);
}
