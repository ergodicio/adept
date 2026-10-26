#pragma once
#include <cmath>

#ifdef __CUDACC__
#define CUBIC_HD __host__ __device__
#else
#define CUBIC_HD
#endif

// interpax method="cubic": C1 Hermite interpolation, arithmetic mean of
// neighbouring secant slopes, one-sided slopes at physical endpoints.
template<class Sample>
CUBIC_HD double local_cubic_value(double query, int guess, const double* v,
                                 int nv, Sample sample, double extrap) {
    if (query < v[0] || query > v[nv - 1]) return extrap;
    int lo = guess < 0 ? 0 : (guess >= nv - 1 ? nv - 2 : guess);
    while (lo > 0 && query < v[lo]) --lo;
    while (lo < nv - 2 && query >= v[lo + 1]) ++lo;
    double f0 = sample(lo), f1 = sample(lo + 1), h = v[lo + 1] - v[lo];
    double secant = (f1 - f0) / h;
    double d0 = lo == 0 ? secant :
        0.5 * ((f0 - sample(lo - 1)) / (v[lo] - v[lo - 1]) + secant);
    double d1 = lo + 1 == nv - 1 ? secant :
        0.5 * (secant + (sample(lo + 2) - f1) / (v[lo + 2] - v[lo + 1]));
    double t = (query - v[lo]) / h;
    double s0 = d0 * h, s1 = d1 * h;
    double c3 = 2.0 * f0 - 2.0 * f1 + s0 + s1;
    double c2 = -3.0 * f0 + 3.0 * f1 - 2.0 * s0 - s1;
    return ((c3 * t + c2) * t + s0) * t + f0;
}
