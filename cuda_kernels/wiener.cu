#include <cuda_runtime.h>

#ifndef WIENER_PI_F
#define WIENER_PI_F 3.14159265358979323846f
#endif

#ifndef WIENER_TWO_PI_F
#define WIENER_TWO_PI_F 6.28318530717958647692f
#endif

__device__ __forceinline__ float frequency_component(int idx, int n, float pixel_size) {
    int half = n / 2;
    int shifted = (idx <= half) ? idx : (idx - n);
    return ((float)shifted) / ((float)n * pixel_size);
}

extern "C" __global__ void apply_wiener_frequency(
    float2* d_freq,
    const float* d_defocus,
    const float* d_wiener_factor,
    int batch,
    int n,
    int freq_width,
    float pixel_size,
    float wavelength,
    float spherical_aberration,
    float q0
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_freq = batch * n * freq_width;
    if (idx >= total_freq) {
        return;
    }

    int image_stride = n * freq_width;
    int b = idx / image_stride;
    int rem = idx % image_stride;
    int ky = rem / freq_width;
    int kx = rem % freq_width;

    float fx = ((float)kx) / ((float)n * pixel_size);
    float fy = frequency_component(ky, n, pixel_size);
    float k2 = fx * fx + fy * fy;

    float wavelength2 = wavelength * wavelength;
    float angle = WIENER_PI_F * wavelength * k2
        * (0.5f * spherical_aberration * wavelength2 * k2 + d_defocus[b]);

    float ctf = sinf(angle) - q0 * cosf(angle);
    float denom = ctf * ctf + d_wiener_factor[b];
    float scale = (denom > 1e-20f) ? (ctf / denom) : 0.0f;

    float2 value = d_freq[idx];
    value.x *= scale;
    value.y *= scale;
    d_freq[idx] = value;
}

extern "C" __global__ void normalize_real_image(
    float* d_out,
    int total_elements,
    float norm
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) {
        return;
    }
    d_out[idx] *= norm;
}
