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

extern "C" __global__ void dft2_forward_real_to_complex(
    const float* d_in,
    float* d_out_real,
    float* d_out_imag,
    int batch,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_freq = batch * n * n;
    if (idx >= total_freq) {
        return;
    }

    int image_stride = n * n;
    int b = idx / image_stride;
    int rem = idx % image_stride;
    int ky = rem / n;
    int kx = rem % n;

    float sum_real = 0.0f;
    float sum_imag = 0.0f;

    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            int input_idx = b * image_stride + y * n + x;
            float angle = WIENER_TWO_PI_F * ((float)kx * (float)x + (float)ky * (float)y) / (float)n;
            float v = d_in[input_idx];
            sum_real += v * cosf(angle);
            sum_imag -= v * sinf(angle);
        }
    }

    d_out_real[idx] = sum_real;
    d_out_imag[idx] = sum_imag;
}

extern "C" __global__ void apply_wiener_frequency(
    float* d_freq_real,
    float* d_freq_imag,
    const float* d_defocus,
    const float* d_wiener_factor,
    int batch,
    int n,
    float pixel_size,
    float wavelength,
    float spherical_aberration,
    float q0
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_freq = batch * n * n;
    if (idx >= total_freq) {
        return;
    }

    int image_stride = n * n;
    int b = idx / image_stride;
    int rem = idx % image_stride;
    int ky = rem / n;
    int kx = rem % n;

    float fx = frequency_component(kx, n, pixel_size);
    float fy = frequency_component(ky, n, pixel_size);
    float k2 = fx * fx + fy * fy;

    float wavelength2 = wavelength * wavelength;
    float angle = WIENER_PI_F * wavelength * k2
        * (0.5f * spherical_aberration * wavelength2 * k2 + d_defocus[b]);

    float ctf = sinf(angle) - q0 * cosf(angle);
    float denom = ctf * ctf + d_wiener_factor[b];
    float scale = (denom > 1e-20f) ? (ctf / denom) : 0.0f;

    d_freq_real[idx] *= scale;
    d_freq_imag[idx] *= scale;
}

extern "C" __global__ void dft2_inverse_complex_to_real(
    const float* d_freq_real,
    const float* d_freq_imag,
    float* d_out,
    int batch,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_spatial = batch * n * n;
    if (idx >= total_spatial) {
        return;
    }

    int image_stride = n * n;
    int b = idx / image_stride;
    int rem = idx % image_stride;
    int y = rem / n;
    int x = rem % n;

    float sum_real = 0.0f;
    for (int ky = 0; ky < n; ++ky) {
        for (int kx = 0; kx < n; ++kx) {
            int freq_idx = b * image_stride + ky * n + kx;
            float angle = WIENER_TWO_PI_F * ((float)kx * (float)x + (float)ky * (float)y) / (float)n;
            float fr = d_freq_real[freq_idx];
            float fi = d_freq_imag[freq_idx];
            sum_real += fr * cosf(angle) - fi * sinf(angle);
        }
    }

    d_out[idx] = sum_real / ((float)n * (float)n);
}
