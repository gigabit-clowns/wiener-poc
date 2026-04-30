use multiversion::multiversion;

// We specify explicit targets to ensure the compiler generates optimized
// versions for SSE, AVX2 and AVX512.
#[multiversion(targets("x86_64+avx2", "x86_64+avx512f", "x86_64+sse4.2"))]
pub fn apply_wiener_cpu_simd(data: &mut [f32]) {
    for val in data.iter_mut() {
        *val = *val * 0.1;
    }
}
