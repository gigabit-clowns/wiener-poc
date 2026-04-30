use multiversion::multiversion;

// Esta macro genera automáticamente código SIMD (AVX2, AVX512, etc.)
// y decide en tiempo de ejecución cuál usar según la CPU del usuario.
#[multiversion(targets("simd"))]
pub fn apply_wiener_cpu_simd(data: &mut [f32]) {
    // Rust vectorizará este bucle automáticamente usando Google Highway por debajo
    for val in data.iter_mut() {
        *val = *val * 0.1;
    }
}
