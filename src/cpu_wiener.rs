use crate::wiener_common::{
    build_frequency2_grid_2d, compute_ctf_values, compute_wiener_factors, WienerParams,
};
use multiversion::multiversion;
use num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

pub fn apply_wiener_cpu_simd(
    images: &[f32],
    batch: usize,
    n: usize,
    defocus: &[f32],
    params: &WienerParams,
) -> Result<Vec<f32>, String> {
    if n == 0 {
        return Err("box_size must be > 0".to_string());
    }

    if images.len() != batch * n * n {
        return Err(format!(
            "Invalid image buffer size. Got {}, expected {}",
            images.len(),
            batch * n * n
        ));
    }

    if defocus.len() != batch {
        return Err(format!(
            "Invalid defocus length. Got {}, expected {}",
            defocus.len(),
            batch
        ));
    }

    let mut planner = FftPlanner::<f32>::new();
    let fft_forward = planner.plan_fft_forward(n);
    let fft_inverse = planner.plan_fft_inverse(n);

    let k2_grid = build_frequency2_grid_2d(n, params.pixel_size);
    let mut out = vec![0.0f32; images.len()];
    let wiener_factors = compute_wiener_factors(defocus, n, params);
    let norm = 1.0 / ((n * n) as f32);

    for b in 0..batch {
        let base = b * n * n;
        let mut freq = images[base..base + n * n]
            .iter()
            .copied()
            .map(|v| Complex32::new(v, 0.0))
            .collect::<Vec<_>>();

        fft2_in_place(&mut freq, n, &fft_forward);

        let ctf = compute_ctf_values(&k2_grid, defocus[b], params);
        apply_wiener_frequency_domain(&mut freq, &ctf, wiener_factors[b]);

        fft2_in_place(&mut freq, n, &fft_inverse);

        for (idx, val) in freq.iter().enumerate() {
            out[base + idx] = val.re * norm;
        }
    }

    Ok(out)
}

fn fft2_in_place(data: &mut [Complex32], n: usize, fft: &Arc<dyn Fft<f32>>) {
    for row in 0..n {
        let row_start = row * n;
        fft.process(&mut data[row_start..row_start + n]);
    }

    let mut column = vec![Complex32::new(0.0, 0.0); n];
    for x in 0..n {
        for y in 0..n {
            column[y] = data[y * n + x];
        }

        fft.process(&mut column);

        for y in 0..n {
            data[y * n + x] = column[y];
        }
    }
}

#[cfg_attr(
    target_arch = "x86_64",
    multiversion(targets("x86_64+sse2", "x86_64+avx2", "x86_64+avx512f"))
)]
fn apply_wiener_frequency_domain(freq: &mut [Complex32], ctf: &[f32], wiener_factor: f32) {
    for (idx, val) in freq.iter_mut().enumerate() {
        let ctf_val = ctf[idx];
        let denom = ctf_val * ctf_val + wiener_factor;
        let scale = if denom > 1e-20 { ctf_val / denom } else { 0.0 };
        val.re *= scale;
        val.im *= scale;
    }
}
