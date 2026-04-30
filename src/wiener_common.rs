use std::f32::consts::PI;

#[derive(Clone, Copy, Debug)]
pub struct WienerParams {
    pub pixel_size: f32,
    pub wavelength: f32,
    pub spherical_aberration: f32,
    pub q0: f32,
}

#[derive(Clone, Copy, Debug)]
pub enum HostMemoryAllocator {
    CudaHostAlloc,
}

impl HostMemoryAllocator {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "cudaHostAlloc" => Ok(Self::CudaHostAlloc),
            _ => Err(format!(
                "Unsupported allocator '{value}'. Allowed values: cudaHostAlloc"
            )),
        }
    }
}

pub fn frequency_component(idx: usize, n: usize, pixel_size: f32) -> f32 {
    let half = n / 2;
    let shifted = if idx <= half {
        idx as i32
    } else {
        idx as i32 - n as i32
    };
    shifted as f32 / ((n as f32) * pixel_size)
}

pub fn build_frequency2_grid_2d(n: usize, pixel_size: f32) -> Vec<f32> {
    let mut k2 = vec![0.0f32; n * n];

    for y in 0..n {
        let fy = frequency_component(y, n, pixel_size);
        let fy2 = fy * fy;

        for x in 0..n {
            let fx = frequency_component(x, n, pixel_size);
            let fx2 = fx * fx;
            k2[y * n + x] = fx2 + fy2;
        }
    }

    k2
}

pub fn build_frequency2_grid_rfft_2d(n: usize, pixel_size: f32) -> Vec<f32> {
    let nx = n / 2 + 1;
    let mut k2 = vec![0.0f32; n * nx];

    for y in 0..n {
        let fy = frequency_component(y, n, pixel_size);
        let fy2 = fy * fy;

        for x in 0..nx {
            let fx = (x as f32) / ((n as f32) * pixel_size);
            let fx2 = fx * fx;
            k2[y * nx + x] = fx2 + fy2;
        }
    }

    k2
}

pub fn compute_ctf_values(k2_grid: &[f32], defocus: f32, params: &WienerParams) -> Vec<f32> {
    let wavelength2 = params.wavelength * params.wavelength;

    k2_grid
        .iter()
        .map(|&k2| {
            let angle = PI
                * params.wavelength
                * k2
                * (0.5 * params.spherical_aberration * wavelength2 * k2 + defocus);
            angle.sin() - params.q0 * angle.cos()
        })
        .collect()
}

pub fn compute_wiener_factor_from_ctf(ctf: &[f32]) -> f32 {
    let ctf2_sum: f32 = ctf.iter().map(|v| v * v).sum();
    0.1 * (ctf2_sum / (ctf.len() as f32))
}

pub fn compute_wiener_factors(defocus: &[f32], n: usize, params: &WienerParams) -> Vec<f32> {
    let k2_grid = build_frequency2_grid_rfft_2d(n, params.pixel_size);

    defocus
        .iter()
        .map(|&d| {
            let ctf = compute_ctf_values(&k2_grid, d, params);
            compute_wiener_factor_from_ctf(&ctf)
        })
        .collect()
}
