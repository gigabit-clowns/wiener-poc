use pyo3::prelude::*;
use numpy::{ndarray::Array3, IntoPyArray, PyArray3, PyReadonlyArray1, PyReadonlyArray3};

mod gpu_wiener;
mod cpu_wiener;
mod wiener_common;

use wiener_common::{HostMemoryAllocator, WienerParams};

fn parse_input_tensors(
    images: PyReadonlyArray3<'_, f32>,
    defocus: PyReadonlyArray1<'_, f32>,
) -> PyResult<(Vec<f32>, Vec<f32>, usize, usize)> {
    let image_view = images.as_array();
    let (batch, n, m) = image_view.dim();

    if n != m {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "images must be square per sample. Got shape ({batch}, {n}, {m})"
        )));
    }

    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "box_size must be > 0",
        ));
    }

    let defocus_vec = if let Ok(slice) = defocus.as_slice() {
        slice.to_vec()
    } else {
        defocus.as_array().iter().copied().collect()
    };

    if defocus_vec.len() != batch {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "defocus must have length B (batch). Got {}, expected {}",
            defocus_vec.len(),
            batch
        )));
    }

    let image_vec = image_view.iter().copied().collect::<Vec<_>>();
    Ok((image_vec, defocus_vec, batch, n))
}

fn build_params(
    pixel_size: f32,
    wavelength: f32,
    spherical_aberration: f32,
    q0: f32,
) -> PyResult<WienerParams> {
    if pixel_size <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "pixel_size must be > 0",
        ));
    }

    if wavelength <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "wavelength must be > 0",
        ));
    }

    if spherical_aberration <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "spherical_aberration must be > 0",
        ));
    }

    Ok(WienerParams {
        pixel_size,
        wavelength,
        spherical_aberration,
        q0,
    })
}

#[pyfunction]
#[pyo3(signature = (
    images,
    defocus,
    pixel_size,
    wavelength,
    spherical_aberration,
    q0,
    iterations=1,
    allocator="cudaHostAlloc"
))]
fn run_wiener_gpu<'py>(
    py: Python<'py>,
    images: PyReadonlyArray3<'py, f32>,
    defocus: PyReadonlyArray1<'py, f32>,
    pixel_size: f32,
    wavelength: f32,
    spherical_aberration: f32,
    q0: f32,
    iterations: usize,
    allocator: &str,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let (image_vec, defocus_vec, batch, n) = parse_input_tensors(images, defocus)?;
    let params = build_params(pixel_size, wavelength, spherical_aberration, q0)?;
    let allocator = HostMemoryAllocator::parse(allocator)
        .map_err(|msg| PyErr::new::<pyo3::exceptions::PyValueError, _>(msg))?;

    let result_vec = gpu_wiener::execute_wiener_pipeline_events(
        &image_vec,
        batch,
        n,
        &defocus_vec,
        &params,
        iterations,
        allocator,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("CUDA Error: {e:?}")))?;

    let out = Array3::from_shape_vec((batch, n, n), result_vec).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to reshape GPU output: {e}"
        ))
    })?;

    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (
    images,
    defocus,
    pixel_size,
    wavelength,
    spherical_aberration,
    q0
))]
fn run_wiener_cpu<'py>(
    py: Python<'py>,
    images: PyReadonlyArray3<'py, f32>,
    defocus: PyReadonlyArray1<'py, f32>,
    pixel_size: f32,
    wavelength: f32,
    spherical_aberration: f32,
    q0: f32,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let (image_vec, defocus_vec, batch, n) = parse_input_tensors(images, defocus)?;
    let params = build_params(pixel_size, wavelength, spherical_aberration, q0)?;

    let result_vec = cpu_wiener::apply_wiener_cpu_simd(&image_vec, batch, n, &defocus_vec, &params)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("CPU Error: {e}")))?;

    let out = Array3::from_shape_vec((batch, n, n), result_vec).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to reshape CPU output: {e}"
        ))
    })?;

    Ok(out.into_pyarray(py))
}

#[pymodule]
fn wiener_poc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_wiener_gpu, m)?)?;
    m.add_function(wrap_pyfunction!(run_wiener_cpu, m)?)?;
    Ok(())
}
