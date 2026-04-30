use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};

mod gpu_wiener;
mod cpu_wiener;

#[pyfunction]
fn run_wiener_gpu<'py>(py: Python<'py>, array: PyReadonlyArray1<f32>) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let data_slice = array.as_slice().unwrap();
    let result_vec = gpu_wiener::execute_wiener_pipeline_events(data_slice).unwrap();
    Ok(result_vec.into_pyarray(py))
}

#[pyfunction]
fn run_wiener_cpu<'py>(py: Python<'py>, array: PyReadonlyArray1<f32>) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let mut data_vec = array.as_slice().unwrap().to_vec();
    // Llamamos a la versión SIMD multiversionada
    cpu_wiener::apply_wiener_cpu_simd(&mut data_vec);
    Ok(data_vec.into_pyarray(py))
}

#[pymodule]
fn wiener_poc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_wiener_gpu, m)?)?;
    m.add_function(wrap_pyfunction!(run_wiener_cpu, m)?)?;
    Ok(())
}
