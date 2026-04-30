use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use numpy::{PyReadonlyArray1, IntoPyArray};

// Delcare gpu_wiener.rs as part of this module
mod gpu_wiener;

/// This function will be visible to Python
#[pyfunction]
fn run_wiener_gpu<'py>(py: Python<'py>, data: PyReadonlyArray1<'py, f32>) -> PyResult<Bound<'py, numpy::PyArray1<f32>>> {
    // as_slice() gives us access to NumPy memory without copying it
    let input_slice = data.as_slice().unwrap();
    
    // Call CUDA orchestration
    match gpu_wiener::execute_wiener_pipeline(input_slice) {
        Ok(result_vec) => {
            // Convert Rust's result vector into a NumPy array
            Ok(result_vec.into_pyarray(py))
        },
        Err(e) => {
            // If CUDA crashes, we throw an Python-native exception
            Err(PyRuntimeError::new_err(format!("CUDA Driver Error: {:?}", e)))
        }
    }
}

/// This module must be called EXCATLY as the 'name' in Cargo.toml
#[pymodule]
fn wiener_poc(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_wiener_gpu, m)?)?;
    Ok(())
}
