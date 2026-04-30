use pyo3::prelude::*;

/// Test function to validate Python-Rust connection
#[pyfunction]
fn check_system() -> String {
    "Systems online! Rust & Python are connected.".to_string()
}

/// This module must be called EXCATLY as the 'name' in Cargo.toml
#[pymodule]
fn wiener_poc(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(check_system, m)?)?;
    Ok(())
}
