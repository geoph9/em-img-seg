use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod imageio;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn test(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn butwhy(a: usize, b: usize) -> PyResult<String> {
    Ok((a - b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn rustem(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(test, m)?)?;
    m.add_function(wrap_pyfunction!(butwhy, m)?)?;

    Ok(())
}