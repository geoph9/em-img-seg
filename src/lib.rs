use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

// pub mod imageio;
pub mod em;

/// A Python module implemented in Rust.
#[pymodule]
fn rustem(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<em::EM>()?;

    Ok(())
}