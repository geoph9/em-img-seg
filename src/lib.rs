use pyo3::prelude::*;

pub mod em;
pub mod imageio;

/// A Python module implemented in Rust.
#[pymodule]
fn rustem(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<em::EM>()?;

    Ok(())
}