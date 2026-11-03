//! Owned snapshots with allocation suited to complete, dense matrix copies.

use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;

pub(super) enum DenseStorage {
    Heap(Vec<f64>),
    #[cfg(target_os = "linux")]
    Mapped(memmap2::MmapMut),
}

impl DenseStorage {
    pub(super) fn copy_from(buffer: &PyBuffer<f64>, py: Python<'_>) -> PyResult<Self> {
        #[cfg(target_os = "linux")]
        if let Some(mut mapping) = dense_mapping(buffer.len_bytes()) {
            buffer.copy_to_fortran_slice(py, bytemuck::cast_slice_mut(&mut mapping))?;
            return Ok(Self::Mapped(mapping));
        }
        Ok(Self::Heap(buffer.to_fortran_vec(py)?))
    }

    pub(super) fn as_slice(&self) -> &[f64] {
        match self {
            Self::Heap(columns) => columns,
            #[cfg(target_os = "linux")]
            Self::Mapped(mapping) => bytemuck::cast_slice(mapping),
        }
    }
}

#[cfg(target_os = "linux")]
fn dense_mapping(bytes: usize) -> Option<memmap2::MmapMut> {
    // Every byte will be copied and read. Huge-page advice reduces page faults
    // for large snapshots without changing the host's allocation policy.
    if bytes < 4 * 1024 * 1024 {
        return None;
    }
    let mapping = memmap2::MmapMut::map_anon(bytes).ok()?;
    bytemuck::try_cast_slice::<u8, f64>(&mapping).ok()?;
    mapping.advise(memmap2::Advice::HugePage).ok()?;
    Some(mapping)
}
