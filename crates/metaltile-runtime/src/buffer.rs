//! GPU buffer management: allocation, transfer, and lifetime tracking.
//!
//! Buffers wrap Metal's MTLBuffer with typed metadata and provide
//! Rust-native read/write for host↔device transfers.

use metaltile_core::dtype::DType;

/// A GPU buffer with shape and dtype metadata.
#[derive(Debug)]
pub struct GpuBuffer {
    /// Element data type.
    pub dtype: DType,
    /// Shape dimensions of the tensor (row-major).
    pub shape: Vec<usize>,
    /// Number of elements.
    pub num_elements: usize,
    /// Byte size.
    pub byte_size: usize,
}

impl GpuBuffer {
    /// Create buffer metadata describing a tensor on the GPU.
    pub fn new(shape: &[usize], dtype: DType) -> Self {
        let num_elements: usize = shape.iter().product();
        GpuBuffer {
            dtype,
            shape: shape.to_vec(),
            num_elements,
            byte_size: num_elements * dtype.size_bytes(),
        }
    }

    /// Rank of the tensor.
    pub fn rank(&self) -> usize { self.shape.len() }
}

/// Host-side data ready for upload to GPU.
#[derive(Debug, Clone)]
pub struct HostData {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl HostData {
    pub fn new(shape: &[usize], dtype: DType, data: Vec<u8>) -> Self {
        HostData { dtype, shape: shape.to_vec(), data }
    }
}
