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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_buffer_derives_num_elements_and_byte_size() {
        let b = GpuBuffer::new(&[4, 8], DType::F32);
        assert_eq!(b.shape, vec![4, 8]);
        assert_eq!(b.num_elements, 32);
        assert_eq!(b.byte_size, 32 * 4);
        assert_eq!(b.dtype, DType::F32);
        assert_eq!(b.rank(), 2);
    }

    #[test]
    fn gpu_buffer_scalar_has_rank_zero_and_one_element() {
        // Empty-product → 1 element by convention.
        let b = GpuBuffer::new(&[], DType::F32);
        assert_eq!(b.num_elements, 1);
        assert_eq!(b.byte_size, 4);
        assert_eq!(b.rank(), 0);
    }

    #[test]
    fn gpu_buffer_byte_size_respects_dtype_width() {
        assert_eq!(GpuBuffer::new(&[16], DType::F16).byte_size, 16 * 2);
        assert_eq!(GpuBuffer::new(&[16], DType::BF16).byte_size, 16 * 2);
        assert_eq!(GpuBuffer::new(&[16], DType::U8).byte_size, 16);
        assert_eq!(GpuBuffer::new(&[16], DType::I64).byte_size, 16 * 8);
        assert_eq!(GpuBuffer::new(&[16], DType::Bool).byte_size, 16);
    }

    #[test]
    fn gpu_buffer_high_rank() {
        let b = GpuBuffer::new(&[2, 3, 4, 5], DType::I32);
        assert_eq!(b.rank(), 4);
        assert_eq!(b.num_elements, 2 * 3 * 4 * 5);
        assert_eq!(b.byte_size, b.num_elements * 4);
    }

    #[test]
    fn host_data_stores_shape_dtype_and_bytes() {
        let bytes = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let h = HostData::new(&[2, 4], DType::U8, bytes.clone());
        assert_eq!(h.dtype, DType::U8);
        assert_eq!(h.shape, vec![2, 4]);
        assert_eq!(h.data, bytes);
    }

    #[test]
    fn host_data_clone_roundtrip() {
        let h = HostData::new(&[3], DType::F32, vec![0u8; 12]);
        let c = h.clone();
        assert_eq!(c.shape, h.shape);
        assert_eq!(c.data, h.data);
        assert_eq!(c.dtype, h.dtype);
    }
}
