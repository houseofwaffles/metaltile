//! Tensor data: host-side tensor representation for the CPU interpreter.
//!
//! `TensorData` wraps a flat byte buffer with shape and dtype metadata,
//! providing an array-like interface for element access.

use metaltile_core::dtype::DType;

/// A host-side tensor: contiguous row-major storage with shape metadata.
#[derive(Debug, Clone)]
pub struct TensorData {
    /// Element data type.
    pub dtype: DType,
    /// Shape dimensions.
    pub shape: Vec<usize>,
    /// Flat row-major buffer of raw bytes.
    pub data: Vec<u8>,
}

impl TensorData {
    /// Create a new tensor with the given shape and dtype, filled with zeros.
    pub fn zeros(shape: &[usize], dtype: DType) -> Self {
        let num_elements: usize = shape.iter().product();
        let byte_size = num_elements * dtype.size_bytes();
        TensorData { dtype, shape: shape.to_vec(), data: vec![0u8; byte_size] }
    }

    /// Create from an existing byte buffer.
    pub fn from_bytes(shape: &[usize], dtype: DType, data: Vec<u8>) -> Self {
        TensorData { dtype, shape: shape.to_vec(), data }
    }

    /// Number of dimensions (rank).
    pub fn rank(&self) -> usize { self.shape.len() }

    /// Total number of elements.
    pub fn num_elements(&self) -> usize { self.shape.iter().product() }

    /// Byte size of the data buffer.
    pub fn byte_size(&self) -> usize { self.data.len() }

    /// Stride for a given dimension (in elements, not bytes).
    pub fn stride(&self, dim: usize) -> usize { self.shape[dim + 1..].iter().product() }

    /// Row-major linear index from multi-dimensional coordinates.
    pub fn linear_index(&self, coords: &[usize]) -> usize {
        assert_eq!(coords.len(), self.rank());
        coords.iter().zip(self.shape.iter()).fold(0, |acc, (&c, &s)| {
            assert!(c < s, "coordinate {c} out of bounds for dim size {s}");
            acc * s + c
        })
    }

    /// Read a single f32 element (panics if dtype is not f32).
    pub fn read_f32(&self, index: usize) -> f32 {
        assert_eq!(self.dtype, DType::F32);
        let offset = index * 4;
        let bytes: [u8; 4] = self.data[offset..offset + 4].try_into().unwrap();
        f32::from_le_bytes(bytes)
    }

    /// Write a single f32 element.
    pub fn write_f32(&mut self, index: usize, value: f32) {
        assert_eq!(self.dtype, DType::F32);
        let offset = index * 4;
        self.data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    /// Read a single f16 element (as f32).
    pub fn read_f16(&self, index: usize) -> f32 {
        assert_eq!(self.dtype, DType::F16);
        let offset = index * 2;
        let bytes: [u8; 2] = self.data[offset..offset + 2].try_into().unwrap();
        half::f16::from_le_bytes(bytes).to_f32()
    }

    /// Write a single f16 element (from f32).
    pub fn write_f16(&mut self, index: usize, value: f32) {
        assert_eq!(self.dtype, DType::F16);
        let offset = index * 2;
        let f16_val = half::f16::from_f32(value);
        self.data[offset..offset + 2].copy_from_slice(&f16_val.to_le_bytes());
    }

    /// Read a single bf16 element (as f32).
    pub fn read_bf16(&self, index: usize) -> f32 {
        assert_eq!(self.dtype, DType::BF16);
        let offset = index * 2;
        let bits = u16::from_le_bytes(self.data[offset..offset + 2].try_into().unwrap());
        f32::from_bits((bits as u32) << 16)
    }

    /// Write a single bf16 element (from f32, round-to-nearest-even).
    pub fn write_bf16(&mut self, index: usize, value: f32) {
        assert_eq!(self.dtype, DType::BF16);
        let x = value.to_bits();
        let bits = ((x + 0x7FFF + ((x >> 16) & 1)) >> 16) as u16;
        let offset = index * 2;
        self.data[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
    }

    /// Read a single i32 element.
    pub fn read_i32(&self, index: usize) -> i32 {
        assert_eq!(self.dtype, DType::I32);
        let offset = index * 4;
        let bytes: [u8; 4] = self.data[offset..offset + 4].try_into().unwrap();
        i32::from_le_bytes(bytes)
    }

    /// Write a single i32 element.
    pub fn write_i32(&mut self, index: usize, value: i32) {
        assert_eq!(self.dtype, DType::I32);
        let offset = index * 4;
        self.data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    /// Generic read: returns f64 regardless of dtype for numeric comparison.
    pub fn read_scalar(&self, index: usize) -> f64 {
        match self.dtype {
            DType::F32 => self.read_f32(index) as f64,
            DType::F16 => self.read_f16(index) as f64,
            DType::BF16 => self.read_bf16(index) as f64,
            DType::I32 => self.read_i32(index) as f64,
            other => panic!("read_scalar not implemented for {other:?}"),
        }
    }

    /// Generic scalar write from f64.
    pub fn write_scalar(&mut self, index: usize, value: f64) {
        match self.dtype {
            DType::F32 => self.write_f32(index, value as f32),
            DType::F16 => self.write_f16(index, value as f32),
            DType::BF16 => self.write_bf16(index, value as f32),
            DType::I32 => self.write_i32(index, value as i32),
            other => panic!("write_scalar not implemented for {other:?}"),
        }
    }
}
