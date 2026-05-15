//! Runtime context: Metal device management, pipeline compilation, and kernel dispatch.

#[cfg(target_os = "macos")]
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

use std::{borrow::Cow, collections::BTreeMap, io};

use metaltile_codegen::msl::MslGenerator;
#[cfg(target_os = "macos")]
use metaltile_core::ir::KernelMode;
use metaltile_core::{
    ir::{Kernel, Param, ParamKind},
    shape::Dim,
};

use crate::{autotune::Autotuner, error::MetalTileError};

/// Grid sizing specification for Metal dispatch.
#[derive(Debug, Clone)]
pub enum GridSpec {
    /// 1D elementwise: N total threads.
    Elementwise { n: usize },
    /// Reduction: B threadgroups × T threads.
    Reduction { num_rows: usize, threads_per_group: usize },
    /// 3D grid with explicit dimensions.
    Grid3D { x: usize, y: usize, z: usize, threads_per_group: usize },
}

#[derive(Debug)]
pub struct DispatchResult {
    pub elapsed_us: f64,
    pub gflops: f64,
    /// Output buffer contents keyed by parameter name.
    pub outputs: BTreeMap<String, Vec<u8>>,
}

pub struct Context {
    tuner: Autotuner,
    has_metal: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParamBufferPlan {
    data_binding_index: usize,
    data_len: usize,
}

fn binding_slots(param: &Param) -> usize { if param.kind == ParamKind::Strided { 3 } else { 1 } }

fn static_buffer_len(param: &Param) -> Result<Option<usize>, MetalTileError> {
    let Some(num_elements) = param.shape.num_elements() else {
        return Ok(None);
    };
    num_elements
        .checked_mul(param.dtype.size_bytes())
        .ok_or_else(|| {
            MetalTileError::Buffer(format!("buffer '{}' size overflows usize", param.name))
        })
        .map(Some)
}

fn planned_data_len(
    param: &Param,
    buffers: &BTreeMap<String, Vec<u8>>,
) -> Result<usize, MetalTileError> {
    let provided_len = buffers.get(&param.name).map_or(0, Vec::len);
    let static_len = static_buffer_len(param)?;

    if let Some(expected_len) = static_len {
        if provided_len > 0 && provided_len < expected_len {
            return Err(MetalTileError::Buffer(format!(
                "buffer '{}' has {} bytes, expected at least {}",
                param.name, provided_len, expected_len
            )));
        }

        if param.is_output {
            return Ok(provided_len.max(expected_len));
        }
    }

    Ok(provided_len)
}

fn build_param_buffer_plans(
    kernel: &Kernel,
    buffers: &BTreeMap<String, Vec<u8>>,
) -> Result<Vec<ParamBufferPlan>, MetalTileError> {
    let mut next_binding_index = 0usize;
    let mut plans = Vec::with_capacity(kernel.params.len());
    for param in &kernel.params {
        plans.push(ParamBufferPlan {
            data_binding_index: next_binding_index,
            data_len: planned_data_len(param, buffers)?,
        });
        next_binding_index += binding_slots(param);
    }
    Ok(plans)
}

fn encode_u32s(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn known_shape_dims(param: &Param) -> Result<Option<Vec<u32>>, MetalTileError> {
    let mut dims = Vec::with_capacity(param.shape.rank());
    for dim in param.shape.iter() {
        let Dim::Known(value) = dim else {
            return Ok(None);
        };
        dims.push(u32::try_from(*value).map_err(|_| {
            MetalTileError::Buffer(format!(
                "shape dimension for '{}' exceeds u32: {}",
                param.name, value
            ))
        })?);
    }
    Ok(Some(dims))
}

fn row_major_strides(name: &str, dims: &[u32]) -> Result<Vec<u32>, MetalTileError> {
    let mut strides = vec![1u32; dims.len()];
    let mut stride = 1u32;
    for (idx, &dim) in dims.iter().enumerate().rev() {
        strides[idx] = stride;
        stride = stride.checked_mul(dim).ok_or_else(|| {
            MetalTileError::Buffer(format!("row-major strides for '{}' overflowed u32", name))
        })?;
    }
    Ok(strides)
}

fn resolve_strided_metadata<'a>(
    param: &Param,
    buffers: &'a BTreeMap<String, Vec<u8>>,
) -> Result<(Cow<'a, [u8]>, Cow<'a, [u8]>), MetalTileError> {
    let expected_len = param.shape.rank() * std::mem::size_of::<u32>();
    let defaults = known_shape_dims(param)?
        .map(|dims| {
            let strides = row_major_strides(&param.name, &dims)?;
            Ok::<(Vec<u8>, Vec<u8>), MetalTileError>((encode_u32s(&dims), encode_u32s(&strides)))
        })
        .transpose()?;

    let shape_key = format!("{}_shape", param.name);
    let strides_key = format!("{}_strides", param.name);

    let shape_data = match buffers.get(&shape_key) {
        Some(bytes) => {
            if expected_len > 0 && bytes.len() < expected_len {
                return Err(MetalTileError::Buffer(format!(
                    "buffer '{}' has {} bytes, expected at least {}",
                    shape_key,
                    bytes.len(),
                    expected_len
                )));
            }
            Cow::Borrowed(bytes.as_slice())
        },
        None => {
            let Some((shape_bytes, _)) = defaults.as_ref() else {
                return Err(MetalTileError::Buffer(format!(
                    "missing required strided metadata buffer '{}'",
                    shape_key
                )));
            };
            Cow::Owned(shape_bytes.clone())
        },
    };

    let strides_data = match buffers.get(&strides_key) {
        Some(bytes) => {
            if expected_len > 0 && bytes.len() < expected_len {
                return Err(MetalTileError::Buffer(format!(
                    "buffer '{}' has {} bytes, expected at least {}",
                    strides_key,
                    bytes.len(),
                    expected_len
                )));
            }
            Cow::Borrowed(bytes.as_slice())
        },
        None => {
            let Some((_, strides_bytes)) = defaults.as_ref() else {
                return Err(MetalTileError::Buffer(format!(
                    "missing required strided metadata buffer '{}'",
                    strides_key
                )));
            };
            Cow::Owned(strides_bytes.clone())
        },
    };

    Ok((shape_data, strides_data))
}

impl Context {
    pub fn new() -> Result<Self, MetalTileError> {
        let has_metal = cfg!(target_os = "macos");
        let tuner = Autotuner::new(Autotuner::default_cache_dir(), has_metal);
        Ok(Context { tuner, has_metal })
    }

    pub fn has_gpu(&self) -> bool { self.has_metal }

    pub fn dispatch(&self, kernel: &Kernel) -> Result<DispatchResult, MetalTileError> {
        self.dispatch_with_buffers(kernel, &BTreeMap::new())
    }

    pub fn dispatch_with_buffers(
        &self,
        kernel: &Kernel,
        buffers: &BTreeMap<String, Vec<u8>>,
    ) -> Result<DispatchResult, MetalTileError> {
        self.dispatch_with_options(kernel, buffers, &BTreeMap::new())
    }

    /// Like `dispatch_with_buffers` but also binds Metal function constants (for rope and similar
    /// kernels that use `[[function_constant(N)]]` annotations).
    /// `fn_consts` maps constant name → u32 value.
    pub fn dispatch_with_options(
        &self,
        kernel: &Kernel,
        buffers: &BTreeMap<String, Vec<u8>>,
        fn_consts: &BTreeMap<String, u32>,
    ) -> Result<DispatchResult, MetalTileError> {
        let msl_source = MslGenerator::default().generate(kernel)?;
        if self.has_metal {
            self.dispatch_metal(kernel, &msl_source, buffers, fn_consts)
        } else {
            Ok(DispatchResult { elapsed_us: 0.0, gflops: 0.0, outputs: BTreeMap::new() })
        }
    }

    #[cfg(target_os = "macos")]
    fn dispatch_metal(
        &self,
        kernel: &Kernel,
        msl_source: &str,
        buffers: &BTreeMap<String, Vec<u8>>,
        fn_consts: &BTreeMap<String, u32>,
    ) -> Result<DispatchResult, MetalTileError> {
        use std::{
            collections::HashMap,
            ptr::NonNull,
            sync::{Mutex, OnceLock},
        };

        use objc2::{rc::Retained, runtime::ProtocolObject};
        use objc2_foundation::NSString;
        use objc2_metal::{
            MTLCommandBuffer,
            MTLCommandEncoder,
            MTLCommandQueue,
            MTLComputeCommandEncoder,
            MTLComputePipelineDescriptor,
            MTLComputePipelineState,
            MTLCreateSystemDefaultDevice,
            MTLDevice,
            MTLLibrary,
            MTLPipelineOption,
            MTLResourceOptions,
            MTLSize,
        };

        type Dev = ProtocolObject<dyn objc2_metal::MTLDevice>;
        type Pso = ProtocolObject<dyn MTLComputePipelineState>;

        static DEV: OnceLock<Retained<Dev>> = OnceLock::new();
        static PSO_CACHE: OnceLock<Mutex<HashMap<u64, Retained<Pso>>>> = OnceLock::new();

        let dev = DEV.get_or_init(|| MTLCreateSystemDefaultDevice().unwrap());
        let cache = PSO_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let binding_plans = build_param_buffer_plans(kernel, buffers)?;

        // FNV-1a hash of "<kernel_name>:<msl_source>:<sorted_fn_consts>" as cache key.
        let cache_key = {
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            let hash_bytes = |h: &mut u64, bytes: &[u8]| {
                for &b in bytes {
                    *h ^= b as u64;
                    *h = h.wrapping_mul(0x0100_0000_01b3);
                }
            };
            hash_bytes(&mut h, kernel.name.as_bytes());
            hash_bytes(&mut h, b":");
            hash_bytes(&mut h, msl_source.as_bytes());
            // Include function constants (BTreeMap iterates in sorted order).
            for (name, val) in fn_consts {
                hash_bytes(&mut h, name.as_bytes());
                hash_bytes(&mut h, &val.to_le_bytes());
            }
            h
        };

        let pipe: Retained<Pso> = {
            let mut lock = cache.lock().unwrap();
            if let Some(cached) = lock.get(&cache_key) {
                cached.clone()
            } else {
                let lib = dev
                    .newLibraryWithSource_options_error(&NSString::from_str(msl_source), None)
                    .map_err(|e| MetalTileError::Compilation(format!("{e:?}")))?;
                let fun = if fn_consts.is_empty() {
                    lib.newFunctionWithName(&NSString::from_str(&kernel.name)).ok_or_else(|| {
                        MetalTileError::Compilation(format!("fn '{}' not found", kernel.name))
                    })?
                } else {
                    use objc2_metal::{MTLDataType, MTLFunctionConstantValues};
                    let fcv = MTLFunctionConstantValues::new();
                    for (name, val) in fn_consts {
                        let val_ref: &u32 = val;
                        unsafe {
                            fcv.setConstantValue_type_withName(
                                NonNull::new(val_ref as *const u32 as *mut _).unwrap(),
                                MTLDataType::UInt,
                                &NSString::from_str(name),
                            );
                        }
                    }
                    lib.newFunctionWithName_constantValues_error(
                        &NSString::from_str(&kernel.name),
                        &fcv,
                    )
                    .map_err(|e| {
                        MetalTileError::Compilation(format!(
                            "fn '{}' with constants: {e:?}",
                            kernel.name
                        ))
                    })?
                };
                let desc = MTLComputePipelineDescriptor::new();
                desc.setComputeFunction(Some(&fun));
                let pso = dev
                    .newComputePipelineStateWithDescriptor_options_reflection_error(
                        &desc,
                        MTLPipelineOption(0),
                        None,
                    )
                    .map_err(|e| MetalTileError::Compilation(format!("pipeline: {e:?}")))?;
                lock.insert(cache_key, pso.clone());
                pso
            }
        };

        // Helper: allocate a Metal buffer from a byte slice or a zeroed allocation.
        let alloc_buf = |data: Option<&[u8]>, len: usize| -> Result<_, MetalTileError> {
            let len = len.max(4);
            if let Some(bytes) = data.filter(|bytes| !bytes.is_empty()) {
                if bytes.len() < len {
                    return Err(MetalTileError::Buffer(format!(
                        "buffer allocation expected {len} bytes but received {}",
                        bytes.len()
                    )));
                }
                unsafe {
                    dev.newBufferWithBytes_length_options(
                        NonNull::new(bytes.as_ptr() as *mut _)
                            .ok_or_else(|| MetalTileError::Buffer("null data pointer".into()))?,
                        len,
                        MTLResourceOptions::StorageModeShared,
                    )
                }
                .ok_or(MetalTileError::NoDevice)
            } else {
                dev.newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
                    .ok_or(MetalTileError::NoDevice)
            }
        };

        // Allocate GPU buffers for each param in declaration order.
        // Strided params bind 3 buffers (data, shape, strides); others bind 1.
        let mut metal_bufs = Vec::new();
        let mut n_threads = 1usize;
        for (param, binding) in kernel.params.iter().zip(&binding_plans) {
            let data = buffers.get(&param.name).map(Vec::as_slice);
            if param.is_output {
                let elem_bytes = param.dtype.size_bytes();
                if elem_bytes > 0 {
                    n_threads = n_threads.max(binding.data_len / elem_bytes);
                }
            }
            metal_bufs.push(alloc_buf(data, binding.data_len)?);
            if param.kind == ParamKind::Strided {
                let (shape_data, stride_data) = resolve_strided_metadata(param, buffers)?;
                metal_bufs.push(alloc_buf(Some(shape_data.as_ref()), shape_data.len())?);
                metal_bufs.push(alloc_buf(Some(stride_data.as_ref()), stride_data.len())?);
            }
        }

        let queue = dev.newCommandQueue().ok_or(MetalTileError::NoDevice)?;
        let cb = queue.commandBuffer().ok_or(MetalTileError::NoDevice)?;
        let enc = (&*cb).computeCommandEncoder().ok_or(MetalTileError::NoDevice)?;
        enc.setComputePipelineState(&pipe);
        for (i, buf) in metal_bufs.iter().enumerate() {
            unsafe { enc.setBuffer_offset_atIndex(Some(buf), 0, i) };
        }

        let n_threads = n_threads.max(1);
        let tpg_w = pipe.maxTotalThreadsPerThreadgroup().min(256);
        let (tgs, tpg) = match kernel.mode {
            KernelMode::Reduction => {
                // One threadgroup per row; threads_per_group covers the row width.
                let rows = n_threads.max(1);
                (MTLSize { width: rows, height: 1, depth: 1 }, MTLSize {
                    width: tpg_w,
                    height: 1,
                    depth: 1,
                })
            },
            KernelMode::Grid3D => {
                // Flat 3-D dispatch: treat total as width, H=D=1.
                let groups = (n_threads + tpg_w - 1) / tpg_w;
                (MTLSize { width: groups, height: 1, depth: 1 }, MTLSize {
                    width: tpg_w,
                    height: 1,
                    depth: 1,
                })
            },
            KernelMode::Tile2D => {
                // Square threadgroup tile; total threads is the flat count.
                let tpg_dim = (tpg_w as f64).sqrt() as usize;
                let groups = (n_threads + tpg_dim * tpg_dim - 1) / (tpg_dim * tpg_dim);
                (MTLSize { width: groups, height: 1, depth: 1 }, MTLSize {
                    width: tpg_dim,
                    height: tpg_dim,
                    depth: 1,
                })
            },
            KernelMode::Elementwise => {
                let groups = (n_threads + tpg_w - 1) / tpg_w;
                (MTLSize { width: groups, height: 1, depth: 1 }, MTLSize {
                    width: tpg_w,
                    height: 1,
                    depth: 1,
                })
            },
        };
        enc.dispatchThreadgroups_threadsPerThreadgroup(tgs, tpg);
        (&*enc).endEncoding();
        (&*cb).commit();
        (&*cb).waitUntilCompleted();

        // Read back output buffers.
        let mut outputs: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        for (param, binding) in kernel.params.iter().zip(&binding_plans) {
            if param.is_output && binding.data_len > 0 {
                if let Some(buf) = metal_bufs.get(binding.data_binding_index) {
                    use objc2_metal::MTLBuffer;
                    let ptr = buf.contents();
                    let bytes = unsafe {
                        std::slice::from_raw_parts(ptr.as_ptr() as *const u8, binding.data_len)
                    }
                    .to_vec();
                    outputs.insert(param.name.clone(), bytes);
                }
            }
        }

        let elapsed_us = ((&*cb).GPUEndTime() - (&*cb).GPUStartTime()) * 1_000_000.0;
        Ok(DispatchResult { elapsed_us, gflops: 0.0, outputs })
    }

    #[cfg(not(target_os = "macos"))]
    fn dispatch_metal(
        &self,
        _k: &Kernel,
        _m: &str,
        _b: &BTreeMap<String, Vec<u8>>,
        _fn_consts: &BTreeMap<String, u32>,
    ) -> Result<DispatchResult, MetalTileError> {
        Ok(DispatchResult { elapsed_us: 0.0, gflops: 0.0, outputs: BTreeMap::new() })
    }

    pub fn tuner_mut(&mut self) -> &mut Autotuner { &mut self.tuner }
    pub fn tuner(&self) -> &Autotuner { &self.tuner }

    pub fn shutdown(&self) -> Result<(), io::Error> { self.tuner.flush() }
}

impl Drop for Context {
    fn drop(&mut self) {
        if let Err(err) = self.shutdown() {
            eprintln!("metaltile-runtime: failed to flush autotuner cache on drop: {err}");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    use metaltile_core::{dtype::DType, shape::Shape};

    use super::*;

    fn tensor_param(
        name: &str,
        dtype: DType,
        dims: &[usize],
        is_output: bool,
        kind: ParamKind,
    ) -> Param {
        Param {
            name: name.into(),
            dtype,
            shape: Shape::new(dims.iter().copied().map(Dim::Known)),
            is_output,
            kind,
        }
    }

    fn unique_path(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before epoch")
            .as_nanos();
        std::env::temp_dir()
            .join(format!("metaltile-runtime-{label}-{}-{nonce}", std::process::id()))
    }

    #[test]
    fn buffer_plans_follow_binding_indices_and_static_output_sizes() {
        let mut kernel = Kernel::new("binding_plan_test");
        kernel.params = vec![
            tensor_param("input", DType::F32, &[2, 2], false, ParamKind::Strided),
            tensor_param("out_a", DType::F32, &[4], true, ParamKind::Tensor),
            tensor_param("out_b", DType::F32, &[4], true, ParamKind::Tensor),
        ];

        let mut buffers = BTreeMap::new();
        buffers.insert("input".into(), vec![0u8; 16]);
        buffers.insert("out_b".into(), vec![0u8; 16]);

        let plans = build_param_buffer_plans(&kernel, &buffers).expect("buffer plans");
        assert_eq!(plans, vec![
            ParamBufferPlan { data_binding_index: 0, data_len: 16 },
            ParamBufferPlan { data_binding_index: 3, data_len: 16 },
            ParamBufferPlan { data_binding_index: 4, data_len: 16 },
        ]);
    }

    #[test]
    fn strided_metadata_defaults_to_row_major_shape_and_strides() {
        let param = tensor_param("input", DType::F32, &[2, 3, 4], false, ParamKind::Strided);
        let buffers = BTreeMap::new();

        let (shape_data, stride_data) =
            resolve_strided_metadata(&param, &buffers).expect("default strided metadata");

        assert_eq!(shape_data.as_ref(), encode_u32s(&[2, 3, 4]).as_slice());
        assert_eq!(stride_data.as_ref(), encode_u32s(&[12, 4, 1]).as_slice());
    }

    #[test]
    fn shutdown_returns_flush_errors() {
        let bad_cache_root = unique_path("flush-error");
        fs::write(&bad_cache_root, b"not a directory").expect("create sentinel file");

        let mut ctx =
            Context { tuner: Autotuner::new(bad_cache_root.clone(), false), has_metal: false };
        let err = ctx.shutdown().expect_err("shutdown should surface flush errors");
        assert!(matches!(
            err.kind(),
            io::ErrorKind::AlreadyExists | io::ErrorKind::NotADirectory | io::ErrorKind::Other
        ));

        let ok_cache_root = unique_path("flush-ok");
        fs::create_dir_all(&ok_cache_root).expect("create recovery cache dir");
        ctx.tuner = Autotuner::new(ok_cache_root.clone(), false);
        ctx.shutdown().expect("recovered shutdown should succeed");

        fs::remove_file(&bad_cache_root).expect("remove sentinel file");
        let _ = fs::remove_file(ok_cache_root.join("tuning_cache.json"));
        let _ = fs::remove_dir_all(ok_cache_root);
    }
}
