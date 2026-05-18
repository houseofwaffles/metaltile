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

/// One pass in a fused dispatch chain. See [`Context::dispatch_chain`].
///
/// Buffers reused across consecutive passes (output of pass i → input
/// of pass j>i) are auto-aliased to a single Metal allocation in
/// `MTLStorageModePrivate` — they never leave GPU memory.
///
/// Pass [`Context::upload_resident`]-produced handles in `resident`
/// to bind a pre-uploaded Metal buffer for a parameter by name; the
/// dispatch then skips the per-call alloc + memcpy for that input.
/// Holding the [`ResidentBuffer`] across iterations keeps the bytes
/// GPU-resident.
pub struct DispatchSpec<'a> {
    pub kernel: &'a Kernel,
    pub buffers: &'a BTreeMap<String, Vec<u8>>,
    pub fn_consts: &'a BTreeMap<String, u32>,
    pub grid_groups: [usize; 3],
    pub threads_per_group: [usize; 3],
    pub resident: &'a BTreeMap<String, ResidentBuffer>,
}

/// Opaque handle to a GPU-resident input buffer. Produced by
/// [`Context::upload_resident`]; pass via [`DispatchSpec::resident`]
/// to bind without per-call alloc + host memcpy.
///
/// Cloning is cheap (Rc::clone) and shares the underlying Metal
/// buffer. The buffer returns to the dispatch buffer pool when the
/// last clone drops.
#[derive(Clone)]
pub struct ResidentBuffer {
    #[cfg(target_os = "macos")]
    inner: std::rc::Rc<
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    >,
    #[cfg(not(target_os = "macos"))]
    _stub: (),
}

#[cfg(target_os = "macos")]
type BufRc =
    std::rc::Rc<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>>;

#[cfg(target_os = "macos")]
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

#[cfg(target_os = "macos")]
fn fnv1a_extend(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h ^= b as u64;
        *h = h.wrapping_mul(0x0100_0000_01b3);
    }
}
#[cfg(target_os = "macos")]
type PoolKey = (usize, u64);

// Thread-local Metal buffer pool. Bucketed by (next_pow_of_two(size),
// storage_mode); each bucket holds Rc-wrapped buffers. `acquire`
// returns one whose strong_count == 1 (only pool owns it), else
// allocates. thread_local because Retained<MTLBuffer> isn't Send.
#[cfg(target_os = "macos")]
std::thread_local! {
    // FxHashMap over HashMap because PoolKey is (usize, u64) — already
    // densely numeric, SipHash would just shuffle bits that don't need it.
    static BUF_POOL: std::cell::RefCell<rustc_hash::FxHashMap<PoolKey, Vec<BufRc>>>
        = std::cell::RefCell::new(rustc_hash::FxHashMap::default());
}

#[cfg(target_os = "macos")]
fn pool_acquire(
    dev: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
    len: usize,
    opts: objc2_metal::MTLResourceOptions,
) -> Result<BufRc, MetalTileError> {
    use objc2_metal::MTLDevice;
    let bucket = len.max(4).next_power_of_two();
    let key: PoolKey = (bucket, opts.0 as u64);
    BUF_POOL.with(|cell| {
        let mut p = cell.borrow_mut();
        let slot = p.entry(key).or_default();
        for buf in slot.iter() {
            if std::rc::Rc::strong_count(buf) == 1 {
                return Ok(buf.clone());
            }
        }
        let new = std::rc::Rc::new(
            dev.newBufferWithLength_options(bucket, opts).ok_or(MetalTileError::NoDevice)?,
        );
        slot.push(new.clone());
        Ok(new)
    })
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

#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
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

#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
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

#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
fn encode_u32s(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
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

#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
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

#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
type StridedMetadata<'a> = (Cow<'a, [u8]>, Cow<'a, [u8]>);

#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
fn resolve_strided_metadata<'a>(
    param: &Param,
    buffers: &'a BTreeMap<String, Vec<u8>>,
) -> Result<StridedMetadata<'a>, MetalTileError> {
    let expected_len = param.shape.rank() * std::mem::size_of::<u32>();
    let defaults = known_shape_dims(param)?
        .map(|dims| {
            let strides = row_major_strides(&param.name, &dims)?;
            Ok::<(Vec<u8>, Vec<u8>), MetalTileError>((encode_u32s(&dims), encode_u32s(&strides)))
        })
        .transpose()?;

    // Single key buffer, reused across the two lookups: allocate once
    // (capacity = name + "_strides" — the longer suffix) and rewrite the
    // suffix in place. Replaces two `format!` allocations per strided
    // param per dispatch.
    let mut key = String::with_capacity(param.name.len() + 8);
    key.push_str(&param.name);
    let prefix_len = key.len();
    key.push_str("_shape");

    let shape_data = match buffers.get(&key) {
        Some(bytes) => {
            if expected_len > 0 && bytes.len() < expected_len {
                return Err(MetalTileError::Buffer(format!(
                    "buffer '{}' has {} bytes, expected at least {}",
                    key,
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
                    key
                )));
            };
            Cow::Owned(shape_bytes.clone())
        },
    };

    key.truncate(prefix_len);
    key.push_str("_strides");

    let strides_data = match buffers.get(&key) {
        Some(bytes) => {
            if expected_len > 0 && bytes.len() < expected_len {
                return Err(MetalTileError::Buffer(format!(
                    "buffer '{}' has {} bytes, expected at least {}",
                    key,
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
                    key
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
            self.dispatch_metal(kernel, &msl_source, buffers, fn_consts, None)
        } else {
            Ok(DispatchResult { elapsed_us: 0.0, gflops: 0.0, outputs: BTreeMap::new() })
        }
    }

    /// Like `dispatch_with_options` but lets the caller specify the
    /// dispatch grid explicitly. Use when the auto-derived grid (from
    /// output buffer size + `kernel.mode`) doesn't fit — e.g. a
    /// reduction kernel that needs one threadgroup per Q head with a
    /// fixed thread count rather than per output-element.
    ///
    /// `grid_groups` is the number of threadgroups along each axis;
    /// `threads_per_group` is the size of each threadgroup. Both are
    /// `[x, y, z]`.
    pub fn dispatch_with_grid(
        &self,
        kernel: &Kernel,
        buffers: &BTreeMap<String, Vec<u8>>,
        fn_consts: &BTreeMap<String, u32>,
        grid_groups: [usize; 3],
        threads_per_group: [usize; 3],
    ) -> Result<DispatchResult, MetalTileError> {
        let msl_source = MslGenerator::default().generate(kernel)?;
        if self.has_metal {
            self.dispatch_metal(
                kernel,
                &msl_source,
                buffers,
                fn_consts,
                Some((grid_groups, threads_per_group)),
            )
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
        // When `Some`, overrides the auto-derived grid: `(groups, threads_per_group)`.
        grid_override: Option<([usize; 3], [usize; 3])>,
    ) -> Result<DispatchResult, MetalTileError> {
        use std::{ptr::NonNull, sync::OnceLock};

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
        use parking_lot::Mutex;
        use rustc_hash::FxHashMap;

        type Dev = ProtocolObject<dyn objc2_metal::MTLDevice>;
        type Pso = ProtocolObject<dyn MTLComputePipelineState>;
        type Queue = ProtocolObject<dyn MTLCommandQueue>;

        static DEV: OnceLock<Retained<Dev>> = OnceLock::new();
        // FxHashMap over HashMap because the key is a pre-hashed FNV-1a
        // u64 — SipHash13 over already-hashed bits is pure waste.
        static PSO_CACHE: OnceLock<Mutex<FxHashMap<u64, Retained<Pso>>>> = OnceLock::new();
        // Persist the command queue: Apple's Best Practices Guide flags
        // newCommandQueue as expensive ("should not be repeatedly created
        // and destroyed"). Per-dispatch construction was costing 10-50µs.
        static QUEUE: OnceLock<Retained<Queue>> = OnceLock::new();

        let dev = DEV.get_or_init(|| {
            MTLCreateSystemDefaultDevice().expect("MTLCreateSystemDefaultDevice returned nil")
        });
        let cache = PSO_CACHE.get_or_init(|| Mutex::new(FxHashMap::default()));
        let queue = QUEUE.get_or_init(|| dev.newCommandQueue().expect("newCommandQueue failed"));
        let binding_plans = build_param_buffer_plans(kernel, buffers)?;

        let cache_key = {
            let mut h = FNV_OFFSET;
            fnv1a_extend(&mut h, kernel.name.as_bytes());
            fnv1a_extend(&mut h, b":");
            fnv1a_extend(&mut h, msl_source.as_bytes());
            for (name, val) in fn_consts {
                fnv1a_extend(&mut h, name.as_bytes());
                fnv1a_extend(&mut h, &val.to_le_bytes());
            }
            h
        };

        let pipe: Retained<Pso> = {
            let mut lock = cache.lock();
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
                                NonNull::new(val_ref as *const u32 as *mut _)
                                    .expect("u32 ref always non-null"),
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

        let mut metal_bufs = Vec::with_capacity(kernel.params.len() * 2);
        let mut n_threads = 1usize;
        for (param, binding) in kernel.params.iter().zip(&binding_plans) {
            let data = buffers.get(&param.name).map(Vec::as_slice);
            if param.is_output {
                let elem_bytes = param.dtype.size_bytes();
                if let Some(quot) = binding.data_len.checked_div(elem_bytes) {
                    n_threads = n_threads.max(quot);
                }
            }
            metal_bufs.push(alloc_buf(data, binding.data_len)?);
            if param.kind == ParamKind::Strided {
                let (shape_data, stride_data) = resolve_strided_metadata(param, buffers)?;
                metal_bufs.push(alloc_buf(Some(shape_data.as_ref()), shape_data.len())?);
                metal_bufs.push(alloc_buf(Some(stride_data.as_ref()), stride_data.len())?);
            }
        }

        // Bind constexpr scalar buffers in the IR-declared order. Codegen
        // emits `[[buffer(N)]]` for each constexpr at indices immediately
        // after the tensor params, so the binding order has to mirror
        // `kernel.constexprs`. Each constexpr is a single scalar (4 or 8
        // bytes depending on dtype); the caller supplies the encoded bytes
        // in `buffers` keyed by the constexpr name.
        for decl in &kernel.constexprs {
            let key = decl.name.name();
            let elem = decl.dtype.size_bytes().max(4);
            let bytes = buffers.get(key).map(Vec::as_slice);
            metal_bufs.push(alloc_buf(bytes, elem)?);
        }

        let cb = queue.commandBuffer().ok_or(MetalTileError::NoDevice)?;
        let enc = (*cb).computeCommandEncoder().ok_or(MetalTileError::NoDevice)?;
        enc.setComputePipelineState(&pipe);
        for (i, buf) in metal_bufs.iter().enumerate() {
            unsafe { enc.setBuffer_offset_atIndex(Some(buf), 0, i) };
        }

        let n_threads = n_threads.max(1);
        let tpg_w = pipe.maxTotalThreadsPerThreadgroup().min(256);
        let (tgs, tpg) = match grid_override {
            Some((g, t)) => (MTLSize { width: g[0], height: g[1], depth: g[2] }, MTLSize {
                width: t[0],
                height: t[1],
                depth: t[2],
            }),
            None => match kernel.mode {
                KernelMode::Reduction => {
                    let rows = n_threads.max(1);
                    (MTLSize { width: rows, height: 1, depth: 1 }, MTLSize {
                        width: tpg_w,
                        height: 1,
                        depth: 1,
                    })
                },
                KernelMode::Grid3D => {
                    let groups = n_threads.div_ceil(tpg_w);
                    (MTLSize { width: groups, height: 1, depth: 1 }, MTLSize {
                        width: tpg_w,
                        height: 1,
                        depth: 1,
                    })
                },
                KernelMode::Tile2D => {
                    let tpg_dim = (tpg_w as f64).sqrt() as usize;
                    let groups = n_threads.div_ceil(tpg_dim * tpg_dim);
                    (MTLSize { width: groups, height: 1, depth: 1 }, MTLSize {
                        width: tpg_dim,
                        height: tpg_dim,
                        depth: 1,
                    })
                },
                KernelMode::Elementwise => {
                    let groups = n_threads.div_ceil(tpg_w);
                    (MTLSize { width: groups, height: 1, depth: 1 }, MTLSize {
                        width: tpg_w,
                        height: 1,
                        depth: 1,
                    })
                },
                // SimdGroup2D: tiled matmul. Threadgroup = WM×WN×32.
                // For bench dispatch: one threadgroup, full threadgroup size.
                KernelMode::SimdGroup2D => (MTLSize { width: 1, height: 1, depth: 1 }, MTLSize {
                    width: tpg_w,
                    height: 1,
                    depth: 1,
                }),
            },
        };
        enc.dispatchThreadgroups_threadsPerThreadgroup(tgs, tpg);
        (*enc).endEncoding();
        (*cb).commit();
        (*cb).waitUntilCompleted();

        let mut outputs: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        for (param, binding) in kernel.params.iter().zip(&binding_plans) {
            if param.is_output
                && binding.data_len > 0
                && let Some(buf) = metal_bufs.get(binding.data_binding_index)
            {
                use objc2_metal::MTLBuffer;
                let ptr = buf.contents();
                let bytes = unsafe {
                    std::slice::from_raw_parts(ptr.as_ptr() as *const u8, binding.data_len)
                }
                .to_vec();
                outputs.insert(param.name.clone(), bytes);
            }
        }

        let elapsed_us = ((*cb).GPUEndTime() - (*cb).GPUStartTime()) * 1_000_000.0;
        Ok(DispatchResult { elapsed_us, gflops: 0.0, outputs })
    }

    #[cfg(not(target_os = "macos"))]
    fn dispatch_metal(
        &self,
        _k: &Kernel,
        _m: &str,
        _b: &BTreeMap<String, Vec<u8>>,
        _fn_consts: &BTreeMap<String, u32>,
        _grid_override: Option<([usize; 3], [usize; 3])>,
    ) -> Result<DispatchResult, MetalTileError> {
        Ok(DispatchResult { elapsed_us: 0.0, gflops: 0.0, outputs: BTreeMap::new() })
    }

    /// Acquire a pool-managed Metal buffer in `MTLStorageModeShared`,
    /// memcpy `bytes` into it, and return an opaque handle. Pass the
    /// handle via [`DispatchSpec::resident`] to bind without per-call
    /// alloc + memcpy. The buffer stays GPU-resident as long as any
    /// clone of the [`ResidentBuffer`] exists; on the last drop it
    /// returns to the pool.
    pub fn upload_resident(&self, bytes: &[u8]) -> Result<ResidentBuffer, MetalTileError> {
        #[cfg(target_os = "macos")]
        {
            if !self.has_metal {
                return Err(MetalTileError::NoDevice);
            }
            use std::sync::OnceLock;

            use objc2::{rc::Retained, runtime::ProtocolObject};
            use objc2_metal::{
                MTLBuffer,
                MTLCreateSystemDefaultDevice,
                MTLDevice,
                MTLResourceOptions,
            };
            type Dev = ProtocolObject<dyn MTLDevice>;
            static DEV: OnceLock<Retained<Dev>> = OnceLock::new();
            let dev = DEV.get_or_init(|| {
                MTLCreateSystemDefaultDevice().expect("MTLCreateSystemDefaultDevice returned nil")
            });
            let opts = MTLResourceOptions::StorageModeShared
                | MTLResourceOptions::HazardTrackingModeUntracked;
            let buf = pool_acquire(dev, bytes.len(), opts)?;
            let dst = buf.contents();
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst.as_ptr() as *mut u8, bytes.len());
            }
            Ok(ResidentBuffer { inner: buf })
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = bytes;
            Ok(ResidentBuffer { _stub: () })
        }
    }

    /// Dispatch N kernel passes through a single Metal command buffer.
    ///
    /// Buffers referenced as outputs of pass i and as inputs of any
    /// later pass are allocated **once** as `MTLStorageModePrivate` and
    /// shared across passes — they never round-trip through host RAM.
    /// Pass-to-pass ordering is enforced with a `memoryBarrierWithScope`
    /// between consecutive encoders. Only buffers consumed by no later
    /// pass are read back at the end.
    ///
    /// For a 2-pass SDPA decode this replaces two separate cmd-buffer
    /// commits + a ~MB-sized host memcpy of `partial_o/m/l` with one
    /// commit and zero host traffic between passes.
    pub fn dispatch_chain(
        &self,
        specs: &[DispatchSpec<'_>],
    ) -> Result<Vec<DispatchResult>, MetalTileError> {
        if specs.is_empty() {
            return Ok(Vec::new());
        }
        #[cfg(target_os = "macos")]
        if self.has_metal {
            return self.dispatch_chain_metal(specs);
        }
        Ok(specs
            .iter()
            .map(|_| DispatchResult { elapsed_us: 0.0, gflops: 0.0, outputs: BTreeMap::new() })
            .collect())
    }

    #[cfg(target_os = "macos")]
    fn dispatch_chain_metal(
        &self,
        specs: &[DispatchSpec<'_>],
    ) -> Result<Vec<DispatchResult>, MetalTileError> {
        use std::{collections::HashSet, ptr::NonNull, sync::OnceLock};

        use objc2::{rc::Retained, runtime::ProtocolObject};
        use objc2_foundation::NSString;
        use objc2_metal::{
            MTLBarrierScope,
            MTLBuffer,
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
        use parking_lot::Mutex;
        use rustc_hash::FxHashMap;

        type Dev = ProtocolObject<dyn objc2_metal::MTLDevice>;
        type Pso = ProtocolObject<dyn MTLComputePipelineState>;
        type Queue = ProtocolObject<dyn MTLCommandQueue>;

        static DEV: OnceLock<Retained<Dev>> = OnceLock::new();
        static PSO_CACHE: OnceLock<Mutex<FxHashMap<u64, Retained<Pso>>>> = OnceLock::new();
        static QUEUE: OnceLock<Retained<Queue>> = OnceLock::new();

        let dev = DEV.get_or_init(|| {
            MTLCreateSystemDefaultDevice().expect("MTLCreateSystemDefaultDevice returned nil")
        });
        let cache = PSO_CACHE.get_or_init(|| Mutex::new(FxHashMap::default()));
        let queue = QUEUE.get_or_init(|| dev.newCommandQueue().expect("newCommandQueue failed"));

        let acquire_shared = |bytes: Option<&[u8]>, len: usize| -> Result<BufRc, MetalTileError> {
            let opts = MTLResourceOptions::StorageModeShared
                | MTLResourceOptions::HazardTrackingModeUntracked;
            let buf = pool_acquire(dev, len, opts)?;
            if let Some(b) = bytes.filter(|b| !b.is_empty()) {
                if b.len() < len {
                    return Err(MetalTileError::Buffer(format!(
                        "buffer expected {len} bytes, got {}",
                        b.len()
                    )));
                }
                let dst = buf.contents();
                unsafe {
                    std::ptr::copy_nonoverlapping(b.as_ptr(), dst.as_ptr() as *mut u8, b.len());
                }
            }
            Ok(buf)
        };
        let acquire_private = |len: usize| -> Result<BufRc, MetalTileError> {
            pool_acquire(
                dev,
                len,
                MTLResourceOptions::StorageModePrivate
                    | MTLResourceOptions::HazardTrackingModeUntracked,
            )
        };

        // Precompute MSL + binding plans per spec; collect param names
        // consumed as INPUT by any later spec — those drive private-storage
        // aliasing for intermediate buffers.
        let mut msl_sources: Vec<String> = Vec::with_capacity(specs.len());
        let mut binding_plans: Vec<Vec<ParamBufferPlan>> = Vec::with_capacity(specs.len());
        // MSL cache. Codegen passes (vectorize, fusion, …) cost tens of
        // µs per kernel — at short context (n_kv ≤ 1K) the total iter
        // time is in the 40 µs range so re-running passes per dispatch
        // is a significant fraction. Cache keyed by the same FNV hash
        // we use for PSO lookup (kernel name + sorted fn_consts +
        // first param dtype) and check it before running the pipeline.
        static MSL_CACHE: OnceLock<Mutex<FxHashMap<u64, String>>> = OnceLock::new();
        let msl_cache = MSL_CACHE.get_or_init(|| Mutex::new(FxHashMap::default()));
        for spec in specs {
            let mut h = FNV_OFFSET;
            fnv1a_extend(&mut h, spec.kernel.name.as_bytes());
            fnv1a_extend(&mut h, b":");
            // First tensor param's dtype distinguishes f32/f16/bf16 specializations.
            if let Some(p) = spec.kernel.params.first() {
                fnv1a_extend(&mut h, &(p.dtype.size_bytes() as u64).to_le_bytes());
            }
            for (n, v) in spec.fn_consts {
                fnv1a_extend(&mut h, n.as_bytes());
                fnv1a_extend(&mut h, &v.to_le_bytes());
            }
            // Drop the read guard BEFORE the match — parking_lot::Mutex isn't
            // reentrant, and temporaries in a match scrutinee live until the
            // end of the match body (RFC 66), so writing back inside None
            // would deadlock against the still-held read guard.
            let cached = msl_cache.lock().get(&h).cloned();
            let msl = match cached {
                Some(m) => m,
                None => {
                    let generated = MslGenerator::default().generate(spec.kernel)?;
                    msl_cache.lock().insert(h, generated.clone());
                    generated
                },
            };
            msl_sources.push(msl);
            binding_plans.push(build_param_buffer_plans(spec.kernel, spec.buffers)?);
        }
        let later_inputs: Vec<HashSet<&str>> = (0..specs.len())
            .map(|i| {
                specs[i + 1..]
                    .iter()
                    .flat_map(|s| s.kernel.params.iter())
                    .filter(|p| !p.is_output)
                    .map(|p| p.name.as_str())
                    .collect()
            })
            .collect();

        // Persistent name→buffer map: outputs from earlier specs go here
        // (private storage) so later specs find them by name and skip alloc.
        let mut alias_pool: FxHashMap<String, BufRc> = FxHashMap::default();

        let mut pipes: Vec<Retained<Pso>> = Vec::with_capacity(specs.len());
        for (spec, msl) in specs.iter().zip(msl_sources.iter()) {
            let cache_key = {
                let mut h = FNV_OFFSET;
                fnv1a_extend(&mut h, spec.kernel.name.as_bytes());
                fnv1a_extend(&mut h, b":");
                fnv1a_extend(&mut h, msl.as_bytes());
                for (n, v) in spec.fn_consts {
                    fnv1a_extend(&mut h, n.as_bytes());
                    fnv1a_extend(&mut h, &v.to_le_bytes());
                }
                h
            };
            let pipe: Retained<Pso> = {
                let mut lock = cache.lock();
                if let Some(p) = lock.get(&cache_key) {
                    p.clone()
                } else {
                    let src = NSString::from_str(msl);
                    let lib = dev
                        .newLibraryWithSource_options_error(&src, None)
                        .map_err(|e| MetalTileError::Compilation(format!("{e:?}")))?;
                    let fn_name = NSString::from_str(&spec.kernel.name);
                    let fun = if spec.fn_consts.is_empty() {
                        lib.newFunctionWithName(&fn_name)
                            .ok_or_else(|| MetalTileError::Compilation(spec.kernel.name.clone()))?
                    } else {
                        use objc2_metal::{MTLDataType, MTLFunctionConstantValues};
                        let consts = MTLFunctionConstantValues::new();
                        for (n, v) in spec.fn_consts {
                            let name = NSString::from_str(n);
                            let val = v.to_le_bytes();
                            unsafe {
                                consts.setConstantValue_type_withName(
                                    NonNull::new(val.as_ptr() as *mut _)
                                        .expect("u32 bytes always non-null"),
                                    MTLDataType::UInt,
                                    &name,
                                );
                            }
                        }
                        lib.newFunctionWithName_constantValues_error(&fn_name, &consts)
                            .map_err(|e| MetalTileError::Compilation(format!("{e:?}")))?
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
            pipes.push(pipe);
        }

        let cb = queue.commandBuffer().ok_or(MetalTileError::NoDevice)?;

        // Per-spec buffer slots kept so true outputs (not consumed by later
        // specs) can be read back at the end.
        let mut per_spec_bufs: Vec<Vec<BufRc>> = Vec::with_capacity(specs.len());

        let push_strided = |bufs: &mut Vec<BufRc>,
                            param: &Param,
                            src: &BTreeMap<String, Vec<u8>>|
         -> Result<(), MetalTileError> {
            if param.kind == ParamKind::Strided {
                let (shape, strides) = resolve_strided_metadata(param, src)?;
                bufs.push(acquire_shared(Some(shape.as_ref()), shape.len())?);
                bufs.push(acquire_shared(Some(strides.as_ref()), strides.len())?);
            }
            Ok(())
        };

        for (i, spec) in specs.iter().enumerate() {
            let mut bufs: Vec<BufRc> = Vec::with_capacity(spec.kernel.params.len() * 2);

            for (param, plan) in spec.kernel.params.iter().zip(&binding_plans[i]) {
                // Inputs: resident-pre-uploaded > aliased from earlier spec.
                if !param.is_output {
                    let pre = spec
                        .resident
                        .get(&param.name)
                        .map(|r| r.inner.clone())
                        .or_else(|| alias_pool.get(&param.name).cloned());
                    if let Some(buf) = pre {
                        bufs.push(buf);
                        push_strided(&mut bufs, param, spec.buffers)?;
                        continue;
                    }
                }
                // Outputs aliased to later specs → private. Otherwise shared.
                let new_buf = if param.is_output && later_inputs[i].contains(param.name.as_str()) {
                    let b = acquire_private(plan.data_len)?;
                    alias_pool.insert(param.name.clone(), b.clone());
                    b
                } else {
                    let bytes = spec.buffers.get(&param.name).map(Vec::as_slice);
                    acquire_shared(bytes, plan.data_len)?
                };
                bufs.push(new_buf);
                push_strided(&mut bufs, param, spec.buffers)?;
            }
            // tensor_binding_count = next free slot after all tensor
            // bindings (data + strided shape/strides). Constexpr scalars
            // bind via setBytes starting here — no MTLBuffer alloc.
            let tensor_binding_count = bufs.len();

            let enc = (*cb).computeCommandEncoder().ok_or(MetalTileError::NoDevice)?;
            enc.setComputePipelineState(&pipes[i]);
            for (idx, buf) in bufs.iter().enumerate() {
                unsafe { enc.setBuffer_offset_atIndex(Some(buf.as_ref()), 0, idx) };
            }
            // Inline constexpr scalars via setBytes (Apple-recommended
            // path for <4KB constants — skips the allocator + binding
            // table indirection for each scalar). Same wire format the
            // codegen expects: each constexpr is a single u32/f32-sized
            // scalar at slot `tensor_binding_count + constexpr_index`.
            for (j, decl) in spec.kernel.constexprs.iter().enumerate() {
                let key = decl.name.name();
                let elem = decl.dtype.size_bytes().max(4);
                let bytes = spec.buffers.get(key).map(Vec::as_slice).unwrap_or(&[]);
                // Pad to elem if caller supplied fewer bytes (legacy
                // alloc_shared zeroed beyond elem too).
                let mut staged = [0u8; 16];
                let n = bytes.len().min(elem).min(staged.len());
                staged[..n].copy_from_slice(&bytes[..n]);
                unsafe {
                    enc.setBytes_length_atIndex(
                        NonNull::new(staged.as_ptr() as *mut _)
                            .ok_or_else(|| MetalTileError::Buffer("setBytes null".into()))?,
                        elem,
                        tensor_binding_count + j,
                    );
                }
            }
            let (g, t) = (spec.grid_groups, spec.threads_per_group);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: g[0], height: g[1], depth: g[2] },
                MTLSize { width: t[0], height: t[1], depth: t[2] },
            );
            // Barrier between consecutive passes — staging buffers go
            // private + untracked, so the driver doesn't auto-track
            // dependencies. We insert an explicit buffer-scope barrier.
            if i + 1 < specs.len() {
                enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
            }
            (*enc).endEncoding();
            per_spec_bufs.push(bufs);
        }

        (*cb).commit();
        (*cb).waitUntilCompleted();

        // Read back outputs from each spec, but only for params NOT
        // consumed as input by a later spec (the aliased ones live in
        // private memory and are intentionally not host-readable).
        let elapsed_us = ((*cb).GPUEndTime() - (*cb).GPUStartTime()) * 1_000_000.0;
        let mut results: Vec<DispatchResult> = Vec::with_capacity(specs.len());
        for (i, spec) in specs.iter().enumerate() {
            let mut outputs: BTreeMap<String, Vec<u8>> = BTreeMap::new();
            for (param, plan) in spec.kernel.params.iter().zip(&binding_plans[i]) {
                if !param.is_output {
                    continue;
                }
                if later_inputs[i].contains(param.name.as_str()) {
                    continue;
                }
                let Some(buf) = per_spec_bufs[i].get(plan.data_binding_index) else { continue };
                use objc2_metal::MTLBuffer as _;
                let ptr = buf.contents();
                let bytes =
                    unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const u8, plan.data_len) }
                        .to_vec();
                outputs.insert(param.name.clone(), bytes);
            }
            // Attribute the entire chain's GPU time to the first result;
            // chained passes share one cmd buffer and can't be split.
            let us = if i == 0 { elapsed_us } else { 0.0 };
            results.push(DispatchResult { elapsed_us: us, gflops: 0.0, outputs });
        }
        Ok(results)
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

    // Perf microbench — run with:
    //   cargo test --release -p metaltile-runtime perf_resolve_strided_metadata \
    //     -- --ignored --nocapture
    //
    // Times `resolve_strided_metadata` in a hot loop over a realistic param
    // name. Pre-change (`format!("{}_shape", …)` + `format!("{}_strides", …)`)
    // allocates two Strings per call; post-change reuses a single pre-sized
    // String via in-place suffix rewrite.
    #[test]
    #[ignore = "perf microbench"]
    fn perf_resolve_strided_metadata() {
        let param = tensor_param(
            "long_strided_kv_cache",
            DType::F32,
            &[8, 4096, 128],
            false,
            ParamKind::Strided,
        );
        let buffers = BTreeMap::new();
        const ITERS: usize = 5_000_000;
        for _ in 0..50_000 {
            std::hint::black_box(
                resolve_strided_metadata(&param, std::hint::black_box(&buffers)).unwrap(),
            );
        }
        let start = std::time::Instant::now();
        for _ in 0..ITERS {
            std::hint::black_box(
                resolve_strided_metadata(&param, std::hint::black_box(&buffers)).unwrap(),
            );
        }
        let elapsed = start.elapsed();
        let ns_per_call = elapsed.as_nanos() as f64 / ITERS as f64;
        println!("resolve_strided_metadata × {ITERS}: {elapsed:?} ({ns_per_call:.1} ns/call)");
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
