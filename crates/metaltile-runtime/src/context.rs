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
    /// Apple GPU family level of the default device, when probeable
    /// (7 = M1, 8 = M2, 9 = M3/M4, 10 = M5). `None` off macOS or when
    /// no Metal device is available. Apple GPU families are cumulative,
    /// so this is the highest level that returned true.
    chip_family: Option<u32>,
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

#[cfg(any(target_os = "macos", test))]
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

#[cfg(any(target_os = "macos", test))]
fn fnv1a_extend(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h ^= b as u64;
        *h = h.wrapping_mul(0x0100_0000_01b3);
    }
}

/// PSO / MSL cache key for a dispatch_chain pass: kernel name +
/// first-param dtype size + sorted fn_consts. The MSL source is fully
/// determined by this tuple, so hashing the source string per pass is
/// pure waste (5–50 KB → ~10–30 µs vs ~16 ns for this key).
#[cfg(any(target_os = "macos", test))]
fn pso_cache_key(kernel: &Kernel, fn_consts: &BTreeMap<String, u32>) -> u64 {
    let mut h = FNV_OFFSET;
    fnv1a_extend(&mut h, kernel.name.as_bytes());
    fnv1a_extend(&mut h, b":");
    if let Some(p) = kernel.params.first() {
        fnv1a_extend(&mut h, &(p.dtype.size_bytes() as u64).to_le_bytes());
    }
    for (n, v) in fn_consts {
        fnv1a_extend(&mut h, n.as_bytes());
        fnv1a_extend(&mut h, &v.to_le_bytes());
    }
    h
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
        let chip_family = detect_apple_family();
        let tuner = Autotuner::new(Autotuner::default_cache_dir(), has_metal);
        Ok(Context { tuner, has_metal, chip_family })
    }

    pub fn has_gpu(&self) -> bool { self.has_metal }

    /// Apple GPU family level detected at construction time, or `None`
    /// off macOS. See the field doc for the level → chip mapping.
    pub fn chip_family(&self) -> Option<u32> { self.chip_family }

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
            let h = pso_cache_key(spec.kernel, spec.fn_consts);
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
            let cache_key = pso_cache_key(spec.kernel, spec.fn_consts);
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

/// Probe the default Metal device for the highest supported Apple GPU
/// family. Apple families are cumulative, so a chip that returns true
/// for `Apple10` also returns true for `Apple9`/`8`/`7` — we report
/// the newest. Returns `None` off macOS or when no Metal device is
/// available.
#[cfg(target_os = "macos")]
fn detect_apple_family() -> Option<u32> {
    use objc2_metal::{MTLDevice, MTLGPUFamily};
    let dev = objc2_metal::MTLCreateSystemDefaultDevice()?;
    for (family, level) in [
        (MTLGPUFamily::Apple10, 10u32),
        (MTLGPUFamily::Apple9, 9),
        (MTLGPUFamily::Apple8, 8),
        (MTLGPUFamily::Apple7, 7),
    ] {
        if dev.supportsFamily(family) {
            return Some(level);
        }
    }
    None
}

#[cfg(not(target_os = "macos"))]
fn detect_apple_family() -> Option<u32> { None }

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
    fn key_kernel(name: &str, dtype: DType) -> Kernel {
        let mut k = Kernel::new(name);
        k.params = vec![tensor_param("input", dtype, &[4], false, ParamKind::Tensor)];
        k
    }

    #[test]
    fn pso_cache_key_is_stable_and_discriminates_inputs() {
        let k = key_kernel("sdpa_decode", DType::F32);
        let mut consts = BTreeMap::new();
        consts.insert("gqa".to_string(), 4u32);
        consts.insert("head_dim".to_string(), 128u32);
        let baseline = pso_cache_key(&k, &consts);
        assert_eq!(baseline, pso_cache_key(&k, &consts), "key must be deterministic");

        // Kernel name change → different key.
        let k_other_name = key_kernel("sdpa_prefill", DType::F32);
        assert_ne!(baseline, pso_cache_key(&k_other_name, &consts));

        // First-param dtype change → different key (f32 vs f16 specializations).
        let k_other_dtype = key_kernel("sdpa_decode", DType::F16);
        assert_ne!(baseline, pso_cache_key(&k_other_dtype, &consts));

        // fn_const value change → different key.
        let mut consts_other_val = consts.clone();
        consts_other_val.insert("gqa".to_string(), 8u32);
        assert_ne!(baseline, pso_cache_key(&k, &consts_other_val));

        // fn_const name change → different key.
        let mut consts_other_name = BTreeMap::new();
        consts_other_name.insert("kv_heads".to_string(), 4u32);
        consts_other_name.insert("head_dim".to_string(), 128u32);
        assert_ne!(baseline, pso_cache_key(&k, &consts_other_name));

        // Empty fn_consts and empty params are both well-defined.
        let empty_consts = BTreeMap::new();
        assert_ne!(baseline, pso_cache_key(&k, &empty_consts));
        let mut k_empty = Kernel::new("noop");
        k_empty.params = vec![];
        let _ = pso_cache_key(&k_empty, &empty_consts);
    }

    #[test]
    fn fnv1a_extend_empty_input_is_identity() {
        // Hashing no bytes must not perturb the accumulator. Guarantees the
        // empty-fn_consts path in `pso_cache_key` doesn't depend on a separator.
        let mut h = FNV_OFFSET;
        fnv1a_extend(&mut h, &[]);
        assert_eq!(h, FNV_OFFSET);

        let mut h2 = 0xdead_beef_dead_beef_u64;
        fnv1a_extend(&mut h2, &[]);
        assert_eq!(h2, 0xdead_beef_dead_beef_u64);
    }

    #[test]
    fn fnv1a_extend_matches_canonical_fnv1a_64() {
        // Spot-check against canonical FNV-1a 64-bit vectors. "" → FNV_OFFSET,
        // "a" → 0xaf63dc4c8601ec8c, "foobar" → 0x85944171f73967e8 (well-known
        // FNV reference values). Pins the constants in case anyone "optimises"
        // the prime/offset.
        let cases: &[(&[u8], u64)] = &[
            (b"", 0xcbf2_9ce4_8422_2325),
            (b"a", 0xaf63_dc4c_8601_ec8c),
            (b"foobar", 0x8594_4171_f739_67e8),
        ];
        for (input, want) in cases {
            let mut h = FNV_OFFSET;
            fnv1a_extend(&mut h, input);
            assert_eq!(h, *want, "input={input:?}");
        }
    }

    #[test]
    fn fnv1a_extend_is_byte_by_byte_associative() {
        // Folding bytes in one shot must equal folding the same bytes split
        // across two calls. This is the property `pso_cache_key` relies on
        // when it threads the accumulator through name → ":" → dtype → consts.
        let bytes = b"sdpa_decode_2pass_pass1";
        let mut whole = FNV_OFFSET;
        fnv1a_extend(&mut whole, bytes);

        for split in 0..=bytes.len() {
            let (a, b) = bytes.split_at(split);
            let mut piecewise = FNV_OFFSET;
            fnv1a_extend(&mut piecewise, a);
            fnv1a_extend(&mut piecewise, b);
            assert_eq!(piecewise, whole, "split at {split}");
        }
    }

    #[test]
    fn fnv1a_extend_handles_large_input_without_overflow_panic() {
        // 10 KB blob matches the perf-microbench size and exercises the inner
        // wrapping_mul on a long stream. Result must be deterministic.
        let blob: Vec<u8> = (0..10_240).map(|i| (i as u8).wrapping_add(0x42)).collect();
        let mut a = FNV_OFFSET;
        let mut b = FNV_OFFSET;
        fnv1a_extend(&mut a, &blob);
        fnv1a_extend(&mut b, &blob);
        assert_eq!(a, b);
        assert_ne!(a, FNV_OFFSET);
    }

    #[test]
    fn pso_cache_key_reorder_does_not_change_key_because_btreemap_is_sorted() {
        // `pso_cache_key` iterates `fn_consts: &BTreeMap`, which yields keys
        // in sorted order regardless of insertion order. Two maps built in
        // different orders must hash identically — otherwise a caller that
        // inserted in a different order would miss the PSO cache.
        let k = key_kernel("sdpa_decode", DType::F32);
        let mut a = BTreeMap::new();
        a.insert("gqa".to_string(), 4u32);
        a.insert("head_dim".to_string(), 128u32);
        a.insert("n_kv_heads".to_string(), 8u32);

        let mut b = BTreeMap::new();
        b.insert("n_kv_heads".to_string(), 8u32);
        b.insert("head_dim".to_string(), 128u32);
        b.insert("gqa".to_string(), 4u32);

        assert_eq!(pso_cache_key(&k, &a), pso_cache_key(&k, &b));
    }

    #[test]
    fn pso_cache_key_dtype_size_collision_groups_match() {
        // Only the first param's dtype *size* is folded in (not the dtype
        // discriminant). f32 and i32 share size_bytes()==4, so kernels with
        // identical names + fn_consts that differ only between two
        // same-size dtypes are intentionally aliased — this documents the
        // behaviour and pins it for future refactors.
        let k_f32 = key_kernel("noop", DType::F32);
        let k_i32 = key_kernel("noop", DType::I32);
        let consts = BTreeMap::new();
        assert_eq!(pso_cache_key(&k_f32, &consts), pso_cache_key(&k_i32, &consts));

        // …but a different *size* (f16 = 2 bytes) must still discriminate.
        let k_f16 = key_kernel("noop", DType::F16);
        assert_ne!(pso_cache_key(&k_f32, &consts), pso_cache_key(&k_f16, &consts));
    }

    #[test]
    fn pso_cache_key_only_first_param_dtype_matters() {
        // `pso_cache_key` only looks at `params.first()`. A second param of
        // any dtype must not change the key. Documents the chosen
        // specialisation surface: a kernel is monomorphised by the dtype of
        // its first tensor operand; downstream dtypes are propagated.
        let mut k_one = Kernel::new("k");
        k_one.params = vec![tensor_param("a", DType::F32, &[4], false, ParamKind::Tensor)];
        let mut k_two = Kernel::new("k");
        k_two.params = vec![
            tensor_param("a", DType::F32, &[4], false, ParamKind::Tensor),
            tensor_param("b", DType::F16, &[4], true, ParamKind::Tensor),
        ];
        let consts = BTreeMap::new();
        assert_eq!(pso_cache_key(&k_one, &consts), pso_cache_key(&k_two, &consts));
    }

    #[test]
    fn pso_cache_key_no_params_still_well_defined() {
        // A kernel with zero params skips the dtype-size fold but must still
        // produce a stable, non-trivial key (the kernel name + ":" + sorted
        // fn_consts get folded in). Pins that the `if let Some(...)` branch
        // is the *only* effect of param absence — separator and consts still
        // fold.
        let mut k = Kernel::new("zero_param_kernel");
        k.params = vec![];
        let mut consts = BTreeMap::new();
        consts.insert("foo".to_string(), 1u32);
        let key = pso_cache_key(&k, &consts);
        assert_ne!(key, FNV_OFFSET);
        assert_eq!(key, pso_cache_key(&k, &consts));

        // A different kernel name still discriminates with no params present.
        let mut k_other = Kernel::new("other_zero_param_kernel");
        k_other.params = vec![];
        assert_ne!(key, pso_cache_key(&k_other, &consts));
    }

    /// Representative MSL source size for a real metaltile kernel:
    /// sdpa_decode_2pass_pass1 generates ~12 KB; sdpa_vector ~6 KB. A
    /// mid-size 10 KB blob is the pre-fix per-pass cost we're comparing
    /// against. Shared by the always-on coverage test and the perf bench.
    fn perf_bench_msl_blob() -> Vec<u8> {
        (0..10_240).map(|i| (i as u8).wrapping_add(0x42)).collect()
    }

    fn perf_bench_kernel() -> Kernel { key_kernel("sdpa_decode_2pass_pass1", DType::F32) }

    fn perf_bench_consts() -> BTreeMap<String, u32> {
        let mut consts = BTreeMap::new();
        consts.insert("gqa".to_string(), 4u32);
        consts
    }

    /// Pre-fix per-pass cost: FNV-1a over the full MSL source.
    fn pre_fix_pass_key(msl_bytes: &[u8]) -> u64 {
        let mut h = FNV_OFFSET;
        fnv1a_extend(&mut h, std::hint::black_box(msl_bytes));
        std::hint::black_box(h)
    }

    /// Post-fix per-pass cost: structured `pso_cache_key` over the small
    /// (name, first-param dtype, sorted fn_consts) tuple.
    fn post_fix_pass_key(kernel: &Kernel, consts: &BTreeMap<String, u32>) -> u64 {
        std::hint::black_box(pso_cache_key(
            std::hint::black_box(kernel),
            std::hint::black_box(consts),
        ))
    }

    #[test]
    fn pso_cache_key_separator_prevents_name_dtype_smudge() {
        // `pso_cache_key` folds `kernel.name` then `b":"` then the dtype-size
        // bytes. Without the separator, a kernel named `"foo"` with dtype
        // size N could collide with one named `"foo<sep_bytes>"` with no
        // params. The literal `":"` byte is what stops that.
        let mut k_a = Kernel::new("foo");
        k_a.params = vec![tensor_param("x", DType::F32, &[1], false, ParamKind::Tensor)];

        // A kernel named "foo:" with no params: if the separator weren't
        // there, the post-name accumulator state would equal `"foo" + ":"`
        // from `k_a`, and a no-params kernel skips the dtype fold — so
        // without the separator these *could* collide. With the separator,
        // they must differ because `k_a` folds in 8 dtype bytes after ":"
        // and `k_b` folds in nothing.
        let mut k_b = Kernel::new("foo:");
        k_b.params = vec![];
        let consts = BTreeMap::new();
        assert_ne!(pso_cache_key(&k_a, &consts), pso_cache_key(&k_b, &consts));
    }

    #[test]
    fn pso_cache_key_const_value_endianness_pinned() {
        // fn_const values are folded as little-endian u32 bytes. Pin that
        // by checking that two values which are byte-swapped versions of
        // each other produce different keys (catches an accidental switch
        // to `to_be_bytes`).
        let k = key_kernel("k", DType::F32);
        let mut a = BTreeMap::new();
        a.insert("c".to_string(), 0x0000_0001_u32);
        let mut b = BTreeMap::new();
        b.insert("c".to_string(), 0x0100_0000_u32);
        assert_ne!(pso_cache_key(&k, &a), pso_cache_key(&k, &b));
    }

    #[test]
    fn pso_cache_key_distinct_for_realistic_kernel_matrix() {
        // Sanity-sweep across a realistic matrix of (kernel, dtype, gqa,
        // head_dim) tuples and assert all keys are pairwise distinct. Any
        // accidental aliasing (e.g. dropping the const name in favour of
        // value-only hashing) would surface here as a duplicate.
        let kernels = ["sdpa_decode", "sdpa_prefill", "rmsnorm", "matmul_4bit"];
        // Use dtypes with distinct size_bytes() — F16/BF16 both fold to 2,
        // so the test would self-collide. The dtype-size alias is
        // intentional (and pinned by `pso_cache_key_dtype_size_collision_groups_match`).
        let dtypes = [DType::F32, DType::F16];
        let gqas = [1u32, 4, 8];
        let head_dims = [64u32, 128];

        let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut total = 0usize;
        for kname in kernels {
            for dt in dtypes {
                for gqa in gqas {
                    for hd in head_dims {
                        let k = key_kernel(kname, dt);
                        let mut consts = BTreeMap::new();
                        consts.insert("gqa".to_string(), gqa);
                        consts.insert("head_dim".to_string(), hd);
                        let key = pso_cache_key(&k, &consts);
                        assert!(
                            seen.insert(key),
                            "duplicate key for ({kname}, {dt:?}, gqa={gqa}, hd={hd})"
                        );
                        total += 1;
                    }
                }
            }
        }
        assert_eq!(total, seen.len());
        assert_eq!(total, kernels.len() * dtypes.len() * gqas.len() * head_dims.len());
    }

    #[test]
    fn perf_pass_keys_run_and_discriminate() {
        // Always-on coverage for the per-pass key helpers used by the
        // `#[ignore]`'d perf bench below. Both paths must produce stable,
        // non-trivial keys, and the two paths must NOT collide — the whole
        // point of the PR is that they're different discriminators for the
        // same kernel.
        let msl_bytes = perf_bench_msl_blob();
        let kernel = perf_bench_kernel();
        let consts = perf_bench_consts();

        let pre = pre_fix_pass_key(&msl_bytes);
        assert_eq!(pre, pre_fix_pass_key(&msl_bytes), "pre-fix key is deterministic");
        assert_ne!(pre, 0);

        let post = post_fix_pass_key(&kernel, &consts);
        assert_eq!(post, post_fix_pass_key(&kernel, &consts), "post-fix key is deterministic");
        assert_ne!(post, 0);

        assert_ne!(pre, post, "pre/post must hash distinct inputs");
    }

    // Perf microbench — run with:
    //   cargo test --release -p metaltile-runtime perf_dispatch_chain_pso_key \
    //     -- --ignored --nocapture
    //
    // Times the per-pass PSO cache key computation in `dispatch_chain_metal`.
    // Pre-fix: FNV-1a over the full MSL source string (5–50 KB per pass).
    // Post-fix: `pso_cache_key` over the kernel name + first-param dtype size
    // + sorted fn_consts (~30–60 bytes). Demonstrates the savings without a
    // GPU; the fix itself is the helper call inside `dispatch_chain_metal`.
    #[test]
    #[ignore = "perf microbench"]
    fn perf_dispatch_chain_pso_key() {
        let msl_bytes = perf_bench_msl_blob();
        let kernel = perf_bench_kernel();
        let consts = perf_bench_consts();

        const ITERS: usize = 1_000_000;
        // Warmup.
        let mut warm: u64 = 0;
        for _ in 0..10_000 {
            warm ^= pre_fix_pass_key(&msl_bytes);
        }
        std::hint::black_box(warm);

        let start = std::time::Instant::now();
        let mut pre_acc: u64 = 0;
        for _ in 0..ITERS {
            pre_acc ^= pre_fix_pass_key(&msl_bytes);
        }
        let pre = start.elapsed();
        std::hint::black_box(pre_acc);

        let start = std::time::Instant::now();
        let mut post_acc: u64 = 0;
        for _ in 0..ITERS {
            post_acc ^= post_fix_pass_key(&kernel, &consts);
        }
        let post = start.elapsed();
        std::hint::black_box(post_acc);

        let pre_ns = pre.as_nanos() as f64 / ITERS as f64;
        let post_ns = post.as_nanos() as f64 / ITERS as f64;
        let saved_ns = pre_ns - post_ns;
        println!(
            "pre-fix  (FNV over 10 KB MSL):  {pre:?}  ({pre_ns:.0} ns/call)\n\
             post-fix (pso_cache_key call):  {post:?}  ({post_ns:.0} ns/call)\n\
             saved per dispatch_chain pass:  {saved_ns:.0} ns",
        );
    }

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

        let mut ctx = Context {
            tuner: Autotuner::new(bad_cache_root.clone(), false),
            has_metal: false,
            chip_family: None,
        };
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

    #[test]
    #[cfg(target_os = "macos")]
    fn context_detects_apple_gpu_family() {
        // Every Mac that compiles + runs this test is at least Apple7
        // (M1). The exact level depends on which Mac runs CI, so just
        // assert the lower bound + the cumulative ordering.
        //
        // GitHub Actions' hosted macOS runners report no Apple GPU family
        // (virtualized / non-Apple GPU), so `chip_family()` is `None` there.
        // Treat that as a pass — real hardware still gets the strict check.
        let ctx = Context::new().expect("Context::new should succeed on macOS");
        let Some(level) = ctx.chip_family() else {
            assert!(
                std::env::var_os("CI").is_some(),
                "macOS Context must report a family on real hardware",
            );
            return;
        };
        assert!(level >= 7, "Apple GPU family level should be >=7 on M-series, got {level}");
        assert!(level <= 20, "level looks unreasonably high; bound is sanity-only ({level})");
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn context_chip_family_is_none_off_macos() {
        let ctx = Context::new().expect("Context::new should succeed off macOS too");
        assert!(ctx.chip_family().is_none());
    }

    #[test]
    fn detect_apple_family_helper_runs() {
        // Sanity: the helper compiles + runs on every target. On macOS
        // it returns Some(>=7); elsewhere it returns None. The specific
        // arm is asserted by the cfg-gated tests above.
        //
        // GitHub Actions' hosted macOS runners have no Apple GPU, so the
        // helper returns `None` there — accept that when `CI` is set.
        let level = detect_apple_family();
        if cfg!(target_os = "macos") {
            if level.is_none() && std::env::var_os("CI").is_some() {
                return;
            }
            assert!(matches!(level, Some(l) if (7..=20).contains(&l)));
        } else {
            assert!(level.is_none());
        }
    }
}
