//! GPU runner: compile Metal source, allocate buffers, dispatch kernels, measure GPU time.
//!
//! All Metal-specific code is gated with `#[cfg(target_os = "macos")]`.
//! On other platforms every method returns `Err` or a zero-filled stub.

use crate::stats::BenchStats;

/// Convert IEEE 754 half-float bits to f32.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) >> 15) << 31;
    let exp5 = ((bits as u32) >> 10) & 0x1f;
    let mantissa = (bits as u32) & 0x3ff;
    if exp5 == 0 {
        return f32::from_bits(sign); // denormal → zero (flush)
    }
    if exp5 == 31 {
        return f32::from_bits(sign | 0x7f80_0000 | (mantissa << 13)); // inf/nan
    }
    let exp8 = (exp5 as i32 - 15 + 127) as u32;
    f32::from_bits(sign | (exp8 << 23) | (mantissa << 13))
}

// ── Public types ─────────────────────────────────────────────────────────────

pub struct GpuRunner {
    pub device_name: String,
    #[cfg(target_os = "macos")]
    inner: MacosRunner,
    /// Pre-compiled kernel and scratch buffer for SLC cache-flush.
    /// Writing 128 MB (> M4 Max's 64 MB SLC) evicts any cached benchmark data.
    #[cfg(target_os = "macos")]
    slc_kernel: CompiledKernel,
    #[cfg(target_os = "macos")]
    slc_buf: GpuBuffer,
}

#[allow(clippy::manual_non_exhaustive)]
pub struct CompiledKernel {
    #[cfg(target_os = "macos")]
    inner: MacosPipeline,
    #[cfg(not(target_os = "macos"))]
    _priv: (),
}

#[allow(clippy::manual_non_exhaustive)]
pub struct GpuBuffer {
    pub size_bytes: usize,
    #[cfg(target_os = "macos")]
    inner: MacosBuffer,
    #[cfg(not(target_os = "macos"))]
    _priv: (),
}

// ── macOS implementation ──────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
mod metal_impl {
    use objc2::{rc::Retained, runtime::ProtocolObject};
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLBuffer,
        MTLCommandBuffer,
        MTLCommandEncoder,
        MTLCommandQueue,
        MTLComputeCommandEncoder,
        MTLComputePipelineDescriptor,
        MTLComputePipelineState,
        MTLDataType,
        MTLDevice,
        MTLFunctionConstantValues,
        MTLLibrary,
        MTLResourceOptions,
    };

    pub struct MacosRunner {
        pub device: Retained<ProtocolObject<dyn MTLDevice>>,
        pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    }

    pub struct MacosPipeline {
        pub pso: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    }

    pub struct MacosBuffer {
        pub buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    }

    impl MacosRunner {
        pub fn new() -> Result<(String, Self), String> {
            let device = objc2_metal::MTLCreateSystemDefaultDevice().ok_or("no Metal device")?;
            let name = device.name().to_string();
            let queue = device.newCommandQueue().ok_or("newCommandQueue failed")?;
            Ok((name, MacosRunner { device, queue }))
        }

        pub fn compile(&self, source: &str, fn_name: &str) -> Result<MacosPipeline, String> {
            let opts = objc2_metal::MTLCompileOptions::new();
            let src = NSString::from_str(source);
            let lib: Retained<ProtocolObject<dyn MTLLibrary>> = self
                .device
                .newLibraryWithSource_options_error(&src, Some(&opts))
                .map_err(|e| format!("compile '{fn_name}': {e}"))?;
            let fname = NSString::from_str(fn_name);
            let func = lib
                .newFunctionWithName(&fname)
                .ok_or_else(|| format!("no function '{fn_name}'"))?;
            let desc = MTLComputePipelineDescriptor::new();
            desc.setComputeFunction(Some(&func));
            let pso = self
                .device
                .newComputePipelineStateWithDescriptor_options_reflection_error(
                    &desc,
                    objc2_metal::MTLPipelineOption::empty(),
                    None,
                )
                .map_err(|e| format!("pipeline '{fn_name}': {e}"))?;
            Ok(MacosPipeline { pso })
        }

        /// Compile a kernel with boolean function constants (index → value).
        pub fn compile_with_bool_constants(
            &self,
            source: &str,
            fn_name: &str,
            bool_constants: &[(usize, bool)],
        ) -> Result<MacosPipeline, String> {
            let opts = objc2_metal::MTLCompileOptions::new();
            let src = NSString::from_str(source);
            let lib: Retained<ProtocolObject<dyn MTLLibrary>> = self
                .device
                .newLibraryWithSource_options_error(&src, Some(&opts))
                .map_err(|e| format!("compile '{fn_name}': {e}"))?;
            let cv = MTLFunctionConstantValues::new();
            for &(idx, val) in bool_constants {
                let val_ptr =
                    std::ptr::NonNull::new(&val as *const bool as *mut std::ffi::c_void).unwrap();
                unsafe {
                    cv.setConstantValue_type_atIndex(val_ptr, MTLDataType::Bool, idx);
                }
            }
            let fname = NSString::from_str(fn_name);
            let func = lib
                .newFunctionWithName_constantValues_error(&fname, &cv)
                .map_err(|e| format!("specialize '{fn_name}': {e}"))?;
            let desc = MTLComputePipelineDescriptor::new();
            desc.setComputeFunction(Some(&func));
            let pso = self
                .device
                .newComputePipelineStateWithDescriptor_options_reflection_error(
                    &desc,
                    objc2_metal::MTLPipelineOption::empty(),
                    None,
                )
                .map_err(|e| format!("pipeline '{fn_name}': {e}"))?;
            Ok(MacosPipeline { pso })
        }

        pub fn alloc_bytes(&self, data: &[u8]) -> MacosBuffer {
            use std::ptr::NonNull;
            let len = data.len().max(4);
            let buf = unsafe {
                self.device
                    .newBufferWithBytes_length_options(
                        NonNull::new(data.as_ptr() as *mut _).unwrap(),
                        len,
                        MTLResourceOptions::StorageModeShared,
                    )
                    .expect("newBufferWithBytes failed")
            };
            MacosBuffer { buf }
        }

        pub fn alloc_zeros(&self, n_bytes: usize) -> MacosBuffer {
            let len = n_bytes.max(4);
            let buf = self
                .device
                .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
                .expect("newBufferWithLength failed");
            MacosBuffer { buf }
        }

        pub fn read_bytes(buf: &MacosBuffer, n_bytes: usize) -> Vec<u8> {
            use objc2_metal::MTLBuffer;
            let ptr = buf.buf.contents();
            unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const u8, n_bytes) }.to_vec()
        }

        pub fn measure(
            &self,
            pso: &MacosPipeline,
            buffers: &[&MacosBuffer],
            tgs: [usize; 3],
            tpg: [usize; 3],
            warmup: usize,
            iters: usize,
        ) -> Vec<f64> {
            use objc2_metal::MTLSize;
            let mut results = Vec::with_capacity(iters);
            for pass in 0..(warmup + iters) {
                unsafe {
                    let cb = self.queue.commandBuffer().expect("commandBuffer");
                    let enc = cb.computeCommandEncoder().expect("computeCommandEncoder");
                    enc.setComputePipelineState(&pso.pso);
                    for (i, b) in buffers.iter().enumerate() {
                        enc.setBuffer_offset_atIndex(Some(&b.buf), 0, i);
                    }
                    enc.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize { width: tgs[0], height: tgs[1], depth: tgs[2] },
                        MTLSize { width: tpg[0], height: tpg[1], depth: tpg[2] },
                    );
                    enc.endEncoding();
                    cb.commit();
                    cb.waitUntilCompleted();
                    if pass >= warmup {
                        let gpu_us = ((*cb).GPUEndTime() - (*cb).GPUStartTime()) * 1_000_000.0;
                        results.push(gpu_us);
                    }
                }
            }
            results
        }
    }
}

#[cfg(target_os = "macos")]
use metal_impl::{MacosBuffer, MacosPipeline, MacosRunner};

// ── GpuRunner ────────────────────────────────────────────────────────────────

impl GpuRunner {
    pub fn new() -> Result<Self, String> {
        #[cfg(target_os = "macos")]
        {
            const SLC_FLUSH_MSL: &str = concat!(
                "#include <metal_stdlib>\nusing namespace metal;\n",
                "kernel void _mt_slc_flush(",
                "device uint* buf [[buffer(0)]],",
                "uint gid [[thread_position_in_grid]]",
                ") { buf[gid] = gid; }"
            );
            const SLC_BYTES: usize = 128 * 1024 * 1024; // 128 MB > M4 Max 64 MB SLC

            let (name, inner) = MacosRunner::new()?;
            let slc_pso = inner
                .compile(SLC_FLUSH_MSL, "_mt_slc_flush")
                .map_err(|e| format!("SLC flush compile: {e}"))?;
            let slc_kernel = CompiledKernel { inner: slc_pso };
            let slc_buf = GpuBuffer { size_bytes: SLC_BYTES, inner: inner.alloc_zeros(SLC_BYTES) };
            Ok(GpuRunner { device_name: name, inner, slc_kernel, slc_buf })
        }
        #[cfg(not(target_os = "macos"))]
        Err("Metal not available on this platform".into())
    }

    #[allow(unused_variables)]
    pub fn compile(&self, source: &str, fn_name: &str) -> Result<CompiledKernel, String> {
        #[cfg(target_os = "macos")]
        {
            Ok(CompiledKernel { inner: self.inner.compile(source, fn_name)? })
        }
        #[cfg(not(target_os = "macos"))]
        Err("not macOS".into())
    }

    /// Compile a kernel with boolean function constants. `bool_constants` is a list of
    /// (function_constant_index, value) pairs.
    #[allow(unused_variables)]
    pub fn compile_with_bool_constants(
        &self,
        source: &str,
        fn_name: &str,
        bool_constants: &[(usize, bool)],
    ) -> Result<CompiledKernel, String> {
        #[cfg(target_os = "macos")]
        {
            Ok(CompiledKernel {
                inner: self.inner.compile_with_bool_constants(source, fn_name, bool_constants)?,
            })
        }
        #[cfg(not(target_os = "macos"))]
        Err("not macOS".into())
    }

    // ── Buffer constructors ──────────────────────────────────────────────────

    pub fn buffer_bytes(&self, data: &[u8]) -> GpuBuffer {
        #[cfg(target_os = "macos")]
        return GpuBuffer { size_bytes: data.len(), inner: self.inner.alloc_bytes(data) };
        #[cfg(not(target_os = "macos"))]
        GpuBuffer { size_bytes: data.len(), _priv: () }
    }

    pub fn buffer_zeros(&self, n_bytes: usize) -> GpuBuffer {
        #[cfg(target_os = "macos")]
        return GpuBuffer { size_bytes: n_bytes, inner: self.inner.alloc_zeros(n_bytes) };
        #[cfg(not(target_os = "macos"))]
        GpuBuffer { size_bytes: n_bytes, _priv: () }
    }

    pub fn buffer_f32(&self, data: &[f32]) -> GpuBuffer {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.buffer_bytes(&bytes)
    }

    /// `data` is raw fp16 bits (e.g. `0x3C00` = 1.0).
    pub fn buffer_f16(&self, data: &[u16]) -> GpuBuffer {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.buffer_bytes(&bytes)
    }

    pub fn buffer_u32(&self, v: u32) -> GpuBuffer { self.buffer_bytes(&v.to_le_bytes()) }
    pub fn buffer_i32(&self, v: i32) -> GpuBuffer { self.buffer_bytes(&v.to_le_bytes()) }
    pub fn buffer_u64(&self, v: u64) -> GpuBuffer { self.buffer_bytes(&v.to_le_bytes()) }
    pub fn buffer_i64(&self, v: i64) -> GpuBuffer { self.buffer_bytes(&v.to_le_bytes()) }
    pub fn buffer_f32_scalar(&self, v: f32) -> GpuBuffer { self.buffer_bytes(&v.to_le_bytes()) }

    // ── Readback ─────────────────────────────────────────────────────────────

    /// Read `n_bytes` raw bytes back from a GPU buffer.
    #[allow(unused_variables)]
    pub fn read_bytes(&self, buf: &GpuBuffer, n_bytes: usize) -> Vec<u8> {
        #[cfg(target_os = "macos")]
        {
            MacosRunner::read_bytes(&buf.inner, n_bytes)
        }
        #[cfg(not(target_os = "macos"))]
        vec![0u8; n_bytes]
    }

    /// Read `n` f32 values back from a GPU buffer allocated with buffer_zeros / buffer_f32.
    /// The buffer must use StorageModeShared (all buffers created by GpuRunner do).
    #[allow(unused_variables)]
    pub fn read_f32_slice(&self, buf: &GpuBuffer, n: usize) -> Vec<f32> {
        #[cfg(target_os = "macos")]
        {
            let bytes = MacosRunner::read_bytes(&buf.inner, n * 4);
            bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
        }
        #[cfg(not(target_os = "macos"))]
        vec![0.0f32; n]
    }

    /// Read `n` bfloat16 values back from a GPU buffer, returned as f32.
    /// BF16 is just the top 16 bits of a float32 representation.
    #[allow(unused_variables)]
    pub fn read_bf16_slice(&self, buf: &GpuBuffer, n: usize) -> Vec<f32> {
        #[cfg(target_os = "macos")]
        {
            let bytes = MacosRunner::read_bytes(&buf.inner, n * 2);
            bytes
                .chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect()
        }
        #[cfg(not(target_os = "macos"))]
        vec![0.0f32; n]
    }

    /// Read `n` f16 values back from a GPU buffer, returned as f32.
    #[allow(unused_variables)]
    pub fn read_f16_slice(&self, buf: &GpuBuffer, n: usize) -> Vec<f32> {
        #[cfg(target_os = "macos")]
        {
            let bytes = MacosRunner::read_bytes(&buf.inner, n * 2);
            bytes
                .chunks_exact(2)
                .map(|b| f16_bits_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect()
        }
        #[cfg(not(target_os = "macos"))]
        vec![0.0f32; n]
    }

    // ── Dispatch ─────────────────────────────────────────────────────────────

    #[allow(unused_variables)]
    pub fn measure(
        &self,
        kernel: &CompiledKernel,
        buffers: &[&GpuBuffer],
        tgs: [usize; 3],
        tpg: [usize; 3],
        warmup: usize,
        iters: usize,
    ) -> Vec<f64> {
        #[cfg(target_os = "macos")]
        {
            let raw: Vec<&MacosBuffer> = buffers.iter().map(|b| &b.inner).collect();
            self.inner.measure(&kernel.inner, &raw, tgs, tpg, warmup, iters)
        }
        #[cfg(not(target_os = "macos"))]
        vec![0.0; iters]
    }

    #[allow(unused_variables)]
    pub fn bench(
        &self,
        kernel: &CompiledKernel,
        buffers: &[&GpuBuffer],
        tgs: [usize; 3],
        tpg: [usize; 3],
        warmup: usize,
        iters: usize,
    ) -> BenchStats {
        BenchStats::from_samples(self.measure(kernel, buffers, tgs, tpg, warmup, iters))
    }

    /// Write 128 MB to a scratch buffer to evict the System Level Cache (SLC).
    ///
    /// Call this before each timed benchmark run so that both the reference and
    /// MetalTile kernels start from the same cold-cache state, eliminating the
    /// measurement noise that arises when the working set fits inside the SLC.
    pub fn flush_slc(&self) {
        #[cfg(target_os = "macos")]
        {
            const N_ELEM: usize = 128 * 1024 * 1024 / 4; // 32 M uint32 elements
            const TPG: usize = 256;
            self.inner.measure(
                &self.slc_kernel.inner,
                &[&self.slc_buf.inner],
                [N_ELEM / TPG, 1, 1],
                [TPG, 1, 1],
                0,
                1,
            );
        }
    }

    /// Returns true if the device supports simdgroup matrix operations (M1+ / Apple GPU family 7+).
    pub fn supports_simd_matrix(&self) -> bool {
        #[cfg(target_os = "macos")]
        {
            use objc2_metal::{MTLDevice, MTLGPUFamily};
            let dev = &self.inner.device;
            // Apple GPU families are cumulative — Apple10 (M5) implies Apple9/8/7 —
            // so any of these returning true is sufficient. Listed newest-first to
            // short-circuit on modern hardware.
            dev.supportsFamily(MTLGPUFamily::Apple10)
                || dev.supportsFamily(MTLGPUFamily::Apple9)
                || dev.supportsFamily(MTLGPUFamily::Apple8)
                || dev.supportsFamily(MTLGPUFamily::Apple7)
        }
        #[cfg(not(target_os = "macos"))]
        false
    }
}
