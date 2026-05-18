//! Programmatic Metal Frame Capture — wrap a span of dispatches in
//! `start_gpu_trace` / `stop_gpu_trace` to emit a `.gputrace` file
//! that Xcode opens with full DRAM/ALU/occupancy counters.
//!
//! ## Requirements
//!
//! Apple gates the `GPUTraceDocument` destination on EITHER:
//! 1. An Info.plist with `MetalCaptureEnabled=YES`, OR
//! 2. An attached debugger (Xcode "Capture GPU Frame") — uses
//!    the `DeveloperTools` destination instead.
//!
//! `cargo test` binaries have no Info.plist so case (1) doesn't apply
//! and `start_gpu_trace` returns an error. To use this API, package the
//! caller as a bundled app or attach Xcode.
//!
//! ## Alternative without code: `xctrace`
//!
//! For ad-hoc profiling of any binary (no Info.plist needed):
//!
//! ```text
//! xcrun xctrace record --template "Metal System Trace" \
//!     --output /tmp/sdpa.trace \
//!     --launch -- ./target/release/deps/<test-binary> \
//!         --ignored sdpa_decode_2pass_chained_perf_bench_f32 --nocapture
//! open /tmp/sdpa.trace
//! ```
//!
//! Opens in Instruments with the same counter set.
//!
//! ## Programmatic usage (when bundled)
//!
//! ```ignore
//! metaltile_runtime::start_gpu_trace("/tmp/sdpa.gputrace")?;
//! // ... dispatches ...
//! metaltile_runtime::stop_gpu_trace();
//! ```

use crate::MetalTileError;

#[cfg(target_os = "macos")]
pub fn start_gpu_trace(output_path: &str) -> Result<(), MetalTileError> {
    use objc2::{rc::Retained, runtime::ProtocolObject};
    use objc2_foundation::{NSString, NSURL};
    use objc2_metal::{
        MTLCaptureDescriptor,
        MTLCaptureDestination,
        MTLCaptureManager,
        MTLCreateSystemDefaultDevice,
        MTLDevice,
    };

    let manager = unsafe { MTLCaptureManager::sharedCaptureManager() };
    let dev: Retained<ProtocolObject<dyn MTLDevice>> =
        MTLCreateSystemDefaultDevice().ok_or(MetalTileError::NoDevice)?;
    if !manager.supportsDestination(MTLCaptureDestination::GPUTraceDocument) {
        return Err(MetalTileError::Compilation(
            "MTLCaptureManager rejects GPUTraceDocument — set MTL_CAPTURE_ENABLED=1 before \
             launching, or attach Xcode and use DeveloperTools destination instead"
                .into(),
        ));
    }
    let desc = MTLCaptureDescriptor::new();
    desc.set_capture_device(&dev);
    desc.setDestination(MTLCaptureDestination::GPUTraceDocument);
    let url_str = NSString::from_str(output_path);
    let url = NSURL::fileURLWithPath(&url_str);
    desc.setOutputURL(Some(&url));
    manager
        .startCaptureWithDescriptor_error(&desc)
        .map_err(|e| MetalTileError::Compilation(format!("startCapture: {e:?}")))
}

#[cfg(target_os = "macos")]
pub fn stop_gpu_trace() {
    use objc2_metal::MTLCaptureManager;
    unsafe { MTLCaptureManager::sharedCaptureManager().stopCapture() };
}

#[cfg(not(target_os = "macos"))]
pub fn start_gpu_trace(_output_path: &str) -> Result<(), MetalTileError> {
    Err(MetalTileError::NoDevice)
}

#[cfg(not(target_os = "macos"))]
pub fn stop_gpu_trace() {}
