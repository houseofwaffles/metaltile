//! Pre-recorded MLX performance baselines, loaded from fixtures/baselines.json.
//! Used to compare MetalTile against MLX on the same hardware without running MLX.

use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct BaselineEntry {
    chip_pattern: String,
    kernel: String,
    shape: String,
    gflops: Option<f64>,
    gbps: Option<f64>,
}

static FIXTURE_JSON: &str = include_str!("../fixtures/baselines.json");

fn entries() -> &'static [BaselineEntry] {
    use std::sync::OnceLock;
    static CACHE: OnceLock<Vec<BaselineEntry>> = OnceLock::new();
    CACHE.get_or_init(|| serde_json::from_str(FIXTURE_JSON).expect("invalid baselines.json"))
}

fn best_match<'a>(
    device: &str,
    entries: &'a [BaselineEntry],
) -> impl Iterator<Item = &'a BaselineEntry> {
    // Find the longest chip_pattern that appears in the device name.
    let best_len = entries
        .iter()
        .filter(|e| device.contains(&e.chip_pattern))
        .map(|e| e.chip_pattern.len())
        .max()
        .unwrap_or(0);
    entries
        .iter()
        .filter(move |e| e.chip_pattern.len() == best_len && device.contains(&e.chip_pattern))
}

fn lookup<'a>(device: &str, kernel: &str, shape: &str) -> Option<&'static BaselineEntry> {
    best_match(device, entries()).find(|e| e.kernel == kernel && e.shape == shape)
}

/// Look up a pre-recorded GFLOPS baseline.
pub fn gflops(device: &str, kernel: &str, shape: &str) -> Option<f64> {
    lookup(device, kernel, shape)?.gflops
}

/// Look up a pre-recorded GB/s baseline.
pub fn gbps(device: &str, kernel: &str, shape: &str) -> Option<f64> {
    lookup(device, kernel, shape)?.gbps
}
