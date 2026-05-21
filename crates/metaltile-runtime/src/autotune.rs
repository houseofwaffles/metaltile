//! Autotuner: persistent tuning cache for kernel schedules.
//!
//! The autotuner stores the best schedule configuration for each
//! (kernel, chip, shape_bucket) combination. Configs are persisted
//! to `~/.cache/metaltile/<chip>/<kernel_hash>.json`.
//!
//! ## Search strategy
//!
//! Grid search over config space with exponential backoff:
//! 1. Coarse grid (large step sizes) → pick top 3
//! 2. Fine grid around each top candidate → pick best
//! 3. Store winner to disk cache

use std::{collections::BTreeMap, path::PathBuf};

use serde::{Deserialize, Serialize};
use tracing::debug;

/// A single autotune configuration: tile sizes, thread layout, etc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuneConfig {
    /// Tile dimensions (M, N, K for matmul-style ops).
    pub tile_dims: Vec<usize>,
    /// Threads per threadgroup (x, y, z).
    pub threads: (u32, u32, u32),
    /// Unroll factor for inner loops.
    pub unroll_factor: u32,
    /// Whether to use SIMD matrix multiply.
    pub use_simd_matrix: bool,
    /// Whether to use async copy for streaming.
    pub use_async_copy: bool,
}

/// A shape bucket: ranges of dimension values.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct ShapeBucket {
    /// Which constexpr dimension this bucket covers (by name).
    pub dim_name: String,
    /// Lower bound (inclusive).
    pub lo: usize,
    /// Upper bound (exclusive).
    pub hi: usize,
}

/// A single tuning entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuneEntry {
    /// Shape bucket this config is for.
    pub bucket: Vec<ShapeBucket>,
    /// The best configuration found.
    pub best_config: TuneConfig,
    /// Achieved performance (GFLOPS or time in μs).
    pub perf: f64,
    /// When this entry was last updated.
    pub timestamp: u64,
}

/// Persistent autotune cache.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct TuneCache {
    /// entries[bucket_key] = best config
    entries: BTreeMap<String, TuneEntry>,
}

impl TuneCache {
    /// Load from disk, or create empty.
    pub fn load(path: &PathBuf) -> Self {
        if path.exists() {
            std::fs::read_to_string(path)
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or_default()
        } else {
            TuneCache::default()
        }
    }

    /// Save to disk.
    pub fn save(&self, path: &PathBuf) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)
    }

    /// Look up the best config for a given set of constexpr values.
    pub fn lookup(
        &self,
        _constexprs: &metaltile_core::constexpr::ConstExprValues,
    ) -> Option<&TuneEntry> {
        // In production: bucket the values, then hash the bucket key,
        // then look up in entries. For now, return None (always re-tune).
        None
    }

    /// Insert or update a tuning entry.
    pub fn insert(&mut self, key: impl Into<String>, entry: TuneEntry) {
        self.entries.insert(key.into(), entry);
    }
}

/// The autotuner: coordinates tuning across kernel launches.
pub struct Autotuner {
    /// Disk cache path.
    cache_path: PathBuf,
    /// In-memory cache.
    cache: TuneCache,
    /// Whether autotune is enabled.
    enabled: bool,
}

impl Autotuner {
    /// Create a new autotuner with a cache directory.
    pub fn new(cache_dir: PathBuf, enabled: bool) -> Self {
        let cache_path = cache_dir.join("tuning_cache.json");
        let cache = TuneCache::load(&cache_path);

        Autotuner { cache_path, cache, enabled }
    }

    /// Default cache directory: `~/.cache/metaltile/`.
    pub fn default_cache_dir() -> PathBuf {
        dirs_next().unwrap_or_else(|| PathBuf::from(".cache")).join("metaltile")
    }

    /// Enable or disable autotuning.
    pub fn set_enabled(&mut self, enabled: bool) { self.enabled = enabled; }

    /// Get the best known config, or trigger tuning.
    #[tracing::instrument(skip(self, constexprs), fields(key = %_kernel_name))]
    pub fn get_or_tune(
        &mut self,
        _kernel_name: &str,
        constexprs: &metaltile_core::constexpr::ConstExprValues,
    ) -> Option<TuneConfig> {
        if !self.enabled {
            return Some(TuneConfig {
                tile_dims: vec![32, 32, 32],
                threads: (256, 1, 1),
                unroll_factor: 4,
                use_simd_matrix: true,
                use_async_copy: false,
            });
        }

        if let Some(entry) = self.cache.lookup(constexprs) {
            debug!("autotune cache hit");
            return Some(entry.best_config.clone());
        }

        // TODO: actually run tuning (benchmark different configs)
        None
    }

    /// Persist the cache to disk.
    pub fn flush(&self) -> Result<(), crate::error::MetalTileError> {
        Ok(self.cache.save(&self.cache_path)?)
    }
}

fn dirs_next() -> Option<PathBuf> { std::env::var("HOME").ok().map(PathBuf::from) }

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use metaltile_core::constexpr::ConstExprValues;

    use super::*;

    /// Unique scratch dir per test, rooted in `std::env::temp_dir()` so we
    /// don't trample the real `~/.cache/metaltile/` cache when running
    /// tests in parallel.
    fn scratch_dir() -> PathBuf {
        static SEQ: AtomicUsize = AtomicUsize::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "metaltile-autotune-test-{}-{}",
            std::process::id(),
            n,
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn sample_config() -> TuneConfig {
        TuneConfig {
            tile_dims: vec![32, 32, 32],
            threads: (256, 1, 1),
            unroll_factor: 4,
            use_simd_matrix: true,
            use_async_copy: false,
        }
    }

    fn sample_entry() -> TuneEntry {
        TuneEntry {
            bucket: vec![ShapeBucket { dim_name: "N".into(), lo: 0, hi: 256 }],
            best_config: sample_config(),
            perf: 12.34,
            timestamp: 0,
        }
    }

    // ── TuneCache ────────────────────────────────────────────────────

    #[test]
    fn cache_load_nonexistent_returns_default() {
        let c = TuneCache::load(&PathBuf::from("/definitely/nonexistent/path.json"));
        assert!(c.entries.is_empty());
    }

    #[test]
    fn cache_insert_save_load_roundtrip() {
        let dir = scratch_dir();
        let path = dir.join("tuning_cache.json");
        let mut c = TuneCache::default();
        c.insert("kernel_a@N=0..256", sample_entry());
        c.save(&path).unwrap();
        assert!(path.exists());

        let loaded = TuneCache::load(&path);
        assert_eq!(loaded.entries.len(), 1);
        let e = loaded.entries.get("kernel_a@N=0..256").expect("entry survived round-trip");
        assert_eq!(e.bucket[0].dim_name, "N");
        assert_eq!(e.best_config.tile_dims, vec![32, 32, 32]);
        assert_eq!(e.perf, 12.34);
    }

    #[test]
    fn cache_lookup_always_none_today() {
        // lookup() is a placeholder — see the comment in autotune.rs.
        // Pinning the behaviour so a future implementation that flips it
        // gets noticed.
        let c = TuneCache::default();
        let ce = ConstExprValues::new();
        assert!(c.lookup(&ce).is_none());
    }

    #[test]
    fn cache_save_creates_parent_dirs() {
        let dir = scratch_dir();
        let nested = dir.join("a").join("b").join("c").join("cache.json");
        let c = TuneCache::default();
        c.save(&nested).expect("save should mkdir -p the parents");
        assert!(nested.exists());
    }

    // ── Autotuner ────────────────────────────────────────────────────

    #[test]
    fn autotuner_get_or_tune_disabled_returns_default_config() {
        let dir = scratch_dir();
        let mut tuner = Autotuner::new(dir, false);
        let ce = ConstExprValues::new();
        let cfg = tuner.get_or_tune("any_kernel", &ce).expect("disabled tuner returns default");
        assert_eq!(cfg.tile_dims, vec![32, 32, 32]);
        assert_eq!(cfg.threads, (256, 1, 1));
        assert!(cfg.use_simd_matrix);
    }

    #[test]
    fn autotuner_get_or_tune_enabled_with_empty_cache_returns_none() {
        let dir = scratch_dir();
        let mut t = Autotuner::new(dir, true);
        let ce = ConstExprValues::new();
        // Cache empty + lookup always returns None today → get_or_tune
        // falls through the placeholder TODO branch.
        assert!(t.get_or_tune("any_kernel", &ce).is_none());
    }

    #[test]
    fn autotuner_set_enabled_flips_state() {
        let dir = scratch_dir();
        let mut t = Autotuner::new(dir, false);
        let ce = ConstExprValues::new();
        assert!(t.get_or_tune("k", &ce).is_some()); // disabled → default
        t.set_enabled(true);
        assert!(t.get_or_tune("k", &ce).is_none()); // enabled, empty cache → None
    }

    #[test]
    fn autotuner_flush_writes_cache_file() {
        let dir = scratch_dir();
        let t = Autotuner::new(dir.clone(), false);
        t.flush().expect("flush succeeds even on empty cache");
        assert!(dir.join("tuning_cache.json").exists());
    }

    #[test]
    fn default_cache_dir_uses_home_or_falls_back() {
        // We can't reliably test the HOME-set path without polluting the
        // environment; just assert the result is non-empty and ends in
        // "metaltile".
        let p = Autotuner::default_cache_dir();
        assert!(p.ends_with("metaltile"));
    }

    // ── ShapeBucket / TuneConfig / TuneEntry serde ───────────────────

    #[test]
    fn shape_bucket_serde_roundtrip() {
        let b = ShapeBucket { dim_name: "M".into(), lo: 0, hi: 128 };
        let s = serde_json::to_string(&b).unwrap();
        let b2: ShapeBucket = serde_json::from_str(&s).unwrap();
        assert_eq!(b, b2);
    }

    #[test]
    fn tune_config_serde_roundtrip() {
        let c = sample_config();
        let s = serde_json::to_string(&c).unwrap();
        let c2: TuneConfig = serde_json::from_str(&s).unwrap();
        assert_eq!(c2.tile_dims, c.tile_dims);
        assert_eq!(c2.threads, c.threads);
        assert_eq!(c2.unroll_factor, c.unroll_factor);
        assert_eq!(c2.use_simd_matrix, c.use_simd_matrix);
        assert_eq!(c2.use_async_copy, c.use_async_copy);
    }
}
