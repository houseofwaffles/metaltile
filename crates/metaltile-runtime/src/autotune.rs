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
            return Some(entry.best_config.clone());
        }

        // TODO: actually run tuning (benchmark different configs)
        None
    }

    /// Persist the cache to disk.
    pub fn flush(&self) -> std::io::Result<()> { self.cache.save(&self.cache_path) }
}

fn dirs_next() -> Option<PathBuf> { std::env::var("HOME").ok().map(PathBuf::from) }
