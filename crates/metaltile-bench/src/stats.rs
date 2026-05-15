/// Summary statistics for a set of GPU timing measurements.
#[derive(Debug, Clone)]
pub struct BenchStats {
    /// Mean GPU execution time in microseconds.
    pub mean_us: f64,
    /// Median (p50) GPU execution time in microseconds.
    pub median_us: f64,
    /// 95th-percentile GPU execution time in microseconds.
    pub p95_us: f64,
    /// 99th-percentile GPU execution time in microseconds.
    pub p99_us: f64,
    /// Standard deviation in microseconds.
    pub stddev_us: f64,
    /// Coefficient of variation (stddev/mean × 100). >5% suggests instability.
    pub cv_pct: f64,
}

impl BenchStats {
    pub fn from_samples(mut samples: Vec<f64>) -> Self {
        assert!(!samples.is_empty());
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = samples.len();
        let mean = samples.iter().sum::<f64>() / n as f64;
        let median = samples[n / 2];
        let p95 = samples[(n * 95 / 100).min(n - 1)];
        let p99 = samples[(n * 99 / 100).min(n - 1)];
        let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let stddev = variance.sqrt();
        let cv_pct = if mean > 0.0 { stddev / mean * 100.0 } else { 0.0 };
        BenchStats {
            mean_us: mean,
            median_us: median,
            p95_us: p95,
            p99_us: p99,
            stddev_us: stddev,
            cv_pct,
        }
    }

    /// True if timing data came from a real GPU dispatch (non-macOS always returns false).
    pub fn is_valid(&self) -> bool { self.mean_us > 0.0 }
}

#[cfg(test)]
mod tests {
    use super::BenchStats;

    #[test]
    fn computes_median_and_tail_percentiles() {
        let st = BenchStats::from_samples(vec![1.0, 2.0, 3.0, 4.0, 10.0]);
        assert_eq!(st.median_us, 3.0);
        assert_eq!(st.p95_us, 10.0);
        assert_eq!(st.p99_us, 10.0);
        assert!(st.stddev_us > 0.0);
    }
}
