pub mod arange;
pub mod arg_reduce;
pub mod binary;
pub mod binary_two;
pub mod conv;
pub mod copy;
pub mod fence;
pub mod fft;
pub mod fp_quantized;
#[cfg(feature = "nax")]
pub mod fp_quantized_nax;
pub mod gemv;
pub mod gemv_masked;
pub mod layer_norm;
pub mod logsumexp;
pub mod quantized;
#[cfg(feature = "nax")]
pub mod quantized_nax;
pub mod random;
pub mod reduce;
pub mod rms_norm;
pub mod rope;
pub mod scaled_dot_product_attention;
pub mod scan;
pub mod softmax;
pub mod sort;
pub mod steel;
pub mod strided;
pub mod ternary;
pub mod unary;

pub use crate::bench_types::{
    CorrectnessStatus,
    DEFAULT_MIN_COSINE_SIM,
    DType,
    DtypeCtx,
    EquivResult,
    EquivTolerance,
    FLOAT_DTYPE_STRS,
    FLOAT_DTYPES,
    INTEGER_DTYPES,
    OpBench,
    OpResult,
    SuitePrinter,
    dtype_label,
    dtype_tol,
    dtype_tol_reduce,
    elem_bytes,
    generate_elementwise_msl,
    generate_reduction_msl,
    mlx_tname,
    print_suite,
    quantize_roundtrip,
    set_result_reporter,
    validate_results,
};
