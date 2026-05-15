// ===== begin: /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/reduce.metal =====
#line 1 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/reduce.metal"
// Copyright © 2024 Apple Inc.

#include <metal_atomic>
#include <metal_simdgroup>

// clang-format off
// ----- expanded "mlx/backend/metal/kernels/defines.h" from /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/reduce.metal:7 -----
// ===== begin: /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/defines.h =====
#line 1 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/defines.h"
// Copyright © 2023 Apple Inc.

#pragma once

#if defined __METAL__ || defined MLX_METAL_JIT
#define MTL_CONST constant
#else
#define MTL_CONST
#endif

static MTL_CONST constexpr int MAX_REDUCE_SPECIALIZED_DIMS = 4;
static MTL_CONST constexpr int REDUCE_N_READS = 4;
static MTL_CONST constexpr int REDUCE_N_WRITES = 4;
static MTL_CONST constexpr int SOFTMAX_N_READS = 4;
static MTL_CONST constexpr int RMS_N_READS = 4;
static MTL_CONST constexpr int RMS_LOOPED_LIMIT = 4096;

// Instantiate a templated kernel.
// Extra args are used as template parameters:
// e.g. instantiate_kernel(binary_int, binary, a, b) ->
// [[host_name(binary_int)]] [kernel] binary<a, b>
#define instantiate_kernel(name, func, ...) \
  template [[host_name(                     \
      name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;
// ===== end:   /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/defines.h =====
#line 8 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/reduce.metal"
// ----- expanded "mlx/backend/metal/kernels/utils.h" from /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/reduce.metal:8 -----
// ===== begin: /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/utils.h =====
#line 1 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/utils.h"
// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <metal_math>

// ----- expanded "mlx/backend/metal/kernels/bf16.h" from /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/utils.h:7 -----
// ===== begin: /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/bf16.h =====
#line 1 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/bf16.h"
// Copyright © 2023 Apple Inc.

#pragma once

#include <metal_stdlib>

using namespace metal;

typedef bfloat bfloat16_t;
inline uint16_t bfloat16_to_uint16(const bfloat16_t x) {
  return as_type<uint16_t>(x);
}

inline bfloat16_t uint16_to_bfloat16(const uint16_t x) {
  return as_type<bfloat16_t>(x);
}
// ===== end:   /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/bf16.h =====
#line 8 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/utils.h"
// ----- expanded "mlx/backend/metal/kernels/bf16_math.h" from /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/utils.h:8 -----
// ===== begin: /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/bf16_math.h =====
#line 1 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/bf16_math.h"
// Copyright © 2023 Apple Inc.

#pragma once

///////////////////////////////////////////////////////////////////////////////
// Metal math for bfloat16
///////////////////////////////////////////////////////////////////////////////

/*

Following the Metal Shading Language Specification (Metal 3.1)

"bfloat is an extended itypeing point type that only allows implicit conversion
 to a type of greater itypeing point rank. While bfloat can be implicitly
 converted to itype, it cannot be implicitly converted to half, and neither
 itype nor half can be implicitly converted to bfloat."

Further, as far as I can tell, the stdlib math/simd functions are not defined
for bfloat and calling with an argument of type bfloat will result in that
argument getting implicitly converted to itype which then returns an output
that is (likely) a itype which cannot be implicitly converted into a bfloat

This leads to situations where
bfloat a = 5.0bf;
bfloat b = metal::abs(a); // this will throw an error since abs return itype
bfloat c = static_cast<bfloat>(metal::abs(a)); // this is fine

For the moment, I will be adding overloaded instantiations of the math
functions to accordingly automatically handle the casting

*/

#define instantiate_metal_math_funcs(itype, otype, ctype, mfast)               \
                                                                               \
  METAL_FUNC otype abs(itype x) {                                              \
    return static_cast<otype>(__metal_fabs(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype acos(itype x) {                                             \
    return static_cast<otype>(__metal_acos(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype acosh(itype x) {                                            \
    return static_cast<otype>(__metal_acosh(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype asin(itype x) {                                             \
    return static_cast<otype>(__metal_asin(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype asinh(itype x) {                                            \
    return static_cast<otype>(__metal_asinh(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype atan(itype y_over_x) {                                      \
    return static_cast<otype>(                                                 \
        __metal_atan(static_cast<ctype>(y_over_x), mfast));                    \
  }                                                                            \
  METAL_FUNC otype atan2(itype y, itype x) {                                   \
    return static_cast<otype>(                                                 \
        __metal_atan2(static_cast<ctype>(y), static_cast<ctype>(x), mfast));   \
  }                                                                            \
  METAL_FUNC otype atanh(itype x) {                                            \
    return static_cast<otype>(__metal_atanh(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype ceil(itype x) {                                             \
    return static_cast<otype>(__metal_ceil(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype cos(itype x) {                                              \
    return static_cast<otype>(__metal_cos(static_cast<ctype>(x), mfast));      \
  }                                                                            \
  METAL_FUNC otype cosh(itype x) {                                             \
    return static_cast<otype>(__metal_cosh(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype cospi(itype x) {                                            \
    return static_cast<otype>(__metal_cospi(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype divide(itype x, itype y) {                                  \
    return static_cast<otype>(                                                 \
        __metal_divide(static_cast<ctype>(x), static_cast<ctype>(y), mfast));  \
  }                                                                            \
  METAL_FUNC otype exp(itype x) {                                              \
    return static_cast<otype>(__metal_exp(static_cast<ctype>(x), mfast));      \
  }                                                                            \
  METAL_FUNC otype exp10(itype x) {                                            \
    return static_cast<otype>(__metal_exp10(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype exp2(itype x) {                                             \
    return static_cast<otype>(__metal_exp2(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype fabs(itype x) {                                             \
    return static_cast<otype>(__metal_fabs(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype fdim(itype x, itype y) {                                    \
    ctype t = static_cast<ctype>(x - y);                                       \
    return static_cast<otype>(select(t, ctype(0), t < ctype(0) || x == y));    \
  }                                                                            \
  METAL_FUNC otype floor(itype x) {                                            \
    return static_cast<otype>(__metal_floor(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype fma(itype x, itype y, itype z) {                            \
    return static_cast<otype>(__metal_fma(                                     \
        static_cast<ctype>(x), static_cast<ctype>(y), static_cast<ctype>(z))); \
  }                                                                            \
  METAL_FUNC otype fmax(itype x, itype y) {                                    \
    return static_cast<otype>(                                                 \
        __metal_fmax(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype fmax3(itype x, itype y, itype z) {                          \
    return static_cast<otype>(__metal_fmax3(                                   \
        static_cast<ctype>(x),                                                 \
        static_cast<ctype>(y),                                                 \
        static_cast<ctype>(z),                                                 \
        mfast));                                                               \
  }                                                                            \
  METAL_FUNC otype fmedian3(itype x, itype y, itype z) {                       \
    return static_cast<otype>(__metal_fmedian3(                                \
        static_cast<ctype>(x),                                                 \
        static_cast<ctype>(y),                                                 \
        static_cast<ctype>(z),                                                 \
        mfast));                                                               \
  }                                                                            \
  METAL_FUNC otype fmin(itype x, itype y) {                                    \
    return static_cast<otype>(                                                 \
        __metal_fmin(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype fmin3(itype x, itype y, itype z) {                          \
    return static_cast<otype>(__metal_fmin3(                                   \
        static_cast<ctype>(x),                                                 \
        static_cast<ctype>(y),                                                 \
        static_cast<ctype>(z),                                                 \
        mfast));                                                               \
  }                                                                            \
  METAL_FUNC otype fmod(itype x, itype y) {                                    \
    return static_cast<otype>(                                                 \
        __metal_fmod(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype fract(itype x) {                                            \
    return static_cast<otype>(__metal_fract(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype frexp(itype x, thread int& exp) {                           \
    return static_cast<otype>(__metal_frexp(static_cast<ctype>(x), &exp));     \
  }                                                                            \
  METAL_FUNC otype ldexp(itype x, int k) {                                     \
    return static_cast<otype>(__metal_ldexp(static_cast<ctype>(x), k, mfast)); \
  }                                                                            \
  METAL_FUNC otype log(itype x) {                                              \
    return static_cast<otype>(__metal_log(static_cast<ctype>(x), mfast));      \
  }                                                                            \
  METAL_FUNC otype log10(itype x) {                                            \
    return static_cast<otype>(__metal_log10(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype log2(itype x) {                                             \
    return static_cast<otype>(__metal_log2(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype max(itype x, itype y) {                                     \
    return static_cast<otype>(                                                 \
        __metal_fmax(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype max3(itype x, itype y, itype z) {                           \
    return static_cast<otype>(__metal_fmax3(                                   \
        static_cast<ctype>(x),                                                 \
        static_cast<ctype>(y),                                                 \
        static_cast<ctype>(z),                                                 \
        mfast));                                                               \
  }                                                                            \
  METAL_FUNC otype median3(itype x, itype y, itype z) {                        \
    return static_cast<otype>(__metal_fmedian3(                                \
        static_cast<ctype>(x),                                                 \
        static_cast<ctype>(y),                                                 \
        static_cast<ctype>(z),                                                 \
        mfast));                                                               \
  }                                                                            \
  METAL_FUNC otype min(itype x, itype y) {                                     \
    return static_cast<otype>(                                                 \
        __metal_fmin(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype min3(itype x, itype y, itype z) {                           \
    return static_cast<otype>(__metal_fmin3(                                   \
        static_cast<ctype>(x),                                                 \
        static_cast<ctype>(y),                                                 \
        static_cast<ctype>(z),                                                 \
        mfast));                                                               \
  }                                                                            \
  METAL_FUNC otype nextafter(itype x, itype y) {                               \
    return static_cast<otype>(                                                 \
        __metal_nextafter(static_cast<ctype>(x), static_cast<ctype>(y)));      \
  }                                                                            \
  METAL_FUNC otype pow(itype x, itype y) {                                     \
    return static_cast<otype>(                                                 \
        __metal_pow(static_cast<ctype>(x), static_cast<ctype>(y), mfast));     \
  }                                                                            \
  METAL_FUNC otype powr(itype x, itype y) {                                    \
    return static_cast<otype>(                                                 \
        __metal_powr(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype rint(itype x) {                                             \
    return static_cast<otype>(__metal_rint(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype round(itype x) {                                            \
    return static_cast<otype>(__metal_round(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype rsqrt(itype x) {                                            \
    return static_cast<otype>(__metal_rsqrt(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype sin(itype x) {                                              \
    return static_cast<otype>(__metal_sin(static_cast<ctype>(x), mfast));      \
  }                                                                            \
  METAL_FUNC otype sinh(itype x) {                                             \
    return static_cast<otype>(__metal_sinh(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype sinpi(itype x) {                                            \
    return static_cast<otype>(__metal_sinpi(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype sqrt(itype x) {                                             \
    return static_cast<otype>(__metal_sqrt(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype tan(itype x) {                                              \
    return static_cast<otype>(__metal_tan(static_cast<ctype>(x), mfast));      \
  }                                                                            \
  METAL_FUNC otype tanh(itype x) {                                             \
    return static_cast<otype>(__metal_tanh(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype tanpi(itype x) {                                            \
    return static_cast<otype>(__metal_tanpi(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype trunc(itype x) {                                            \
    return static_cast<otype>(__metal_trunc(static_cast<ctype>(x), mfast));    \
  }

namespace metal {

instantiate_metal_math_funcs(
    bfloat16_t,
    bfloat16_t,
    float,
    __METAL_MAYBE_FAST_MATH__);

namespace fast {

instantiate_metal_math_funcs(
    bfloat16_t,
    bfloat16_t,
    float,
    __METAL_FAST_MATH__);

} // namespace fast

namespace precise {

instantiate_metal_math_funcs(
    bfloat16_t,
    bfloat16_t,
    float,
    __METAL_PRECISE_MATH__);

} // namespace precise

} // namespace metal

///////////////////////////////////////////////////////////////////////////////
// Metal simd for bfloat16
///////////////////////////////////////////////////////////////////////////////

#define instantiate_metal_simd_comm_funcs(                                   \
    itype, otype, ctype, itype_to_ctype, ctype_to_otype)                     \
                                                                             \
  METAL_FUNC otype simd_broadcast(itype data, ushort broadcast_lane_id) {    \
    return ctype_to_otype(                                                   \
        __metal_simd_broadcast(itype_to_ctype(data), broadcast_lane_id));    \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle(itype data, ushort simd_lane_id) {           \
    return ctype_to_otype(                                                   \
        __metal_simd_shuffle(itype_to_ctype(data), simd_lane_id));           \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_and_fill_down(                               \
      itype data, itype filling_data, ushort delta, ushort modulo) {         \
    return ctype_to_otype(__metal_simd_shuffle_and_fill_down(                \
        itype_to_ctype(data), itype_to_ctype(filling_data), delta, modulo)); \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_and_fill_down(                               \
      itype data, itype filling_data, ushort delta) {                        \
    return ctype_to_otype(__metal_simd_shuffle_and_fill_down(                \
        itype_to_ctype(data),                                                \
        itype_to_ctype(filling_data),                                        \
        delta,                                                               \
        __metal_get_simdgroup_size(ushort())));                              \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_and_fill_up(                                 \
      itype data, itype filling_data, ushort delta, ushort modulo) {         \
    return ctype_to_otype(__metal_simd_shuffle_and_fill_up(                  \
        itype_to_ctype(data), itype_to_ctype(filling_data), delta, modulo)); \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_and_fill_up(                                 \
      itype data, itype filling_data, ushort delta) {                        \
    return ctype_to_otype(__metal_simd_shuffle_and_fill_up(                  \
        itype_to_ctype(data),                                                \
        itype_to_ctype(filling_data),                                        \
        delta,                                                               \
        __metal_get_simdgroup_size(ushort())));                              \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_down(itype data, ushort delta) {             \
    return ctype_to_otype(                                                   \
        __metal_simd_shuffle_down(itype_to_ctype(data), delta));             \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_rotate_down(itype data, ushort delta) {      \
    return ctype_to_otype(                                                   \
        __metal_simd_shuffle_rotate_down(itype_to_ctype(data), delta));      \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_rotate_up(itype data, ushort delta) {        \
    return ctype_to_otype(                                                   \
        __metal_simd_shuffle_rotate_up(itype_to_ctype(data), delta));        \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_up(itype data, ushort delta) {               \
    return ctype_to_otype(                                                   \
        __metal_simd_shuffle_up(itype_to_ctype(data), delta));               \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_xor(itype data, ushort mask) {               \
    return ctype_to_otype(                                                   \
        __metal_simd_shuffle_xor(itype_to_ctype(data), mask));               \
  }

#define instantiate_metal_simd_reduction_funcs(itype, otype, ctype)            \
                                                                               \
  METAL_FUNC otype simd_max(itype data) {                                      \
    return static_cast<otype>(__metal_simd_max(static_cast<ctype>(data)));     \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_min(itype data) {                                      \
    return static_cast<otype>(__metal_simd_min(static_cast<ctype>(data)));     \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_prefix_exclusive_product(itype data) {                 \
    return static_cast<otype>(                                                 \
        __metal_simd_prefix_exclusive_product(static_cast<ctype>(data)));      \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_prefix_exclusive_sum(itype data) {                     \
    return static_cast<otype>(                                                 \
        __metal_simd_prefix_exclusive_sum(static_cast<ctype>(data)));          \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_prefix_inclusive_product(itype data) {                 \
    return static_cast<otype>(                                                 \
        __metal_simd_prefix_inclusive_product(static_cast<ctype>(data)));      \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_prefix_inclusive_sum(itype data) {                     \
    return static_cast<otype>(                                                 \
        __metal_simd_prefix_inclusive_sum(static_cast<ctype>(data)));          \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_product(itype data) {                                  \
    return static_cast<otype>(__metal_simd_product(static_cast<ctype>(data))); \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_sum(itype data) {                                      \
    return static_cast<otype>(__metal_simd_sum(static_cast<ctype>(data)));     \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_xor(itype data) {                                      \
    return static_cast<otype>(__metal_simd_xor(static_cast<ctype>(data)));     \
  }

namespace metal {

instantiate_metal_simd_comm_funcs(
    bfloat16_t,
    bfloat16_t,
    uint16_t,
    bfloat16_to_uint16,
    uint16_to_bfloat16);
instantiate_metal_simd_reduction_funcs(bfloat16_t, bfloat16_t, float);

} // namespace metal
// ===== end:   /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/bf16_math.h =====
#line 9 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/utils.h"
// ----- expanded "mlx/backend/metal/kernels/complex.h" from /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/utils.h:9 -----
// ===== begin: /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/complex.h =====
#line 1 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/complex.h"
// Copyright © 2023 Apple Inc.

#pragma once

#include <metal_stdlib>

using namespace metal;

struct complex64_t;

template <typename T>
static constexpr constant bool can_convert_to_complex64 =
    !is_same_v<T, complex64_t> && is_convertible_v<T, float>;

template <typename T>
static constexpr constant bool can_convert_from_complex64 =
    !is_same_v<T, complex64_t> &&
    (is_convertible_v<float, T> || is_convertible_v<bfloat16_t, T>);

struct complex64_t {
  float real;
  float imag;

  // Constructors
  constexpr complex64_t(float real, float imag) : real(real), imag(imag) {};
  constexpr complex64_t() : real(0), imag(0) {};
  constexpr complex64_t() threadgroup : real(0), imag(0) {};

  // Conversions to complex64_t
  template <
      typename T,
      typename = typename enable_if<can_convert_to_complex64<T>>::type>
  constexpr complex64_t(T x) thread : real(x), imag(0) {}

  template <
      typename T,
      typename = typename enable_if<can_convert_to_complex64<T>>::type>
  constexpr complex64_t(T x) threadgroup : real(x), imag(0) {}

  template <
      typename T,
      typename = typename enable_if<can_convert_to_complex64<T>>::type>
  constexpr complex64_t(T x) device : real(x), imag(0) {}

  template <
      typename T,
      typename = typename enable_if<can_convert_to_complex64<T>>::type>
  constexpr complex64_t(T x) constant : real(x), imag(0) {}

  // Conversions from complex64_t
  template <
      typename T,
      typename = typename enable_if<can_convert_from_complex64<T>>::type>
  constexpr operator T() const thread {
    return static_cast<T>(real);
  }

  template <
      typename T,
      typename = typename enable_if<can_convert_from_complex64<T>>::type>
  constexpr operator T() const threadgroup {
    return static_cast<T>(real);
  }

  template <
      typename T,
      typename = typename enable_if<can_convert_from_complex64<T>>::type>
  constexpr operator T() const device {
    return static_cast<T>(real);
  }

  template <
      typename T,
      typename = typename enable_if<can_convert_from_complex64<T>>::type>
  constexpr operator T() const constant {
    return static_cast<T>(real);
  }
};

constexpr complex64_t operator-(complex64_t x) {
  return {-x.real, -x.imag};
}

constexpr bool operator>=(complex64_t a, complex64_t b) {
  return (a.real > b.real) || (a.real == b.real && a.imag >= b.imag);
}

constexpr bool operator>(complex64_t a, complex64_t b) {
  return (a.real > b.real) || (a.real == b.real && a.imag > b.imag);
}

constexpr bool operator<=(complex64_t a, complex64_t b) {
  return operator>=(b, a);
}

constexpr bool operator<(complex64_t a, complex64_t b) {
  return operator>(b, a);
}

constexpr bool operator==(complex64_t a, complex64_t b) {
  return a.real == b.real && a.imag == b.imag;
}

constexpr complex64_t operator+(complex64_t a, complex64_t b) {
  return {a.real + b.real, a.imag + b.imag};
}

constexpr thread complex64_t& operator+=(thread complex64_t& a, complex64_t b) {
  a.real += b.real;
  a.imag += b.imag;
  return a;
}

constexpr threadgroup complex64_t& operator+=(
    threadgroup complex64_t& a,
    complex64_t b) {
  a.real += b.real;
  a.imag += b.imag;
  return a;
}

constexpr device complex64_t& operator+=(device complex64_t& a, complex64_t b) {
  a.real += b.real;
  a.imag += b.imag;
  return a;
}

constexpr complex64_t operator+(float a, complex64_t b) {
  return {a + b.real, b.imag};
}
constexpr complex64_t operator+(complex64_t a, float b) {
  return {a.real + b, a.imag};
}

constexpr complex64_t operator-(complex64_t a, complex64_t b) {
  return {a.real - b.real, a.imag - b.imag};
}
constexpr complex64_t operator-(float a, complex64_t b) {
  return {a - b.real, -b.imag};
}
constexpr complex64_t operator-(complex64_t a, float b) {
  return {a.real - b, a.imag};
}

constexpr complex64_t operator*(complex64_t a, complex64_t b) {
  return {a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real};
}

constexpr complex64_t operator/(complex64_t a, complex64_t b) {
  auto denom = b.real * b.real + b.imag * b.imag;
  auto x = a.real * b.real + a.imag * b.imag;
  auto y = a.imag * b.real - a.real * b.imag;
  return {x / denom, y / denom};
}

constexpr complex64_t operator/(float a, complex64_t b) {
  auto denom = b.real * b.real + b.imag * b.imag;
  auto x = a * b.real;
  auto y = -a * b.imag;
  return {x / denom, y / denom};
}

constexpr complex64_t operator%(complex64_t a, complex64_t b) {
  auto real = a.real - (b.real * static_cast<int64_t>(a.real / b.real));
  auto imag = a.imag - (b.imag * static_cast<int64_t>(a.imag / b.imag));
  if (real != 0 && (real < 0 != b.real < 0)) {
    real += b.real;
  }
  if (imag != 0 && (imag < 0 != b.imag < 0)) {
    imag += b.imag;
  }
  return {real, imag};
}
// ===== end:   /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/complex.h =====
#line 10 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/utils.h"
// ----- expanded "mlx/backend/metal/kernels/defines.h" from /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/utils.h:10 -----
// [metal_flatten] skipped duplicate include: /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/defines.h
#line 11 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/utils.h"
// ----- expanded "mlx/backend/metal/kernels/logging.h" from /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/utils.h:11 -----
// ===== begin: /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/logging.h =====
#line 1 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/logging.h"
// Copyright © 2025 Apple Inc.

#pragma once

#if defined(__METAL_VERSION__) && (__METAL_VERSION__ >= 320)
#include <metal_logging>

namespace mlx {
using os_log = metal::os_log;
} // namespace mlx

#else

namespace mlx {
struct os_log {
  constexpr os_log(constant char*, constant char*) constant {}

  template <typename... Args>
  void log_debug(constant char*, Args...) const {}

  template <typename... Args>
  void log_debug(constant char*, Args...) const constant {}
};
} // namespace mlx

#endif// ===== end:   /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/logging.h =====
#line 12 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/utils.h"

typedef half float16_t;

// Work per thread values for different types. The values here are expected to
// match get_work_per_thread in mlx/backend/metal/utils.h
template <typename U>
struct WorkPerThread {
  static_assert(sizeof(U) <= 8, "Type too large");
  static constexpr int constant n = 8 / sizeof(U);
};

///////////////////////////////////////////////////////////////////////////////
// Type limits utils
///////////////////////////////////////////////////////////////////////////////

template <typename U>
struct Limits {
  static const constant U max = metal::numeric_limits<U>::max();
  static const constant U min = metal::numeric_limits<U>::min();
  static const constant U finite_max = metal::numeric_limits<U>::max();
  static const constant U finite_min = metal::numeric_limits<U>::min();
};

#define instantiate_default_limit(type)                                      \
  template <>                                                                \
  struct Limits<type> {                                                      \
    static constexpr constant type max = metal::numeric_limits<type>::max(); \
    static constexpr constant type min = metal::numeric_limits<type>::min(); \
    static constexpr constant type finite_max =                              \
        metal::numeric_limits<type>::max();                                  \
    static constexpr constant type finite_min =                              \
        metal::numeric_limits<type>::min();                                  \
  };

instantiate_default_limit(uint8_t);
instantiate_default_limit(uint16_t);
instantiate_default_limit(uint32_t);
instantiate_default_limit(uint64_t);
instantiate_default_limit(int8_t);
instantiate_default_limit(int16_t);
instantiate_default_limit(int32_t);
instantiate_default_limit(int64_t);

#define instantiate_float_limit(type)             \
  template <>                                     \
  struct Limits<type> {                           \
    static constexpr constant type max =          \
        metal::numeric_limits<type>::infinity();  \
    static constexpr constant type min =          \
        -metal::numeric_limits<type>::infinity(); \
    static constexpr constant type finite_max =   \
        metal::numeric_limits<type>::max();       \
    static constexpr constant type finite_min =   \
        -metal::numeric_limits<type>::max();      \
  };

instantiate_float_limit(half);
instantiate_float_limit(float);
instantiate_float_limit(bfloat16_t);

template <>
struct Limits<bool> {
  static constexpr constant bool max = true;
  static constexpr constant bool min = false;
};

template <>
struct Limits<complex64_t> {
  static constexpr constant complex64_t max = complex64_t(
      metal::numeric_limits<float>::infinity(),
      metal::numeric_limits<float>::infinity());
  static constexpr constant complex64_t min = complex64_t(
      -metal::numeric_limits<float>::infinity(),
      -metal::numeric_limits<float>::infinity());
};

///////////////////////////////////////////////////////////////////////////////
// Indexing utils
///////////////////////////////////////////////////////////////////////////////

#define MLX_MTL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

///////////////////////////////////////////////////////////////////////////////
// Single Array with generic dims

template <typename IdxT = int64_t>
METAL_FUNC IdxT elem_to_loc(
    IdxT elem,
    constant const int* shape,
    constant const int64_t* strides,
    int ndim) {
  IdxT loc = 0;
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    loc += (elem % shape[i]) * IdxT(strides[i]);
    elem /= shape[i];
  }
  return loc;
}

// Non templated version to handle arbitrary dims
template <typename IdxT = int64_t>
METAL_FUNC IdxT elem_to_loc(
    uint3 elem,
    constant const int* shape,
    constant const int64_t* strides,
    int ndim) {
  IdxT loc =
      elem.x * IdxT(strides[ndim - 1]) + elem.y * IdxT(strides[ndim - 2]);
  for (int d = ndim - 3; d >= 0; --d) {
    loc += (elem.z % shape[d]) * IdxT(strides[d]);
    elem.z /= shape[d];
  }
  return loc;
}

///////////////////////////////////////////////////////////////////////////////
// Single Array with fixed N dims

template <typename IdxT = int64_t>
METAL_FUNC IdxT elem_to_loc_1(uint elem, constant const int64_t& stride) {
  return elem * IdxT(stride);
}

template <typename IdxT = int64_t>
METAL_FUNC IdxT elem_to_loc_2(uint2 elem, constant const int64_t strides[2]) {
  return elem.x * IdxT(strides[1]) + elem.y * IdxT(strides[0]);
}

template <typename IdxT = int64_t>
METAL_FUNC IdxT elem_to_loc_3(uint3 elem, constant const int64_t strides[3]) {
  return elem.x * IdxT(strides[2]) + elem.y * IdxT(strides[1]) +
      elem.z * IdxT(strides[0]);
}

///////////////////////////////////////////////////////////////////////////////
// Multiple Arrays with generic dims

template <typename IdxT = int64_t>
METAL_FUNC vec<IdxT, 2> elem_to_loc_2_nd(
    uint3 elem,
    constant const int* shape,
    constant const int64_t* a_strides,
    constant const int64_t* b_strides,
    int ndim) {
  vec<IdxT, 2> loc = {
      IdxT(
          elem.x * IdxT(a_strides[ndim - 1]) +
          IdxT(elem.y) * IdxT(a_strides[ndim - 2])),
      IdxT(
          elem.x * IdxT(b_strides[ndim - 1]) +
          elem.y * IdxT(b_strides[ndim - 2]))};
  for (int d = ndim - 3; d >= 0; --d) {
    uint l = elem.z % shape[d];
    loc.x += l * IdxT(a_strides[d]);
    loc.y += l * IdxT(b_strides[d]);
    elem.z /= shape[d];
  }
  return loc;
}

template <typename IdxT = int64_t>
METAL_FUNC vec<IdxT, 3> elem_to_loc_3_nd(
    uint3 elem,
    constant const int* shape,
    constant const int64_t* a_strides,
    constant const int64_t* b_strides,
    constant const int64_t* c_strides,
    int ndim) {
  vec<IdxT, 3> loc = {
      IdxT(elem.x * IdxT(a_strides[ndim - 1])) +
          IdxT(elem.y * IdxT(a_strides[ndim - 2])),
      IdxT(elem.x * IdxT(b_strides[ndim - 1])) +
          IdxT(elem.y * IdxT(b_strides[ndim - 2])),
      IdxT(elem.x * IdxT(c_strides[ndim - 1])) +
          IdxT(elem.y * IdxT(c_strides[ndim - 2]))};
  for (int d = ndim - 3; d >= 0; --d) {
    uint l = elem.z % shape[d];
    loc.x += l * IdxT(a_strides[d]);
    loc.y += l * IdxT(b_strides[d]);
    loc.z += l * IdxT(c_strides[d]);
    elem.z /= shape[d];
  }
  return loc;
}

///////////////////////////////////////////////////////////////////////////////
// Elem to loc in a loop utils
///////////////////////////////////////////////////////////////////////////////

template <int DIM, typename OffsetT = size_t, bool General = true>
struct LoopedElemToLoc {
  int dim;
  LoopedElemToLoc<DIM - 1, OffsetT, General> inner_looper;
  OffsetT offset{0};
  int index{0};

  LoopedElemToLoc(int dim) : dim(dim), inner_looper(dim - 1) {}

  void next(const constant int* shape, const constant int64_t* strides) {
    if (dim == 0) {
      return;
    }
    index++;
    offset += OffsetT(strides[dim - 1]);
    if (index >= shape[dim - 1]) {
      index = 0;
      inner_looper.next(shape, strides);
      offset = inner_looper.offset;
    }
  }

  void next(int n, const constant int* shape, const constant int64_t* strides) {
    if (dim == 0) {
      return;
    }
    index += n;
    offset += n * OffsetT(strides[dim - 1]);

    if (index >= shape[dim - 1]) {
      int extra = index - shape[dim - 1];
      if (extra >= shape[dim - 1]) {
        inner_looper.next(1 + extra / shape[dim - 1], shape, strides);
        extra = extra % shape[dim - 1];
      } else {
        inner_looper.next(shape, strides);
      }
      index = 0;
      offset = inner_looper.offset;
      if (extra > 0) {
        next(extra, shape, strides);
      }
    }
  }

  OffsetT location() {
    return offset;
  }
};

template <typename OffsetT>
struct LoopedElemToLoc<1, OffsetT, true> {
  int dim;
  OffsetT offset{0};
  uint index{0};

  LoopedElemToLoc(int dim) : dim(dim) {}

  void next(const constant int* shape, const constant int64_t* strides) {
    index++;
    if (dim > 1) {
      offset = elem_to_loc<OffsetT>(index, shape, strides, dim);
    } else {
      offset += OffsetT(strides[0]);
    }
  }

  void next(int n, const constant int* shape, const constant int64_t* strides) {
    index += n;
    if (dim > 1) {
      offset = elem_to_loc<OffsetT>(index, shape, strides, dim);
    } else {
      offset = index * OffsetT(strides[0]);
    }
  }

  OffsetT location() {
    return offset;
  }
};

template <typename OffsetT>
struct LoopedElemToLoc<1, OffsetT, false> {
  OffsetT offset{0};

  LoopedElemToLoc(int) {}

  void next(const constant int*, const constant int64_t* strides) {
    offset += OffsetT(strides[0]);
  }

  void next(int n, const constant int*, const constant int64_t* strides) {
    offset += n * OffsetT(strides[0]);
  }

  OffsetT location() {
    return offset;
  }
};

///////////////////////////////////////////////////////////////////////////////
// Calculation utils
///////////////////////////////////////////////////////////////////////////////

/** Compute ceil((float)N/(float)M) */
template <typename T, typename U>
inline T ceildiv(T N, U M) {
  return (N + M - 1) / M;
}

// https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html#1202
inline float log1p(float x) {
  float xp1 = 1.0f + x;
  if (xp1 == Limits<float>::max) {
    return Limits<float>::max;
  }
  if (xp1 == 1.0f) {
    return x;
  }

  return x * (metal::log(xp1) / (xp1 - 1.0f));
}

inline bfloat16_t log1p(bfloat16_t x) {
  float xp1 = 1.0f + static_cast<float>(x);
  if (xp1 == Limits<float>::max) {
    return Limits<bfloat16_t>::max;
  }
  if (xp1 == 1.0f) {
    return x;
  }

  return bfloat16_t(x * (metal::log(xp1) / (xp1 - 1.0f)));
}

inline complex64_t log1p(complex64_t in) {
  float x = in.real;
  float y = in.imag;
  float zabs = metal::precise::sqrt(x * x + y * y);
  float theta = metal::atan2(y, x + 1);
  if (zabs < 0.5f) {
    float r = x * (2 + x) + y * y;
    if (r == 0) { // handle underflow
      return {x, theta};
    }
    return {0.5f * log1p(r), theta};
  } else {
    auto z0 = metal::sqrt((x + 1) * (x + 1) + y * y);
    return {metal::log(z0), theta};
  }
}

///////////////////////////////////////////////////////////////////////////////
// SIMD shuffle ops
///////////////////////////////////////////////////////////////////////////////

inline uint64_t simd_shuffle_down(uint64_t data, uint16_t delta) {
  return as_type<uint64_t>(
      metal::simd_shuffle_down(as_type<uint2>(data), delta));
}

inline int64_t simd_shuffle_down(int64_t data, uint16_t delta) {
  return as_type<int64_t>(
      metal::simd_shuffle_down(as_type<uint2>(data), delta));
}

inline bool simd_shuffle_down(bool data, uint16_t delta) {
  return simd_shuffle_down(static_cast<uint32_t>(data), delta);
}

inline complex64_t simd_shuffle_down(complex64_t data, uint16_t delta) {
  return complex64_t(
      simd_shuffle_down(data.real, delta), simd_shuffle_down(data.imag, delta));
}

inline uint64_t simd_shuffle_up(uint64_t data, uint16_t delta) {
  return as_type<uint64_t>(metal::simd_shuffle_up(as_type<uint2>(data), delta));
}

inline int64_t simd_shuffle_up(int64_t data, uint16_t delta) {
  return as_type<int64_t>(metal::simd_shuffle_up(as_type<uint2>(data), delta));
}

inline bool simd_shuffle_up(bool data, uint16_t delta) {
  return simd_shuffle_up(static_cast<uint32_t>(data), delta);
}

inline complex64_t simd_shuffle_up(complex64_t data, uint16_t delta) {
  return complex64_t(
      simd_shuffle_up(data.real, delta), simd_shuffle_up(data.imag, delta));
}

inline uint64_t
simd_shuffle_and_fill_up(uint64_t data, uint64_t filling, uint16_t delta) {
  return as_type<uint64_t>(metal::simd_shuffle_and_fill_up(
      as_type<uint2>(data), as_type<uint2>(filling), delta));
}

inline int64_t
simd_shuffle_and_fill_up(int64_t data, int64_t filling, uint16_t delta) {
  return as_type<int64_t>(metal::simd_shuffle_and_fill_up(
      as_type<uint2>(data), as_type<uint2>(filling), delta));
}

inline bool simd_shuffle_and_fill_up(bool data, bool filling, uint16_t delta) {
  return simd_shuffle_and_fill_up(
      static_cast<uint32_t>(data), static_cast<uint32_t>(filling), delta);
}

inline complex64_t simd_shuffle_and_fill_up(
    complex64_t data,
    complex64_t filling,
    uint16_t delta) {
  return complex64_t(
      simd_shuffle_and_fill_up(data.real, filling.real, delta),
      simd_shuffle_and_fill_up(data.imag, filling.imag, delta));
}

inline uint64_t simd_shuffle(uint64_t data, uint16_t lane) {
  return as_type<uint64_t>(metal::simd_shuffle(as_type<uint2>(data), lane));
}

inline int64_t simd_shuffle(int64_t data, uint16_t lane) {
  return as_type<int64_t>(metal::simd_shuffle(as_type<uint2>(data), lane));
}

inline bool simd_shuffle(bool data, uint16_t lane) {
  return simd_shuffle(static_cast<uint32_t>(data), lane);
}

inline complex64_t simd_shuffle(complex64_t data, uint16_t lane) {
  return complex64_t(
      simd_shuffle(data.real, lane), simd_shuffle(data.imag, lane));
}

// std::conditional is not included with Metal
template <bool condition, typename T, typename U>
struct ConditionalType {
  using type = U;
};

template <typename T, typename U>
struct ConditionalType<true, T, U> {
  using type = T;
};
// ===== end:   /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/utils.h =====
#line 9 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/reduce.metal"
// ----- expanded "mlx/backend/metal/kernels/atomic.h" from /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/reduce.metal:9 -----
// ===== begin: /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/atomic.h =====
#line 1 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/atomic.h"
// Copyright © 2023 Apple Inc.

#pragma once

#include <metal_atomic>
#include <metal_stdlib>

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
// Atomic utils
///////////////////////////////////////////////////////////////////////////////

#pragma METAL internals : enable
template <typename T>
constexpr constant bool is_metal_atomic = _disjunction<
    is_same<T, int>,
    is_same<T, uint>,
    is_same<T, ulong>,
    is_same<T, float>>::value;

#pragma METAL internals : disable

template <typename T, typename = void>
struct mlx_atomic {
  atomic<uint> val;
};

template <typename T>
struct mlx_atomic<T, enable_if_t<is_metal_atomic<T>>> {
  atomic<T> val;
};

///////////////////////////////////////////////////////////////////////////////
// Native metal atomics
///////////////////////////////////////////////////////////////////////////////

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC T
mlx_atomic_load_explicit(device mlx_atomic<T>* object, size_t offset) {
  return atomic_load_explicit(&(object[offset].val), memory_order_relaxed);
}

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC void
mlx_atomic_store_explicit(device mlx_atomic<T>* object, T val, size_t offset) {
  atomic_store_explicit(&(object[offset].val), val, memory_order_relaxed);
}

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC void mlx_atomic_fetch_and_explicit(
    device mlx_atomic<T>* object,
    T val,
    size_t offset) {
  atomic_fetch_and_explicit(&(object[offset].val), val, memory_order_relaxed);
}

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC void mlx_atomic_fetch_or_explicit(
    device mlx_atomic<T>* object,
    T val,
    size_t offset) {
  atomic_fetch_or_explicit(&(object[offset].val), val, memory_order_relaxed);
}

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC void mlx_atomic_fetch_min_explicit(
    device mlx_atomic<T>* object,
    T val,
    size_t offset) {
  atomic_fetch_min_explicit(&(object[offset].val), val, memory_order_relaxed);
}

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC void mlx_atomic_fetch_max_explicit(
    device mlx_atomic<T>* object,
    T val,
    size_t offset) {
  atomic_fetch_max_explicit(&(object[offset].val), val, memory_order_relaxed);
}

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC void mlx_atomic_fetch_add_explicit(
    device mlx_atomic<T>* object,
    T val,
    size_t offset) {
  atomic_fetch_add_explicit(&(object[offset].val), val, memory_order_relaxed);
}

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC void mlx_atomic_fetch_mul_explicit(
    device mlx_atomic<T>* object,
    T val,
    size_t offset) {
  T expected = mlx_atomic_load_explicit(object, offset);
  while (!mlx_atomic_compare_exchange_weak_explicit(
      object, &expected, val * expected, offset)) {
  }
}

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC bool mlx_atomic_compare_exchange_weak_explicit(
    device mlx_atomic<T>* object,
    thread T* expected,
    T val,
    size_t offset) {
  return atomic_compare_exchange_weak_explicit(
      &(object[offset].val),
      expected,
      val,
      memory_order_relaxed,
      memory_order_relaxed);
}

// Specialization for float since it does not atomic_fetch_min_explicit
template <>
METAL_FUNC void mlx_atomic_fetch_min_explicit<float>(
    device mlx_atomic<float>* object,
    float val,
    size_t offset) {
  float expected = mlx_atomic_load_explicit(object, offset);
  while (val < expected) {
    if (mlx_atomic_compare_exchange_weak_explicit(
            object, &expected, val, offset)) {
      return;
    }
  }
}

// Specialization for float since it does not atomic_fetch_max_explicit
template <>
METAL_FUNC void mlx_atomic_fetch_max_explicit<float>(
    device mlx_atomic<float>* object,
    float val,
    size_t offset) {
  float expected = mlx_atomic_load_explicit(object, offset);
  while (val > expected) {
    if (mlx_atomic_compare_exchange_weak_explicit(
            object, &expected, val, offset)) {
      return;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// Custom atomics
///////////////////////////////////////////////////////////////////////////////

namespace {

template <typename T>
constexpr constant uint packing_size = sizeof(uint) / sizeof(T);

template <typename T>
union uint_or_packed {
  T val[packing_size<T>];
  uint bits;
};

template <typename T, typename Op>
struct mlx_atomic_update_helper {
  uint operator()(uint_or_packed<T> init, T update, size_t elem_offset) {
    Op op;
    init.val[elem_offset] = op(update, init.val[elem_offset]);
    return init.bits;
  }
};

template <typename T, typename Op>
METAL_FUNC void mlx_atomic_update_and_store(
    device mlx_atomic<T>* object,
    T update,
    size_t offset) {
  size_t pack_offset = offset / packing_size<T>;
  size_t elem_offset = offset % packing_size<T>;

  mlx_atomic_update_helper<T, Op> helper;
  uint_or_packed<T> expected;
  expected.bits =
      atomic_load_explicit(&(object[pack_offset].val), memory_order_relaxed);

  while (Op::condition(update, expected.val[elem_offset]) &&
         !mlx_atomic_compare_exchange_weak_explicit(
             object,
             &(expected.bits),
             helper(expected, update, elem_offset),
             pack_offset)) {
  }
}

template <typename T>
struct __None {
  static bool condition(T a, T b) {
#pragma unused(a)
#pragma unused(b)
    return true;
  }

  T operator()(T a, T b) {
#pragma unused(b)
    return a;
  }
};

template <typename T>
struct __Add {
  static bool condition(T a, T b) {
#pragma unused(a)
#pragma unused(b)
    return true;
  }

  T operator()(T a, T b) {
    return a + b;
  }
};

template <typename T>
struct __Mul {
  static bool condition(T a, T b) {
#pragma unused(a)
    return b != 0;
  }

  T operator()(T a, T b) {
    return a * b;
  }
};

template <typename T>
struct __Max {
  static bool condition(T a, T b) {
    return a > b;
  }

  T operator()(T a, T b) {
    return max(a, b);
  }
};

template <typename T>
struct __Min {
  static bool condition(T a, T b) {
    return a < b;
  }

  T operator()(T a, T b) {
    return min(a, b);
  }
};

} // namespace

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC T
mlx_atomic_load_explicit(device mlx_atomic<T>* object, size_t offset) {
  size_t pack_offset = offset / sizeof(T);
  size_t elem_offset = offset % sizeof(T);
  uint_or_packed<T> packed_val;
  packed_val.bits =
      atomic_load_explicit(&(object[pack_offset].val), memory_order_relaxed);
  return packed_val.val[elem_offset];
}

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC void
mlx_atomic_store_explicit(device mlx_atomic<T>* object, T val, size_t offset) {
  mlx_atomic_update_and_store<T, __None<T>>(object, val, offset);
}

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC void mlx_atomic_fetch_and_explicit(
    device mlx_atomic<T>* object,
    T val,
    size_t offset) {
  size_t pack_offset = offset / packing_size<T>;
  size_t elem_offset = offset % packing_size<T>;
  uint_or_packed<T> identity;
  identity.bits = __UINT32_MAX__;
  identity.val[elem_offset] = val;

  atomic_fetch_and_explicit(
      &(object[pack_offset].val), identity.bits, memory_order_relaxed);
}

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC void mlx_atomic_fetch_or_explicit(
    device mlx_atomic<T>* object,
    T val,
    size_t offset) {
  size_t pack_offset = offset / packing_size<T>;
  size_t elem_offset = offset % packing_size<T>;
  uint_or_packed<T> identity;
  identity.bits = 0;
  identity.val[elem_offset] = val;

  atomic_fetch_or_explicit(
      &(object[pack_offset].val), identity.bits, memory_order_relaxed);
}

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC void mlx_atomic_fetch_min_explicit(
    device mlx_atomic<T>* object,
    T val,
    size_t offset) {
  mlx_atomic_update_and_store<T, __Min<T>>(object, val, offset);
}

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC void mlx_atomic_fetch_max_explicit(
    device mlx_atomic<T>* object,
    T val,
    size_t offset) {
  mlx_atomic_update_and_store<T, __Max<T>>(object, val, offset);
}

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC void mlx_atomic_fetch_add_explicit(
    device mlx_atomic<T>* object,
    T val,
    size_t offset) {
  mlx_atomic_update_and_store<T, __Add<T>>(object, val, offset);
}

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC void mlx_atomic_fetch_mul_explicit(
    device mlx_atomic<T>* object,
    T val,
    size_t offset) {
  mlx_atomic_update_and_store<T, __Mul<T>>(object, val, offset);
}

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC bool mlx_atomic_compare_exchange_weak_explicit(
    device mlx_atomic<T>* object,
    thread uint* expected,
    uint val,
    size_t offset) {
  return atomic_compare_exchange_weak_explicit(
      &(object[offset].val),
      expected,
      val,
      memory_order_relaxed,
      memory_order_relaxed);
}
// ===== end:   /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/atomic.h =====
#line 10 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/reduce.metal"
// ----- expanded "mlx/backend/metal/kernels/reduction/ops.h" from /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/reduce.metal:10 -----
// ===== begin: /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/reduction/ops.h =====
#line 1 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/reduction/ops.h"
// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <metal_atomic>
#include <metal_simdgroup>

#define DEFINE_SIMD_REDUCE()                                             \
  template <typename T, metal::enable_if_t<sizeof(T) < 8, bool> = true>  \
  T simd_reduce(T val) {                                                 \
    return simd_reduce_impl(val);                                        \
  }                                                                      \
                                                                         \
  template <typename T, metal::enable_if_t<sizeof(T) == 8, bool> = true> \
  T simd_reduce(T val) {                                                 \
    for (short i = simd_size / 2; i > 0; i /= 2) {                       \
      val = operator()(val, simd_shuffle_down(val, i));                  \
    }                                                                    \
    return val;                                                          \
  }

static constant constexpr const uint8_t simd_size = 32;

union bool4_or_uint {
  bool4 b;
  unsigned int i;
};

struct None {
  template <typename T>
  void atomic_update(device mlx_atomic<T>* out, T val, size_t offset = 0) {
    mlx_atomic_store_explicit(out, val, offset);
  }
};

template <typename U = bool>
struct And {
  DEFINE_SIMD_REDUCE()

  bool simd_reduce_impl(bool val) {
    return simd_all(val);
  }

  static constexpr constant bool init = true;

  void atomic_update(
      device mlx_atomic<unsigned int>* out,
      bool val,
      int elem_idx,
      size_t offset = 0) {
    if (!val) {
      bool4_or_uint update;
      update.b = {true, true, true, true};
      update.b[elem_idx] = false;
      mlx_atomic_fetch_and_explicit(out, update.i, offset);
    }
  }

  void
  atomic_update(device mlx_atomic<bool>* out, bool val, size_t offset = 0) {
    if (!val) {
      mlx_atomic_store_explicit(out, val, offset);
    }
  }

  // Non atomic update
  void update(device bool* out, bool val) {
    *out &= val;
  }

  // Operator
  bool operator()(bool a, bool b) {
    return a && b;
  }
};

template <typename U = bool>
struct Or {
  DEFINE_SIMD_REDUCE()

  bool simd_reduce_impl(bool val) {
    return simd_any(val);
  }

  static constexpr constant bool init = false;

  void atomic_update(
      device mlx_atomic<unsigned int>* out,
      bool val,
      int elem_idx,
      size_t offset = 0) {
    if (val) {
      bool4_or_uint update;
      update.b = {false, false, false, false};
      update.b[elem_idx] = true;
      mlx_atomic_fetch_or_explicit(out, update.i, offset);
    }
  }

  void
  atomic_update(device mlx_atomic<bool>* out, bool val, size_t offset = 0) {
    if (val) {
      mlx_atomic_store_explicit(out, val, offset);
    }
  }

  // Non atomic update
  void update(device bool* out, bool val) {
    *out |= val;
  }

  // Operator
  bool operator()(bool a, bool b) {
    return a || b;
  }
};

template <typename U>
struct Sum {
  DEFINE_SIMD_REDUCE()

  template <typename T>
  T simd_reduce_impl(T val) {
    return simd_sum(val);
  }

  static constexpr constant U init = U(0);

  template <typename T>
  void atomic_update(device mlx_atomic<T>* out, T val, size_t offset = 0) {
    mlx_atomic_fetch_add_explicit(out, val, offset);
  }

  // Operator
  U operator()(U a, U b) {
    return a + b;
  }
};

template <typename U>
struct Prod {
  DEFINE_SIMD_REDUCE()

  template <typename T>
  T simd_reduce_impl(T val) {
    return simd_product(val);
  }

  static constexpr constant U init = U(1);

  template <typename T>
  void atomic_update(device mlx_atomic<T>* out, T val, size_t offset = 0) {
    mlx_atomic_fetch_mul_explicit(out, val, offset);
  }

  // Operator
  U operator()(U a, U b) {
    return a * b;
  }
};

template <typename U>
struct Min {
  DEFINE_SIMD_REDUCE()

  template <typename T>
  metal::enable_if_t<metal::is_integral_v<T>, T> simd_reduce_impl(T val) {
    return simd_min(val);
  }

  template <typename T>
  metal::enable_if_t<!metal::is_integral_v<T>, T> simd_reduce_impl(T val) {
    if (simd_any(val != val)) {
      return static_cast<T>(NAN);
    }
    return simd_min(val);
  }

  static constexpr constant U init = Limits<U>::max;

  template <typename T>
  void atomic_update(device mlx_atomic<T>* out, T val, size_t offset = 0) {
    mlx_atomic_fetch_min_explicit(out, val, offset);
  }

  // Operator
  template <typename T>
  metal::enable_if_t<metal::is_integral_v<T>, T> operator()(T a, T b) {
    return a < b ? a : b;
  }

  template <typename T>
  metal::enable_if_t<!metal::is_integral_v<T>, T> operator()(T a, T b) {
    if (metal::isnan(a) || metal::isnan(b)) {
      return static_cast<T>(NAN);
    } else {
      return a < b ? a : b;
    }
  }

  template <>
  complex64_t operator()(complex64_t a, complex64_t b) {
    bool real_is_nan = metal::isnan(a.real) || metal::isnan(b.real);
    bool imag_is_nan = metal::isnan(a.imag) || metal::isnan(b.imag);

    if (!real_is_nan && !imag_is_nan) {
      return a < b ? a : b;
    } else if (real_is_nan && !imag_is_nan) {
      return complex64_t(
          static_cast<float>(NAN), a.imag < b.imag ? a.imag : b.imag);
    } else if (!real_is_nan && imag_is_nan) {
      return complex64_t(
          a.real < b.real ? a.real : b.real, static_cast<float>(NAN));
    } else {
      return complex64_t(static_cast<float>(NAN), static_cast<float>(NAN));
    }
  };
};
template <typename U>
struct Max {
  DEFINE_SIMD_REDUCE()

  template <typename T>
  metal::enable_if_t<metal::is_integral_v<T>, T> simd_reduce_impl(T val) {
    return simd_max(val);
  }

  template <typename T>
  metal::enable_if_t<!metal::is_integral_v<T>, T> simd_reduce_impl(T val) {
    if (simd_any(val != val)) {
      return static_cast<T>(NAN);
    }
    return simd_max(val);
  }

  static constexpr constant U init = Limits<U>::min;

  template <typename T>
  void atomic_update(device mlx_atomic<T>* out, T val, size_t offset = 0) {
    mlx_atomic_fetch_max_explicit(out, val, offset);
  }

  // Operator
  template <typename T>
  metal::enable_if_t<metal::is_integral_v<T>, T> operator()(T a, T b) {
    return a > b ? a : b;
  }

  template <typename T>
  metal::enable_if_t<!metal::is_integral_v<T>, T> operator()(T a, T b) {
    if (metal::isnan(a) || metal::isnan(b)) {
      return static_cast<T>(NAN);
    } else {
      return a > b ? a : b;
    }
  }

  template <>
  complex64_t operator()(complex64_t a, complex64_t b) {
    bool real_is_nan = metal::isnan(a.real) || metal::isnan(b.real);
    bool imag_is_nan = metal::isnan(a.imag) || metal::isnan(b.imag);

    if (!real_is_nan && !imag_is_nan) {
      return a > b ? a : b;
    } else if (real_is_nan && !imag_is_nan) {
      return complex64_t(
          static_cast<float>(NAN), a.imag > b.imag ? a.imag : b.imag);
    } else if (!real_is_nan && imag_is_nan) {
      return complex64_t(
          a.real > b.real ? a.real : b.real, static_cast<float>(NAN));
    } else {
      return complex64_t(static_cast<float>(NAN), static_cast<float>(NAN));
    }
  }
};
// ===== end:   /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/reduction/ops.h =====
#line 11 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/reduce.metal"
// ----- expanded "mlx/backend/metal/kernels/reduce.h" from /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/reduce.metal:11 -----
// ===== begin: /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/reduce.h =====
#line 1 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/reduce.h"
#pragma once
// ----- expanded "mlx/backend/metal/kernels/reduction/reduce_all.h" from /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/reduce.h:2 -----
// ===== begin: /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/reduction/reduce_all.h =====
#line 1 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/reduction/reduce_all.h"
// Copyright © 2023-2024 Apple Inc.

template <
    typename T,
    typename U,
    typename Op,
    typename IdxT = int64_t,
    int N_READS = REDUCE_N_READS>
[[kernel]] void all_reduce(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& in_size [[buffer(2)]],
    const constant size_t& row_size [[buffer(3)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  Op op;
  threadgroup U shared_vals[simd_size];

  U total = Op::init;
  IdxT start_idx = gid.y * IdxT(row_size);
  IdxT actual_row =
      (start_idx + row_size <= in_size) ? row_size : in_size - start_idx;
  IdxT blocks = actual_row / (lsize.x * N_READS);
  int extra = actual_row - blocks * (lsize.x * N_READS);
  extra -= lid.x * N_READS;
  start_idx += lid.x * N_READS;
  in += start_idx;

  if (extra >= N_READS) {
    blocks++;
    extra = 0;
  }

  for (IdxT b = 0; b < blocks; b++) {
    for (int i = 0; i < N_READS; i++) {
      total = op(static_cast<U>(in[i]), total);
    }
    in += lsize.x * N_READS;
  }
  if (extra > 0) {
    for (int i = 0; i < extra; i++) {
      total = op(static_cast<U>(in[i]), total);
    }
  }

  // Reduction within simd group
  total = op.simd_reduce(total);
  if (simd_per_group > 1) {
    if (simd_lane_id == 0) {
      shared_vals[simd_group_id] = total;
    }

    // Reduction within thread group
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = lid.x < simd_per_group ? shared_vals[lid.x] : op.init;
    total = op.simd_reduce(total);
  }

  if (lid.x == 0) {
    out[gid.y] = total;
  }
}
// ===== end:   /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/reduction/reduce_all.h =====
#line 3 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/reduce.h"
// ----- expanded "mlx/backend/metal/kernels/reduction/reduce_col.h" from /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/reduce.h:3 -----
// ===== begin: /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/reduction/reduce_col.h =====
#line 1 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/reduction/reduce_col.h"
// Copyright © 2023-2024 Apple Inc.

template <typename T, typename U, typename Op, typename IdxT, int NDIMS>
[[kernel]] void col_reduce_small(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant int64_t& reduction_stride [[buffer(3)]],
    const constant int* shape [[buffer(4)]],
    const constant int64_t* strides [[buffer(5)]],
    const constant int& ndim [[buffer(6)]],
    const constant int* reduce_shape [[buffer(7)]],
    const constant int64_t* reduce_strides [[buffer(8)]],
    const constant int& reduce_ndim [[buffer(9)]],
    const constant size_t& non_col_reductions [[buffer(10)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]]) {
  constexpr int n_reads = 4;
  Op op;
  LoopedElemToLoc<NDIMS, IdxT, (NDIMS > 2)> loop(reduce_ndim);
  const device T* row;

  U totals[n_reads];
  for (int i = 0; i < n_reads; i++) {
    totals[i] = Op::init;
  }

  IdxT column = IdxT(gid.x) * lsize.x * n_reads + lid.x * n_reads;
  if (column >= reduction_stride) {
    return;
  }
  bool safe = column + n_reads <= reduction_stride;

  IdxT out_idx = gid.y + gsize.y * IdxT(gid.z);
  IdxT in_idx = elem_to_loc<IdxT>(out_idx, shape, strides, ndim);
  in += in_idx + column;

  IdxT total_rows = IdxT(non_col_reductions) * IdxT(reduction_size);
  loop.next(lid.y, reduce_shape, reduce_strides);
  for (IdxT r = lid.y; r < total_rows; r += lsize.y) {
    row = in + loop.location();
    if (safe) {
      for (int i = 0; i < n_reads; i++) {
        totals[i] = op(static_cast<U>(row[i]), totals[i]);
      }
    } else {
      U vals[n_reads];
      for (int i = 0; i < n_reads; i++) {
        vals[i] =
            (column + i < reduction_stride) ? static_cast<U>(row[i]) : op.init;
      }
      for (int i = 0; i < n_reads; i++) {
        totals[i] = op(vals[i], totals[i]);
      }
    }
    loop.next(lsize.y, reduce_shape, reduce_strides);
  }

  if (lsize.y > 1) {
    // lsize.y should be <= 8
    threadgroup U shared_vals[32 * 8 * n_reads];
    for (int i = 0; i < n_reads; i++) {
      shared_vals[lid.y * lsize.x * n_reads + lid.x * n_reads + i] = totals[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid.y == 0) {
      for (int i = 0; i < n_reads; i++) {
        totals[i] = shared_vals[lid.x * n_reads + i];
      }
      for (uint j = 1; j < lsize.y; j++) {
        for (int i = 0; i < n_reads; i++) {
          totals[i] =
              op(shared_vals[j * lsize.x * n_reads + lid.x * n_reads + i],
                 totals[i]);
        }
      }
    }
  }

  if (lid.y == 0) {
    out += out_idx * IdxT(reduction_stride) + column;
    if (safe) {
      for (int i = 0; i < n_reads; i++) {
        out[i] = totals[i];
      }
    } else {
      for (int i = 0; column + i < reduction_stride; i++) {
        out[i] = totals[i];
      }
    }
  }
}

template <typename T, typename U, typename Op, typename IdxT, int NDIMS>
[[kernel]] void col_reduce_longcolumn(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& reduction_stride [[buffer(3)]],
    const constant int* shape [[buffer(4)]],
    const constant int64_t* strides [[buffer(5)]],
    const constant int& ndim [[buffer(6)]],
    const constant int* reduce_shape [[buffer(7)]],
    const constant int64_t* reduce_strides [[buffer(8)]],
    const constant int& reduce_ndim [[buffer(9)]],
    const constant size_t& non_col_reductions [[buffer(10)]],
    const constant size_t& out_size [[buffer(11)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]]) {
  Op op;
  LoopedElemToLoc<NDIMS, IdxT, (NDIMS > 2)> loop(reduce_ndim);
  const device T* row;

  IdxT out_idx = gid.x + gsize.x * IdxT(gid.y);
  IdxT in_idx = elem_to_loc<IdxT>(out_idx, shape, strides, ndim);
  in += in_idx + lid.x;

  U total = Op::init;
  IdxT total_rows = IdxT(non_col_reductions) * IdxT(reduction_size);
  loop.next(gid.z * lsize.y + lid.y, reduce_shape, reduce_strides);
  for (IdxT r = gid.z * lsize.y + lid.y; r < total_rows;
       r += lsize.y * gsize.z) {
    row = in + loop.location();
    total = op(static_cast<U>(*row), total);
    loop.next(lsize.y * gsize.z, reduce_shape, reduce_strides);
  }

  threadgroup U shared_vals[32 * 32];
  shared_vals[lid.y * lsize.x + lid.x] = total;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid.y == 0) {
    for (uint i = 1; i < lsize.y; i++) {
      total = op(total, shared_vals[i * lsize.x + lid.x]);
    }
    out[gid.z * IdxT(out_size) + out_idx * IdxT(reduction_stride) + lid.x] =
        total;
  }
}

/**
 * Our approach is the following simple looped approach:
 *  1. Each thread keeps running totals for BN / n_simdgroups outputs.
 *  2. Load a tile BM, BN in registers and accumulate in the running totals
 *  3. Move ahead by BM steps until the column axis and the non column
 *     reductions are exhausted.
 *  6. If BM == 32 then transpose in SM and simd reduce the running totals.
 *     Otherwise write in shared memory and BN threads accumulate the running
 *     totals with a loop.
 *  7. Write them to the output
 */
template <
    typename T,
    typename U,
    typename Op,
    typename IdxT,
    int NDIMS,
    int BM,
    int BN>
[[kernel]] void col_reduce_looped(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant int64_t& reduction_stride [[buffer(3)]],
    const constant int* shape [[buffer(4)]],
    const constant int64_t* strides [[buffer(5)]],
    const constant int& ndim [[buffer(6)]],
    const constant int* reduce_shape [[buffer(7)]],
    const constant int64_t* reduce_strides [[buffer(8)]],
    const constant int& reduce_ndim [[buffer(9)]],
    const constant size_t& non_col_reductions [[buffer(10)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  Op op;
  constexpr int n_simdgroups = 8;
  constexpr short tgp_size = n_simdgroups * simd_size;
  constexpr short n_reads = (BM * BN) / tgp_size;
  constexpr short n_read_blocks = BN / n_reads;

  threadgroup U shared_vals[BN * BM];
  U totals[n_reads];
  LoopedElemToLoc<NDIMS, IdxT, (NDIMS > 2)> loop(reduce_ndim);
  const device T* row;

  for (int i = 0; i < n_reads; i++) {
    totals[i] = Op::init;
  }

  short lid = simd_group_id * simd_size + simd_lane_id;
  short2 offset((lid % n_read_blocks) * n_reads, lid / n_read_blocks);
  IdxT column = BN * gid.x + offset.x;
  bool safe = column + n_reads <= reduction_stride;

  IdxT out_idx = gid.y + gsize.y * IdxT(gid.z);
  IdxT in_idx = elem_to_loc<IdxT>(out_idx, shape, strides, ndim);
  in += in_idx + column;

  IdxT total = IdxT(non_col_reductions) * IdxT(reduction_size);
  loop.next(offset.y, reduce_shape, reduce_strides);
  for (IdxT r = offset.y; r < total; r += BM) {
    row = in + loop.location();

    if (safe) {
      for (int i = 0; i < n_reads; i++) {
        totals[i] = op(static_cast<U>(row[i]), totals[i]);
      }
    } else {
      U vals[n_reads];
      for (int i = 0; i < n_reads; i++) {
        vals[i] =
            (column + i < reduction_stride) ? static_cast<U>(row[i]) : op.init;
      }
      for (int i = 0; i < n_reads; i++) {
        totals[i] = op(vals[i], totals[i]);
      }
    }

    loop.next(BM, reduce_shape, reduce_strides);
  }

  // We can use a simd reduction to accumulate across BM so each thread writes
  // the partial output to SM and then each simdgroup does BN / n_simdgroups
  // accumulations.
  if (BM == 32) {
    constexpr int n_outputs = BN / n_simdgroups;
    static_assert(
        BM != 32 || n_outputs == n_reads,
        "The tile should be selected such that n_outputs == n_reads");
    for (int i = 0; i < n_reads; i++) {
      shared_vals[offset.y * BN + offset.x + i] = totals[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    short2 out_offset(simd_group_id * n_outputs, simd_lane_id);
    for (int i = 0; i < n_outputs; i++) {
      totals[i] =
          op.simd_reduce(shared_vals[out_offset.y * BN + out_offset.x + i]);
    }

    // Write the output.
    if (simd_lane_id == 0) {
      IdxT out_column = BN * gid.x + out_offset.x;
      out += out_idx * IdxT(reduction_stride) + out_column;
      if (out_column + n_outputs <= reduction_stride) {
        for (int i = 0; i < n_outputs; i++) {
          out[i] = totals[i];
        }
      } else {
        for (int i = 0; out_column + i < reduction_stride; i++) {
          out[i] = totals[i];
        }
      }
    }
  }

  // Each thread holds n_reads partial results. We write them all out to shared
  // memory and threads with offset.y == 0 aggregate the columns and write the
  // outputs.
  else {
    short x_block = offset.x / n_reads;
    for (int i = 0; i < n_reads; i++) {
      shared_vals[x_block * BM * n_reads + i * BM + offset.y] = totals[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (offset.y == 0) {
      for (int i = 0; i < n_reads; i++) {
        for (int j = 1; j < BM; j++) {
          totals[i] =
              op(shared_vals[x_block * BM * n_reads + i * BM + j], totals[i]);
        }
      }
    }

    // Write the output.
    if (offset.y == 0) {
      out += out_idx * IdxT(reduction_stride) + column;
      if (safe) {
        for (int i = 0; i < n_reads; i++) {
          out[i] = totals[i];
        }
      } else {
        for (int i = 0; column + i < reduction_stride; i++) {
          out[i] = totals[i];
        }
      }
    }
  }
}

template <
    typename T,
    typename U,
    typename Op,
    typename IdxT,
    int NDIMS,
    int BM,
    int BN>
[[kernel]] void col_reduce_2pass(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant int64_t& reduction_stride [[buffer(3)]],
    const constant int* shape [[buffer(4)]],
    const constant int64_t* strides [[buffer(5)]],
    const constant int& ndim [[buffer(6)]],
    const constant int* reduce_shape [[buffer(7)]],
    const constant int64_t* reduce_strides [[buffer(8)]],
    const constant int& reduce_ndim [[buffer(9)]],
    const constant size_t& non_col_reductions [[buffer(10)]],
    const constant size_t& out_size [[buffer(11)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  Op op;
  constexpr int n_simdgroups = 8;
  constexpr short tgp_size = n_simdgroups * simd_size;
  constexpr short n_reads = (BM * BN) / tgp_size;
  constexpr short n_read_blocks = BN / n_reads;
  constexpr int n_outputs = BN / n_simdgroups;
  constexpr short outer_blocks = 32;
  static_assert(BM == 32, "BM should be equal to 32");

  threadgroup U shared_vals[BN * BM];
  U totals[n_reads];
  LoopedElemToLoc<NDIMS, IdxT, (NDIMS > 2)> loop(reduce_ndim);
  const device T* row;

  for (int i = 0; i < n_reads; i++) {
    totals[i] = Op::init;
  }

  short lid = simd_group_id * simd_size + simd_lane_id;
  short2 offset((lid % n_read_blocks) * n_reads, lid / n_read_blocks);
  IdxT column = BN * gid.x + offset.x;
  bool safe = column + n_reads <= reduction_stride;

  IdxT full_idx = gid.y + gsize.y * IdxT(gid.z);
  IdxT block_idx = full_idx / IdxT(out_size);
  IdxT out_idx = full_idx % IdxT(out_size);
  IdxT in_idx = elem_to_loc<IdxT>(out_idx, shape, strides, ndim);
  in += in_idx + column;

  IdxT total = IdxT(non_col_reductions) * IdxT(reduction_size);
  loop.next(offset.y + block_idx * BM, reduce_shape, reduce_strides);
  for (IdxT r = offset.y + block_idx * BM; r < total; r += outer_blocks * BM) {
    row = in + loop.location();

    if (safe) {
      for (int i = 0; i < n_reads; i++) {
        totals[i] = op(static_cast<U>(row[i]), totals[i]);
      }
    } else {
      U vals[n_reads];
      for (int i = 0; i < n_reads; i++) {
        vals[i] =
            (column + i < reduction_stride) ? static_cast<U>(row[i]) : op.init;
      }
      for (int i = 0; i < n_reads; i++) {
        totals[i] = op(vals[i], totals[i]);
      }
    }

    loop.next(outer_blocks * BM, reduce_shape, reduce_strides);
  }

  // We can use a simd reduction to accumulate across BM so each thread writes
  // the partial output to SM and then each simdgroup does BN / n_simdgroups
  // accumulations.
  for (int i = 0; i < n_reads; i++) {
    shared_vals[offset.y * BN + offset.x + i] = totals[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  short2 out_offset(simd_group_id * n_outputs, simd_lane_id);
  for (int i = 0; i < n_outputs; i++) {
    totals[i] =
        op.simd_reduce(shared_vals[out_offset.y * BN + out_offset.x + i]);
  }

  // Write the output.
  if (simd_lane_id == 0) {
    IdxT out_column = BN * gid.x + out_offset.x;
    out += full_idx * IdxT(reduction_stride) + out_column;
    if (out_column + n_outputs <= reduction_stride) {
      for (int i = 0; i < n_outputs; i++) {
        out[i] = totals[i];
      }
    } else {
      for (int i = 0; out_column + i < reduction_stride; i++) {
        out[i] = totals[i];
      }
    }
  }
}
// ===== end:   /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/reduction/reduce_col.h =====
#line 4 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/reduce.h"
// ----- expanded "mlx/backend/metal/kernels/reduction/reduce_init.h" from /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/reduce.h:4 -----
// ===== begin: /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/reduction/reduce_init.h =====
#line 1 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/reduction/reduce_init.h"
// Copyright © 2023-2024 Apple Inc.

template <typename T, typename Op>
[[kernel]] void init_reduce(
    device T* out [[buffer(0)]],
    uint tid [[thread_position_in_grid]]) {
  out[tid] = Op::init;
}
// ===== end:   /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/reduction/reduce_init.h =====
#line 5 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/reduce.h"
// ----- expanded "mlx/backend/metal/kernels/reduction/reduce_row.h" from /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/reduce.h:5 -----
// ===== begin: /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/reduction/reduce_row.h =====
#line 1 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/reduction/reduce_row.h"
// Copyright © 2023-2024 Apple Inc.

// Row reduction utilities
// - `per_thread_row_reduce` collaborative partial reduction in the threadgroup
// - `threadgroup_reduce` collaborative reduction in the threadgroup such that
//   lid.x == 0 holds the reduced value
// - `thread_reduce` simple loop and reduce the row

/**
 * The thread group collaboratively reduces across the rows with bounds
 * checking. In the end each thread holds a part of the reduction.
 */
template <
    typename T,
    typename U,
    typename Op,
    int N_READS = REDUCE_N_READS,
    int N_WRITES = REDUCE_N_WRITES>
METAL_FUNC void per_thread_row_reduce(
    thread U totals[N_WRITES],
    const device T* inputs[N_WRITES],
    int blocks,
    int extra,
    uint lsize_x,
    uint lid_x) {
  Op op;

  // Set up the accumulator registers
  for (int i = 0; i < N_WRITES; i++) {
    totals[i] = Op::init;
  }

  // Loop over the reduction size within thread group
  for (int i = 0; i < blocks; i++) {
    for (int j = 0; j < N_WRITES; j++) {
      for (int i = 0; i < N_READS; i++) {
        totals[j] = op(static_cast<U>(inputs[j][i]), totals[j]);
      }

      inputs[j] += lsize_x * N_READS;
    }
  }

  // Separate case for the last set as we close the reduction size
  int index = lid_x * N_READS;
  if (index + N_READS <= extra) {
    for (int j = 0; j < N_WRITES; j++) {
      for (int i = 0; i < N_READS; i++) {
        totals[j] = op(static_cast<U>(inputs[j][i]), totals[j]);
      }
    }
  } else {
    for (int j = 0; j < N_WRITES; j++) {
      for (int i = 0; index + i < extra; i++) {
        totals[j] = op(static_cast<U>(inputs[j][i]), totals[j]);
      }
    }
  }
}

/**
 * Consecutive rows in a contiguous array.
 */
template <
    typename T,
    typename U,
    typename Op,
    int N_READS = REDUCE_N_READS,
    int N_WRITES = REDUCE_N_WRITES>
METAL_FUNC void per_thread_row_reduce(
    thread U totals[N_WRITES],
    const device T* in,
    const constant size_t& reduction_size,
    int blocks,
    int extra,
    uint lsize_x,
    uint lid_x) {
  // Set up the input pointers
  const device T* inputs[N_WRITES];
  inputs[0] = in + lid_x * N_READS;
  for (int i = 1; i < N_READS; i++) {
    inputs[i] = inputs[i - 1] + reduction_size;
  }

  per_thread_row_reduce<T, U, Op, N_READS, N_WRITES>(
      totals, inputs, blocks, extra, lsize_x, lid_x);
}

/**
 * Consecutive rows in an arbitrarily ordered array.
 */
template <
    typename T,
    typename U,
    typename Op,
    int N_READS = REDUCE_N_READS,
    int N_WRITES = REDUCE_N_WRITES>
METAL_FUNC void per_thread_row_reduce(
    thread U totals[N_WRITES],
    const device T* in,
    const int64_t row_idx,
    int blocks,
    int extra,
    const constant int* shape,
    const constant int64_t* strides,
    const constant int& ndim,
    uint lsize_x,
    uint lid_x) {
  // Set up the input pointers
  const device T* inputs[N_WRITES];
  in += lid_x * N_READS;
  for (int i = 0; i < N_READS; i++) {
    inputs[i] = in + elem_to_loc(row_idx + i, shape, strides, ndim);
  }

  per_thread_row_reduce<T, U, Op, N_READS, N_WRITES>(
      totals, inputs, blocks, extra, lsize_x, lid_x);
}

/**
 * Reduce within the threadgroup.
 */
template <
    typename T,
    typename U,
    typename Op,
    int N_READS = REDUCE_N_READS,
    int N_WRITES = REDUCE_N_WRITES>
METAL_FUNC void threadgroup_reduce(
    thread U totals[N_WRITES],
    threadgroup U* shared_vals,
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  Op op;

  // Simdgroup first
  for (int i = 0; i < N_WRITES; i++) {
    totals[i] = op.simd_reduce(totals[i]);
  }

  // Across simdgroups
  if (simd_per_group > 1) {
    if (simd_lane_id == 0) {
      for (int i = 0; i < N_WRITES; i++) {
        shared_vals[simd_group_id * N_WRITES + i] = totals[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    U values[N_WRITES];
    for (int i = 0; i < N_WRITES; i++) {
      values[i] = (lid.x < simd_per_group) ? shared_vals[lid.x * N_WRITES + i]
                                           : op.init;
    }

    for (int i = 0; i < N_WRITES; i++) {
      totals[i] = op.simd_reduce(values[i]);
    }
  }
}

template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
METAL_FUNC void
thread_reduce(thread U& total, const device T* row, int blocks, int extra) {
  Op op;
  for (int i = 0; i < blocks; i++) {
    U vals[N_READS];
    for (int j = 0; j < N_READS; j++) {
      vals[j] = row[j];
    }
    for (int j = 0; j < N_READS; j++) {
      total = op(vals[j], total);
    }
    row += N_READS;
  }
  for (int i = 0; i < extra; i++) {
    total = op(*row++, total);
  }
}

// Reduction kernels
// - `row_reduce_small` depending on the non-row reductions and row size it
//   either just loops over everything or a simd collaboratively reduces the
//   non_row reductions. In the first case one thread is responsible for one
//   output on the 2nd one simd is responsible for one output.
// - `row_reduce_simple` simple contiguous row reduction
// - `row_reduce_looped` simply loop and reduce each row for each non-row
//   reduction. One threadgroup is responsible for one output.

template <
    typename T,
    typename U,
    typename Op,
    typename IdxT,
    int NDIMS,
    int N_READS = REDUCE_N_READS>
[[kernel]] void row_reduce_small(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant int64_t& row_size [[buffer(2)]],
    const constant int64_t& non_row_reductions [[buffer(3)]],
    const constant int* shape [[buffer(4)]],
    const constant int64_t* strides [[buffer(5)]],
    const constant int& ndim [[buffer(6)]],
    const constant int* reduce_shape [[buffer(7)]],
    const constant int64_t* reduce_strides [[buffer(8)]],
    const constant int& reduce_ndim [[buffer(9)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 tid [[thread_position_in_grid]],
    uint3 tsize [[threads_per_grid]]) {
  Op op;

  U total_val = Op::init;
  LoopedElemToLoc<NDIMS, IdxT, (NDIMS > 2)> loop(reduce_ndim);

  // Precompute some row reduction numbers
  const device T* row;
  int blocks = IdxT(row_size) / N_READS;
  int extra = IdxT(row_size) % N_READS;

  if ((non_row_reductions < 32 && row_size <= 8) || non_row_reductions <= 8) {
    // Simple loop over non_row_reductions and reduce the row in the thread.
    IdxT out_idx = tid.x + tsize.x * IdxT(tid.y);
    in += elem_to_loc<IdxT>(out_idx, shape, strides, ndim);

    for (uint r = 0; r < non_row_reductions; r++) {
      row = in + loop.location();
      thread_reduce<T, U, Op, N_READS>(total_val, row, blocks, extra);
      loop.next(reduce_shape, reduce_strides);
    }

    out[out_idx] = total_val;
  } else {
    // Collaboratively reduce over non_row_reductions in the simdgroup. Each
    // thread reduces every 32nd row and then a simple simd reduce.
    IdxT out_idx = gid.y + gsize.y * IdxT(gid.z);
    in += elem_to_loc<IdxT>(out_idx, shape, strides, ndim);

    loop.next(simd_lane_id, reduce_shape, reduce_strides);

    for (uint r = simd_lane_id; r < non_row_reductions; r += simd_size) {
      row = in + loop.location();
      thread_reduce<T, U, Op, N_READS>(total_val, row, blocks, extra);
      loop.next(simd_size, reduce_shape, reduce_strides);
    }

    total_val = op.simd_reduce(total_val);

    if (simd_lane_id == 0) {
      out[out_idx] = total_val;
    }
  }
}

template <
    typename T,
    typename U,
    typename Op,
    typename IdxT = int64_t,
    int N_READS = REDUCE_N_READS,
    int N_WRITES = REDUCE_N_WRITES>
[[kernel]] void row_reduce_simple(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant int64_t& out_size [[buffer(3)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup U shared_vals[simd_size * N_WRITES];
  U totals[N_WRITES];

  // Move to the row
  IdxT out_idx = N_WRITES * (gid.y + gsize.y * IdxT(gid.z));
  if (out_idx + N_WRITES > out_size) {
    out_idx = out_size - N_WRITES;
  }
  in += out_idx * IdxT(reduction_size);
  out += out_idx;

  // Each thread reduces across the row
  int blocks = IdxT(reduction_size) / (lsize.x * N_READS);
  int extra = reduction_size - blocks * (lsize.x * N_READS);
  per_thread_row_reduce<T, U, Op, N_READS, N_WRITES>(
      totals, in, reduction_size, blocks, extra, lsize.x, lid.x);

  // Reduce across the threadgroup
  threadgroup_reduce<T, U, Op, N_READS, N_WRITES>(
      totals, shared_vals, lid, simd_lane_id, simd_per_group, simd_group_id);

  // Write the output
  if (lid.x == 0) {
    for (int i = 0; i < N_WRITES; i++) {
      out[i] = totals[i];
    }
  }
}

template <
    typename T,
    typename U,
    typename Op,
    typename IdxT,
    int NDIMS,
    int N_READS = REDUCE_N_READS>
[[kernel]] void row_reduce_looped(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant int64_t& row_size [[buffer(2)]],
    const constant int64_t& non_row_reductions [[buffer(3)]],
    const constant int* shape [[buffer(4)]],
    const constant int64_t* strides [[buffer(5)]],
    const constant int& ndim [[buffer(6)]],
    const constant int* reduce_shape [[buffer(7)]],
    const constant int64_t* reduce_strides [[buffer(8)]],
    const constant int& reduce_ndim [[buffer(9)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  Op op;
  threadgroup U shared_vals[simd_size];
  U total = Op::init;

  IdxT out_idx = gid.y + gsize.y * IdxT(gid.z);

  // lid.x * N_READS breaks the per_thread_row_reduce interface a bit. Maybe it
  // needs a small refactor.
  in += elem_to_loc<IdxT>(out_idx, shape, strides, ndim) + lid.x * N_READS;

  LoopedElemToLoc<NDIMS, IdxT, (NDIMS > 2)> loop(reduce_ndim);
  const device T* row;
  int blocks = IdxT(row_size) / (lsize.x * N_READS);
  int extra = row_size - blocks * (lsize.x * N_READS);

  for (IdxT i = 0; i < non_row_reductions; i++) {
    row = in + loop.location();

    // Each thread reduces across the row
    U row_total;
    per_thread_row_reduce<T, U, Op, N_READS, 1>(
        &row_total, &row, blocks, extra, lsize.x, lid.x);

    // Aggregate across rows
    total = op(total, row_total);

    loop.next(reduce_shape, reduce_strides);
  }

  // Reduce across the threadgroup
  threadgroup_reduce<T, U, Op, N_READS, 1>(
      &total, shared_vals, lid, simd_lane_id, simd_per_group, simd_group_id);

  // Write the output
  if (lid.x == 0) {
    out[out_idx] = total;
  }
}
// ===== end:   /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/reduction/reduce_row.h =====
#line 6 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/reduce.h"
// ===== end:   /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/reduce.h =====
#line 12 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/reduce.metal"

#define instantiate_init_reduce(name, tname, type, op) \
  instantiate_kernel("init_reduce_" #name #tname, init_reduce, type, op<type>)

instantiate_init_reduce(and, bool_, bool, And)
instantiate_init_reduce(or, bool_, bool, Or)

#define instantiate_init_sum_prod(name, op)                 \
  instantiate_init_reduce(name, int32, int32_t, op)         \
  instantiate_init_reduce(name, int64, int64_t, op)         \
  instantiate_init_reduce(name, float16, float16_t, op)     \
  instantiate_init_reduce(name, bfloat16, bfloat16_t, op)   \
  instantiate_init_reduce(name, float32, float, op)         \
  instantiate_init_reduce(name, complex64, complex64_t, op)

instantiate_init_sum_prod(sum, Sum)
instantiate_init_sum_prod(prod, Prod)

#define instantiate_init_min_max(name, op)                   \
  instantiate_init_reduce(name, bool_, bool, op)             \
  instantiate_init_reduce(name, int8, int8_t, op)            \
  instantiate_init_reduce(name, int16, int16_t, op)          \
  instantiate_init_reduce(name, int32, int32_t, op)          \
  instantiate_init_reduce(name, int64, int64_t, op)          \
  instantiate_init_reduce(name, uint8, uint8_t, op)          \
  instantiate_init_reduce(name, uint16, uint16_t, op)        \
  instantiate_init_reduce(name, uint32, uint32_t, op)        \
  instantiate_init_reduce(name, uint64, uint64_t, op)        \
  instantiate_init_reduce(name, float16, float16_t, op)      \
  instantiate_init_reduce(name, bfloat16, bfloat16_t, op)    \
  instantiate_init_reduce(name, float32, float, op)          \
  instantiate_init_reduce(name, complex64, complex64_t, op)

instantiate_init_min_max(min, Min)
instantiate_init_min_max(max, Max)

#define instantiate_all_reduce(name, itype, otype, op) \
  instantiate_kernel("all_reduce_" #name,              \
                     all_reduce,                       \
                     itype, otype, op)

#define instantiate_col_reduce_small(name, itype, otype, op, dim)          \
  instantiate_kernel("col_reduce_small_" #dim "_reduce_" #name,            \
                     col_reduce_small,                                     \
                     itype, otype, op, int, dim)                           \
  instantiate_kernel("col_reduce_longcolumn_" #dim "_reduce_" #name,       \
                     col_reduce_longcolumn,                                \
                     itype, otype, op, int, dim)                           \
  instantiate_kernel("col_reduce_small_large_" #dim "_reduce_" #name,      \
                     col_reduce_small,                                     \
                     itype, otype, op, int64_t, dim)                       \
  instantiate_kernel("col_reduce_longcolumn_large_" #dim "_reduce_" #name, \
                     col_reduce_longcolumn,                                \
                     itype, otype, op, int64_t, dim)

#define instantiate_col_reduce_looped_tile(name, itype, otype, op, dim, bm, bn)        \
  instantiate_kernel("col_reduce_looped_" #dim "_" #bm "_" #bn "_reduce_" #name,       \
                     col_reduce_looped,                                                \
                     itype, otype, op, int, dim, bm, bn)                               \
  instantiate_kernel("col_reduce_looped_large_" #dim "_" #bm "_" #bn "_reduce_" #name, \
                     col_reduce_looped,                                                \
                     itype, otype, op, int64_t, dim, bm, bn)

#define instantiate_col_reduce_2pass_tile(name, itype, otype, op, dim, bm, bn)        \
  instantiate_kernel("col_reduce_2pass_" #dim "_" #bm "_" #bn "_reduce_" #name,       \
                     col_reduce_2pass,                                                \
                     itype, otype, op, int, dim, bm, bn)                              \
  instantiate_kernel("col_reduce_2pass_large_" #dim "_" #bm "_" #bn "_reduce_" #name, \
                     col_reduce_2pass,                                                \
                     itype, otype, op, int64_t, dim, bm, bn)

#define instantiate_col_reduce_looped(name, itype, otype, op, dim)        \
  instantiate_col_reduce_looped_tile(name, itype, otype, op, dim, 32, 32) \
  instantiate_col_reduce_2pass_tile(name, itype, otype, op, dim, 32, 32)

#define instantiate_col_reduce_general(name, itype, otype, op) \
  instantiate_col_reduce_small(name, itype, otype, op, 1)      \
  instantiate_col_reduce_small(name, itype, otype, op, 2)      \
  instantiate_col_reduce_small(name, itype, otype, op, 5)      \
  instantiate_col_reduce_looped(name, itype, otype, op, 1)     \
  instantiate_col_reduce_looped(name, itype, otype, op, 2)     \
  instantiate_col_reduce_looped(name, itype, otype, op, 5)

#define instantiate_row_reduce_small(name, itype, otype, op, dim)     \
  instantiate_kernel("row_reduce_small_" #dim "_reduce_" #name,       \
                     row_reduce_small,                                \
                     itype, otype, op, int, dim)                      \
  instantiate_kernel("row_reduce_small_large_" #dim "_reduce_" #name, \
                     row_reduce_small,                                \
                     itype, otype, op, int64_t, dim)

#define instantiate_row_reduce_looped(name, itype, otype, op, dim)       \
  instantiate_kernel("row_reduce_looped_" #dim "_reduce_" #name,         \
                     row_reduce_looped,                                  \
                     itype, otype, op, int, dim)                         \
  instantiate_kernel("row_reduce_looped_large_" #dim "_reduce_" #name,   \
                     row_reduce_looped,                                  \
                     itype, otype, op, int64_t, dim)

#define instantiate_row_reduce_general(name, itype, otype, op) \
  instantiate_row_reduce_small(name, itype, otype, op, 1)      \
  instantiate_row_reduce_small(name, itype, otype, op, 2)      \
  instantiate_row_reduce_small(name, itype, otype, op, 5)      \
  instantiate_row_reduce_looped(name, itype, otype, op, 1)     \
  instantiate_row_reduce_looped(name, itype, otype, op, 2)     \
  instantiate_row_reduce_looped(name, itype, otype, op, 5)     \
  instantiate_kernel("row_reduce_simple_" #name,               \
                     row_reduce_simple,                        \
                     itype, otype, op)

#define instantiate_reduce_functions(name, tname, itype, otype, op)    \
  instantiate_all_reduce(name##tname, itype, otype, op<otype>)         \
  instantiate_row_reduce_general(name##tname, itype, otype, op<otype>) \
  instantiate_col_reduce_general(name##tname, itype, otype, op<otype>)

#define instantiate_and_or(name, op)                           \
  instantiate_reduce_functions(name, bool_, bool, bool, op)    \
  instantiate_reduce_functions(name, int16, int16_t, bool, op) \
  instantiate_reduce_functions(name, int32, int32_t, bool, op) \
  instantiate_reduce_functions(name, int64, int64_t, bool, op)

instantiate_and_or(and, And)
instantiate_and_or(or, Or)

#define instantiate_sum_prod(name, op)                                       \
  instantiate_reduce_functions(name, uint8, uint8_t, int32_t, op)            \
  instantiate_reduce_functions(name, uint16, uint16_t, uint32_t, op)         \
  instantiate_reduce_functions(name, uint32, uint32_t, uint32_t, op)         \
  instantiate_reduce_functions(name, uint64, uint64_t, uint64_t, op)         \
  instantiate_reduce_functions(name, int8, int8_t, int32_t, op)              \
  instantiate_reduce_functions(name, int16, int16_t, int32_t, op)            \
  instantiate_reduce_functions(name, int32, int32_t, int32_t, op)            \
  instantiate_reduce_functions(name, int64, int64_t, int64_t, op)            \
  instantiate_reduce_functions(name, float16, float16_t, float16_t, op)      \
  instantiate_reduce_functions(name, bfloat16, bfloat16_t, bfloat16_t, op)   \
  instantiate_reduce_functions(name, float32, float, float, op)              \
  instantiate_reduce_functions(name, complex64, complex64_t, complex64_t, op)

instantiate_sum_prod(sum, Sum)
instantiate_sum_prod(prod, Prod)

#define instantiate_min_max(name, op)                                        \
  instantiate_reduce_functions(name, int8, int8_t, int8_t, op)               \
  instantiate_reduce_functions(name, int16, int16_t, int16_t, op)            \
  instantiate_reduce_functions(name, int32, int32_t, int32_t, op)            \
  instantiate_reduce_functions(name, int64, int64_t, int64_t, op)            \
  instantiate_reduce_functions(name, uint8, uint8_t, uint8_t, op)            \
  instantiate_reduce_functions(name, uint16, uint16_t, uint16_t, op)         \
  instantiate_reduce_functions(name, uint32, uint32_t, uint32_t, op)         \
  instantiate_reduce_functions(name, uint64, uint64_t, uint64_t, op)         \
  instantiate_reduce_functions(name, float16, float16_t, float16_t, op)      \
  instantiate_reduce_functions(name, bfloat16, bfloat16_t, bfloat16_t, op)   \
  instantiate_reduce_functions(name, float32, float, float, op)              \
  instantiate_reduce_functions(name, complex64, complex64_t, complex64_t, op)

instantiate_min_max(min, Min)
instantiate_min_max(max, Max)
    // clang-format on
// ===== end:   /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/reduce.metal =====
