// ===== begin: /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/scan.metal =====
#line 1 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/scan.metal"
// Copyright © 2023-2024 Apple Inc.

#include <metal_math>
#include <metal_simdgroup>

// clang-format off

using namespace metal;

// ----- expanded "mlx/backend/metal/kernels/defines.h" from /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/scan.metal:10 -----
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
#line 11 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/scan.metal"
// ----- expanded "mlx/backend/metal/kernels/utils.h" from /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/scan.metal:11 -----
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
#line 12 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/scan.metal"
// ----- expanded "mlx/backend/metal/kernels/scan.h" from /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/scan.metal:12 -----
// ===== begin: /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/scan.h =====
#line 1 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/scan.h"
// Copyright © 2023-2024 Apple Inc.

#pragma once

// ----- expanded "mlx/backend/metal/kernels/binary_ops.h" from /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/scan.h:5 -----
// ===== begin: /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/binary_ops.h =====
#line 1 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/binary_ops.h"
// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <metal_integer>
#include <metal_math>

constant mlx::os_log logger("mlx", "binary_ops");

struct Add {
  template <typename T>
  T operator()(T x, T y) {
    return x + y;
  }
};

struct FloorDivide {
  template <typename T>
  T operator()(T x, T y) {
    return x / y;
  }
  template <>
  float operator()(float x, float y) {
    return trunc(x / y);
  }
  template <>
  half operator()(half x, half y) {
    return trunc(x / y);
  }
  template <>
  bfloat16_t operator()(bfloat16_t x, bfloat16_t y) {
    return trunc(x / y);
  }
};

struct Divide {
  template <typename T>
  T operator()(T x, T y) {
    return x / y;
  }
};

struct Remainder {
  template <typename T>
  metal::enable_if_t<metal::is_integral_v<T> & !metal::is_signed_v<T>, T>
  operator()(T x, T y) {
    return x % y;
  }
  template <typename T>
  metal::enable_if_t<metal::is_integral_v<T> & metal::is_signed_v<T>, T>
  operator()(T x, T y) {
    auto r = x % y;
    if (r != 0 && (r < 0 != y < 0)) {
      r += y;
    }
    return r;
  }
  template <typename T>
  metal::enable_if_t<!metal::is_integral_v<T>, T> operator()(T x, T y) {
    T r = fmod(x, y);
    if (r != 0 && (r < 0 != y < 0)) {
      r += y;
    }
    return r;
  }
  template <>
  complex64_t operator()(complex64_t x, complex64_t y) {
    return x % y;
  }
};

struct Equal {
  template <typename T>
  bool operator()(T x, T y) {
    return x == y;
  }
};

struct NaNEqual {
  template <typename T>
  bool operator()(T x, T y) {
    return x == y || (metal::isnan(x) && metal::isnan(y));
  }
  template <>
  bool operator()(complex64_t x, complex64_t y) {
    return x == y ||
        (metal::isnan(x.real) && metal::isnan(y.real) && metal::isnan(x.imag) &&
         metal::isnan(y.imag)) ||
        (x.real == y.real && metal::isnan(x.imag) && metal::isnan(y.imag)) ||
        (metal::isnan(x.real) && metal::isnan(y.real) && x.imag == y.imag);
  }
};

struct Greater {
  template <typename T>
  bool operator()(T x, T y) {
    return x > y;
  }
};

struct GreaterEqual {
  template <typename T>
  bool operator()(T x, T y) {
    return x >= y;
  }
};

struct Less {
  template <typename T>
  bool operator()(T x, T y) {
    return x < y;
  }
};

struct LessEqual {
  template <typename T>
  bool operator()(T x, T y) {
    return x <= y;
  }
};

struct LogAddExp {
  template <typename T>
  T operator()(T x, T y) {
    if (metal::isnan(x) || metal::isnan(y)) {
      return metal::numeric_limits<T>::quiet_NaN();
    }
    constexpr T inf = metal::numeric_limits<T>::infinity();
    T maxval = metal::max(x, y);
    T minval = metal::min(x, y);
    return (minval == -inf || maxval == inf)
        ? maxval
        : (maxval + log1p(metal::exp(minval - maxval)));
  };

  complex64_t operator()(complex64_t x, complex64_t y) {
    if (metal::isnan(x.real) || metal::isnan(x.imag) || metal::isnan(y.real) ||
        metal::isnan(y.imag)) {
      return metal::numeric_limits<float>::quiet_NaN();
    }
    constexpr float inf = metal::numeric_limits<float>::infinity();
    complex64_t maxval = x > y ? x : y;
    complex64_t minval = x < y ? x : y;
    if (minval.real == -inf || maxval.real == inf)
      return maxval;
    float m = metal::exp(minval.real - maxval.real);
    complex64_t dexp{
        m * metal::cos(minval.imag - maxval.imag),
        m * metal::sin(minval.imag - maxval.imag),
    };
    return maxval + log1p(dexp);
  }
};

struct Maximum {
  template <typename T>
  metal::enable_if_t<metal::is_integral_v<T>, T> operator()(T x, T y) {
    return metal::max(x, y);
  }

  template <typename T>
  metal::enable_if_t<!metal::is_integral_v<T>, T> operator()(T x, T y) {
    if (metal::isnan(x)) {
      return x;
    }
    return x > y ? x : y;
  }

  template <>
  complex64_t operator()(complex64_t x, complex64_t y) {
    if (metal::isnan(x.real) || metal::isnan(x.imag)) {
      return x;
    }
    return x > y ? x : y;
  }
};

struct Minimum {
  template <typename T>
  metal::enable_if_t<metal::is_integral_v<T>, T> operator()(T x, T y) {
    return metal::min(x, y);
  }

  template <typename T>
  metal::enable_if_t<!metal::is_integral_v<T>, T> operator()(T x, T y) {
    if (metal::isnan(x)) {
      return x;
    }
    return x < y ? x : y;
  }

  template <>
  complex64_t operator()(complex64_t x, complex64_t y) {
    if (metal::isnan(x.real) || metal::isnan(x.imag)) {
      return x;
    }
    return x < y ? x : y;
  }
};

struct Multiply {
  template <typename T>
  T operator()(T x, T y) {
    return x * y;
  }
};

struct NotEqual {
  template <typename T>
  bool operator()(T x, T y) {
    return x != y;
  }
  template <>
  bool operator()(complex64_t x, complex64_t y) {
    return x.real != y.real || x.imag != y.imag;
  }
};

struct Power {
  template <typename T>
  metal::enable_if_t<!metal::is_integral_v<T>, T> operator()(T base, T exp) {
    return metal::pow(base, exp);
  }

  template <typename T>
  metal::enable_if_t<metal::is_integral_v<T>, T> operator()(T base, T exp) {
    T res = 1;
    // Undefined to raise integer to negative power
    if (exp < 0) {
      logger.log_debug(
          "int pow exp<0 (base=%ld exp=%ld)", (long)base, (long)exp);
      return 0;
    }

    while (exp) {
      if (exp & 1) {
        res *= base;
      }
      exp >>= 1;
      base *= base;
    }
    return res;
  }

  template <>
  complex64_t operator()(complex64_t x, complex64_t y) {
    if (x.real == 0 && x.imag == 0) {
      if (metal::isnan(y.real) || metal::isnan(y.imag)) {
        auto nan = metal::numeric_limits<float>::quiet_NaN();
        return {nan, nan};
      }
      return {0.0, 0.0};
    }
    auto x_theta = metal::atan2(x.imag, x.real);
    auto x_ln_r = 0.5 * metal::log(x.real * x.real + x.imag * x.imag);
    auto mag = metal::exp(y.real * x_ln_r - y.imag * x_theta);
    auto phase = y.imag * x_ln_r + y.real * x_theta;
    return {mag * metal::cos(phase), mag * metal::sin(phase)};
  }
};

struct Subtract {
  template <typename T>
  T operator()(T x, T y) {
    return x - y;
  }
};

struct LogicalAnd {
  template <typename T>
  T operator()(T x, T y) {
    return x && y;
  };
};

struct LogicalOr {
  template <typename T>
  T operator()(T x, T y) {
    return x || y;
  };
};

struct BitwiseAnd {
  template <typename T>
  T operator()(T x, T y) {
    return x & y;
  };
};

struct BitwiseOr {
  template <typename T>
  T operator()(T x, T y) {
    return x | y;
  };
};

struct BitwiseXor {
  template <typename T>
  T operator()(T x, T y) {
    return x ^ y;
  };
};

struct LeftShift {
  template <typename T>
  T operator()(T x, T y) {
    return x << y;
  };
};

struct RightShift {
  template <typename T>
  T operator()(T x, T y) {
    return x >> y;
  };
};

struct ArcTan2 {
  template <typename T>
  T operator()(T y, T x) {
    return metal::precise::atan2(y, x);
  }
};

struct DivMod {
  template <typename T>
  metal::array<T, 2> operator()(T x, T y) {
    return {FloorDivide{}(x, y), Remainder{}(x, y)};
  };
};
// ===== end:   /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/binary_ops.h =====
#line 6 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/scan.h"

#define DEFINE_SIMD_SCAN()                                               \
  template <typename T, metal::enable_if_t<sizeof(T) < 8, bool> = true>  \
  T simd_scan(T val) {                                                   \
    return simd_scan_impl(val);                                          \
  }                                                                      \
                                                                         \
  template <typename T, metal::enable_if_t<sizeof(T) == 8, bool> = true> \
  T simd_scan(T val) {                                                   \
    for (int i = 1; i <= 16; i *= 2) {                                   \
      val = operator()(val, simd_shuffle_and_fill_up(val, init, i));     \
    }                                                                    \
    return val;                                                          \
  }

#define DEFINE_SIMD_EXCLUSIVE_SCAN()                                     \
  template <typename T, metal::enable_if_t<sizeof(T) < 8, bool> = true>  \
  T simd_exclusive_scan(T val) {                                         \
    return simd_exclusive_scan_impl(val);                                \
  }                                                                      \
                                                                         \
  template <typename T, metal::enable_if_t<sizeof(T) == 8, bool> = true> \
  T simd_exclusive_scan(T val) {                                         \
    val = simd_scan(val);                                                \
    return simd_shuffle_and_fill_up(val, init, 1);                       \
  }

template <typename U>
struct CumSum {
  DEFINE_SIMD_SCAN()
  DEFINE_SIMD_EXCLUSIVE_SCAN()

  static constexpr constant U init = static_cast<U>(0);

  template <typename T>
  U operator()(U a, T b) {
    return a + b;
  }

  U simd_scan_impl(U x) {
    return simd_prefix_inclusive_sum(x);
  }

  U simd_exclusive_scan_impl(U x) {
    return simd_prefix_exclusive_sum(x);
  }
};

template <typename U>
struct CumProd {
  DEFINE_SIMD_SCAN()
  DEFINE_SIMD_EXCLUSIVE_SCAN()

  static constexpr constant U init = static_cast<U>(1.0f);

  template <typename T>
  U operator()(U a, T b) {
    return a * b;
  }

  U simd_scan_impl(U x) {
    return simd_prefix_inclusive_product(x);
  }

  U simd_exclusive_scan_impl(U x) {
    return simd_prefix_exclusive_product(x);
  }
};

template <>
struct CumProd<bool> {
  static constexpr constant bool init = true;

  template <typename T>
  bool operator()(bool a, T b) {
    return a & static_cast<bool>(b);
  }

  bool simd_scan(bool x) {
    for (int i = 1; i <= 16; i *= 2) {
      bool other = simd_shuffle_and_fill_up(x, init, i);
      x &= other;
    }
    return x;
  }

  bool simd_exclusive_scan(bool x) {
    x = simd_scan(x);
    return simd_shuffle_and_fill_up(x, init, 1);
  }
};

template <typename U>
struct CumMax {
  static constexpr constant U init = Limits<U>::min;

  template <typename T>
  U operator()(U a, T b) {
    return (a >= b) ? a : b;
  }

  U simd_scan(U x) {
    for (int i = 1; i <= 16; i *= 2) {
      U other = simd_shuffle_and_fill_up(x, init, i);
      x = (x >= other) ? x : other;
    }
    return x;
  }

  U simd_exclusive_scan(U x) {
    x = simd_scan(x);
    return simd_shuffle_and_fill_up(x, init, 1);
  }
};

template <typename U>
struct CumMin {
  static constexpr constant U init = Limits<U>::max;

  template <typename T>
  U operator()(U a, T b) {
    return (a <= b) ? a : b;
  }

  U simd_scan(U x) {
    for (int i = 1; i <= 16; i *= 2) {
      U other = simd_shuffle_and_fill_up(x, init, i);
      x = (x <= other) ? x : other;
    }
    return x;
  }

  U simd_exclusive_scan(U x) {
    x = simd_scan(x);
    return simd_shuffle_and_fill_up(x, init, 1);
  }
};

template <typename U>
struct CumLogaddexp {
  static constexpr constant U init = Limits<U>::min;

  template <typename T>
  U operator()(U a, T b) {
    return LogAddExp{}(a, static_cast<U>(b));
  }

  U simd_scan(U x) {
    for (int i = 1; i <= 16; i *= 2) {
      U other = simd_shuffle_and_fill_up(x, init, i);
      x = LogAddExp{}(x, other);
    }
    return x;
  }

  U simd_exclusive_scan(U x) {
    x = simd_scan(x);
    return simd_shuffle_and_fill_up(x, init, 1);
  }
};

template <typename T, typename U, int N_READS, bool reverse>
inline void load_unsafe(U values[N_READS], const device T* input) {
  if (reverse) {
    for (int i = 0; i < N_READS; i++) {
      values[N_READS - i - 1] = input[i];
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      values[i] = input[i];
    }
  }
}

template <typename T, typename U, int N_READS, bool reverse>
inline void load_safe(
    U values[N_READS],
    const device T* input,
    int start,
    int total,
    U init) {
  if (reverse) {
    for (int i = 0; i < N_READS; i++) {
      values[N_READS - i - 1] =
          (start + N_READS - i - 1 < total) ? input[i] : init;
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      values[i] = (start + i < total) ? input[i] : init;
    }
  }
}

template <typename U, int N_READS, bool reverse>
inline void write_unsafe(U values[N_READS], device U* out) {
  if (reverse) {
    for (int i = 0; i < N_READS; i++) {
      out[i] = values[N_READS - i - 1];
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      out[i] = values[i];
    }
  }
}

template <typename U, int N_READS, bool reverse>
inline void write_safe(U values[N_READS], device U* out, int start, int total) {
  if (reverse) {
    for (int i = 0; i < N_READS; i++) {
      if (start + N_READS - i - 1 < total) {
        out[i] = values[N_READS - i - 1];
      }
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if (start + i < total) {
        out[i] = values[i];
      }
    }
  }
}

template <
    typename T,
    typename U,
    typename Op,
    int N_READS,
    bool inclusive,
    bool reverse>
[[kernel]] void contiguous_scan(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& axis_size [[buffer(2)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int simd_size = 32;
  Op op;

  // Position the pointers
  size_t offset = (gid.y + gsize.y * size_t(gid.z)) * axis_size;
  in += offset;
  out += offset;

  // Compute the number of simd_groups
  uint simd_groups = lsize.x / simd_size;

  // Allocate memory
  U prefix = Op::init;
  U values[N_READS];
  threadgroup U simdgroup_sums[32];

  // Loop over the reduced axis in blocks of size ceildiv(axis_size,
  // N_READS*lsize)
  //    Read block
  //    Compute inclusive scan of the block
  //      Compute inclusive scan per thread
  //      Compute exclusive scan of thread sums in simdgroup
  //      Write simdgroup sums in SM
  //      Compute exclusive scan of simdgroup sums
  //      Compute the output by scanning prefix, prev_simdgroup, prev_thread,
  //      value
  //    Write block

  for (uint r = 0; r < ceildiv(axis_size, N_READS * lsize.x); r++) {
    // Compute the block offset
    uint offset = r * lsize.x * N_READS + lid.x * N_READS;

    // Read the values
    if (reverse) {
      if ((offset + N_READS) < axis_size) {
        load_unsafe<T, U, N_READS, reverse>(
            values, in + axis_size - offset - N_READS);
      } else {
        load_safe<T, U, N_READS, reverse>(
            values,
            in + axis_size - offset - N_READS,
            offset,
            axis_size,
            Op::init);
      }
    } else {
      if ((offset + N_READS) < axis_size) {
        load_unsafe<T, U, N_READS, reverse>(values, in + offset);
      } else {
        load_safe<T, U, N_READS, reverse>(
            values, in + offset, offset, axis_size, Op::init);
      }
    }

    // Compute an inclusive scan per thread
    for (int i = 1; i < N_READS; i++) {
      values[i] = op(values[i], values[i - 1]);
    }

    // Compute exclusive scan of thread sums
    U prev_thread = op.simd_exclusive_scan(values[N_READS - 1]);

    // Write simdgroup_sums to SM
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == simd_size - 1) {
      simdgroup_sums[simd_group_id] = op(prev_thread, values[N_READS - 1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute exclusive scan of simdgroup_sums
    if (simd_group_id == 0) {
      U prev_simdgroup = op.simd_exclusive_scan(simdgroup_sums[simd_lane_id]);
      simdgroup_sums[simd_lane_id] = prev_simdgroup;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute the output
    for (int i = 0; i < N_READS; i++) {
      values[i] = op(values[i], prefix);
      values[i] = op(values[i], simdgroup_sums[simd_group_id]);
      values[i] = op(values[i], prev_thread);
    }

    // Write the values
    if (reverse) {
      if (inclusive) {
        if ((offset + N_READS) < axis_size) {
          write_unsafe<U, N_READS, reverse>(
              values, out + axis_size - offset - N_READS);
        } else {
          write_safe<U, N_READS, reverse>(
              values, out + axis_size - offset - N_READS, offset, axis_size);
        }
      } else {
        if (lid.x == 0 && offset == 0) {
          out[axis_size - 1] = Op::init;
        }
        if ((offset + N_READS + 1) < axis_size) {
          write_unsafe<U, N_READS, reverse>(
              values, out + axis_size - offset - 1 - N_READS);
        } else {
          write_safe<U, N_READS, reverse>(
              values,
              out + axis_size - offset - 1 - N_READS,
              offset + 1,
              axis_size);
        }
      }
    } else {
      if (inclusive) {
        if ((offset + N_READS) < axis_size) {
          write_unsafe<U, N_READS, reverse>(values, out + offset);
        } else {
          write_safe<U, N_READS, reverse>(
              values, out + offset, offset, axis_size);
        }
      } else {
        if (lid.x == 0 && offset == 0) {
          out[0] = Op::init;
        }
        if ((offset + N_READS + 1) < axis_size) {
          write_unsafe<U, N_READS, reverse>(values, out + offset + 1);
        } else {
          write_safe<U, N_READS, reverse>(
              values, out + offset + 1, offset + 1, axis_size);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Share the prefix
    if (simd_group_id == simd_groups - 1 && simd_lane_id == simd_size - 1) {
      simdgroup_sums[0] = values[N_READS - 1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prefix = simdgroup_sums[0];
  }
}

template <
    typename T,
    typename U,
    typename Op,
    int N_READS,
    bool inclusive,
    bool reverse>
[[kernel]] void strided_scan(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& axis_size [[buffer(2)]],
    const constant size_t& stride [[buffer(3)]],
    const constant size_t& stride_blocks [[buffer(4)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int simd_size = 32;
  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int BN_pad = 32 + 16 / sizeof(U);
  constexpr int n_simds = BN / N_READS;
  constexpr int n_scans = BN / n_simds;
  Op op;

  threadgroup U read_buffer[BM * BN_pad];
  U values[n_scans];
  U prefix[n_scans];
  for (int i = 0; i < n_scans; i++) {
    prefix[i] = Op::init;
  }

  // Compute offsets
  size_t full_gid = gid.y + gsize.y * size_t(gid.z);
  size_t offset = full_gid / stride_blocks * axis_size * stride;
  size_t global_index_x = full_gid % stride_blocks * BN;
  uint read_offset_y = (lid.x * N_READS) / BN;
  uint read_offset_x = (lid.x * N_READS) % BN;
  uint scan_offset_y = simd_lane_id;
  uint scan_offset_x = simd_group_id * n_scans;

  uint stride_limit = stride - global_index_x;
  in += offset + global_index_x + read_offset_x;
  out += offset + global_index_x + read_offset_x;
  threadgroup U* read_into =
      read_buffer + read_offset_y * BN_pad + read_offset_x;
  threadgroup U* read_from =
      read_buffer + scan_offset_y * BN_pad + scan_offset_x;

  for (uint j = 0; j < axis_size; j += BM) {
    // Calculate the indices for the current thread
    uint index_y = j + read_offset_y;
    uint check_index_y = index_y;
    if (reverse) {
      index_y = axis_size - 1 - index_y;
    }

    // Read in SM
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (check_index_y < axis_size && (read_offset_x + N_READS) < stride_limit) {
      for (int i = 0; i < N_READS; i++) {
        read_into[i] = in[index_y * stride + i];
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if (check_index_y < axis_size && (read_offset_x + i) < stride_limit) {
          read_into[i] = in[index_y * stride + i];
        } else {
          read_into[i] = Op::init;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Read strided into registers
    for (int i = 0; i < n_scans; i++) {
      values[i] = read_from[i];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Perform the scan
    for (int i = 0; i < n_scans; i++) {
      values[i] = op.simd_scan(values[i]);
      values[i] = op(values[i], prefix[i]);
      prefix[i] = simd_shuffle(values[i], simd_size - 1);
    }

    // Write to SM
    for (int i = 0; i < n_scans; i++) {
      read_from[i] = values[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write to device memory
    if (!inclusive) {
      if (check_index_y == 0) {
        if ((read_offset_x + N_READS) < stride_limit) {
          for (int i = 0; i < N_READS; i++) {
            out[index_y * stride + i] = Op::init;
          }
        } else {
          for (int i = 0; i < N_READS; i++) {
            if ((read_offset_x + i) < stride_limit) {
              out[index_y * stride + i] = Op::init;
            }
          }
        }
      }
      if (reverse) {
        index_y -= 1;
        check_index_y += 1;
      } else {
        index_y += 1;
        check_index_y += 1;
      }
    }
    if (check_index_y < axis_size && (read_offset_x + N_READS) < stride_limit) {
      for (int i = 0; i < N_READS; i++) {
        out[index_y * stride + i] = read_into[i];
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if (check_index_y < axis_size && (read_offset_x + i) < stride_limit) {
          out[index_y * stride + i] = read_into[i];
        }
      }
    }
  }
}
// ===== end:   /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/scan.h =====
#line 13 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/scan.metal"

#define instantiate_contiguous_scan(                                    \
    name, itype, otype, op, inclusive, reverse, nreads)                 \
  template [[host_name("contig_scan_" #name)]] [[kernel]] void          \
  contiguous_scan<itype, otype, op<otype>, nreads, inclusive, reverse>( \
      const device itype* in [[buffer(0)]],                             \
      device otype* out [[buffer(1)]],                                  \
      const constant size_t& axis_size [[buffer(2)]],                   \
      uint3 gid [[threadgroup_position_in_grid]],                       \
      uint3 gsize [[threadgroups_per_grid]],                            \
      uint3 lid [[thread_position_in_threadgroup]],                     \
      uint3 lsize [[threads_per_threadgroup]],                          \
      uint simd_lane_id [[thread_index_in_simdgroup]],                  \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_strided_scan(                                    \
    name, itype, otype, op, inclusive, reverse, nreads)              \
  template [[host_name("strided_scan_" #name)]] [[kernel]] void      \
  strided_scan<itype, otype, op<otype>, nreads, inclusive, reverse>( \
      const device itype* in [[buffer(0)]],                          \
      device otype* out [[buffer(1)]],                               \
      const constant size_t& axis_size [[buffer(2)]],                \
      const constant size_t& stride [[buffer(3)]],                   \
      const constant size_t& stride_blocks [[buffer(4)]],            \
      uint3 gid [[threadgroup_position_in_grid]],                    \
      uint3 gsize [[threadgroups_per_grid]],                         \
      uint3 lid [[thread_position_in_threadgroup]],                  \
      uint simd_lane_id [[thread_index_in_simdgroup]],               \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_scan_helper(name, itype, otype, op, nreads)                                \
  instantiate_contiguous_scan(inclusive_##name, itype, otype, op, true, false, nreads)         \
  instantiate_contiguous_scan(exclusive_##name, itype, otype, op, false, false, nreads)        \
  instantiate_contiguous_scan(reverse_inclusive_##name, itype, otype, op, true, true, nreads)  \
  instantiate_contiguous_scan(reverse_exclusive_##name, itype, otype, op, false, true, nreads) \
  instantiate_strided_scan(inclusive_##name, itype, otype, op, true, false, nreads)            \
  instantiate_strided_scan(exclusive_##name, itype, otype, op, false, false, nreads)           \
  instantiate_strided_scan(reverse_inclusive_##name, itype, otype, op, true, true, nreads)     \
  instantiate_strided_scan(reverse_exclusive_##name, itype, otype, op, false, true, nreads)

instantiate_scan_helper(sum_bool__int32,         bool,        int32_t,     CumSum, 4)
instantiate_scan_helper(sum_bool__uint32,        bool,        uint32_t,    CumSum, 4)
instantiate_scan_helper(sum_uint8_uint8,         uint8_t,     uint8_t,     CumSum, 4)
instantiate_scan_helper(sum_uint16_uint16,       uint16_t,    uint16_t,    CumSum, 4)
instantiate_scan_helper(sum_uint32_uint32,       uint32_t,    uint32_t,    CumSum, 4)
instantiate_scan_helper(sum_uint64_uint64,       uint64_t,    uint64_t,    CumSum, 2)
instantiate_scan_helper(sum_int8_int8,           int8_t,      int8_t,      CumSum, 4)
instantiate_scan_helper(sum_int16_int16,         int16_t,     int16_t,     CumSum, 4)
instantiate_scan_helper(sum_int32_int32,         int32_t,     int32_t,     CumSum, 4)
instantiate_scan_helper(sum_int64_int64,         int64_t,     int64_t,     CumSum, 2)
instantiate_scan_helper(sum_float16_float16,     half,        half,        CumSum, 4)
instantiate_scan_helper(sum_float32_float32,     float,       float,       CumSum, 4)
instantiate_scan_helper(sum_bfloat16_bfloat16,   bfloat16_t,  bfloat16_t,  CumSum, 4)
instantiate_scan_helper(sum_complex64_complex64, complex64_t, complex64_t, CumSum, 2)
instantiate_scan_helper(prod_bool__bool_,         bool,        bool,        CumProd, 4)
instantiate_scan_helper(prod_uint8_uint8,         uint8_t,     uint8_t,     CumProd, 4)
instantiate_scan_helper(prod_uint16_uint16,       uint16_t,    uint16_t,    CumProd, 4)
instantiate_scan_helper(prod_uint32_uint32,       uint32_t,    uint32_t,    CumProd, 4)
instantiate_scan_helper(prod_uint64_uint64,       uint64_t,    uint64_t,    CumProd, 2)
instantiate_scan_helper(prod_int8_int8,           int8_t,      int8_t,      CumProd, 4)
instantiate_scan_helper(prod_int16_int16,         int16_t,     int16_t,     CumProd, 4)
instantiate_scan_helper(prod_int32_int32,         int32_t,     int32_t,     CumProd, 4)
instantiate_scan_helper(prod_int64_int64,         int64_t,     int64_t,     CumProd, 2)
instantiate_scan_helper(prod_float16_float16,     half,        half,        CumProd, 4)
instantiate_scan_helper(prod_float32_float32,     float,       float,       CumProd, 4)
instantiate_scan_helper(prod_bfloat16_bfloat16,   bfloat16_t,  bfloat16_t,  CumProd, 4)
instantiate_scan_helper(prod_complex64_complex64, complex64_t, complex64_t, CumProd, 2)
instantiate_scan_helper(max_bool__bool_,         bool,        bool,        CumMax, 4)
instantiate_scan_helper(max_uint8_uint8,         uint8_t,     uint8_t,     CumMax, 4)
instantiate_scan_helper(max_uint16_uint16,       uint16_t,    uint16_t,    CumMax, 4)
instantiate_scan_helper(max_uint32_uint32,       uint32_t,    uint32_t,    CumMax, 4)
instantiate_scan_helper(max_uint64_uint64,       uint64_t,    uint64_t,    CumMax, 2)
instantiate_scan_helper(max_int8_int8,           int8_t,      int8_t,      CumMax, 4)
instantiate_scan_helper(max_int16_int16,         int16_t,     int16_t,     CumMax, 4)
instantiate_scan_helper(max_int32_int32,         int32_t,     int32_t,     CumMax, 4)
instantiate_scan_helper(max_int64_int64,         int64_t,     int64_t,     CumMax, 2)
instantiate_scan_helper(max_float16_float16,     half,        half,        CumMax, 4)
instantiate_scan_helper(max_float32_float32,     float,       float,       CumMax, 4)
instantiate_scan_helper(max_bfloat16_bfloat16,   bfloat16_t,  bfloat16_t,  CumMax, 4)
instantiate_scan_helper(max_complex64_complex64, complex64_t, complex64_t, CumMax, 2)
instantiate_scan_helper(min_bool__bool_,         bool,        bool,        CumMin, 4)
instantiate_scan_helper(min_uint8_uint8,         uint8_t,     uint8_t,     CumMin, 4)
instantiate_scan_helper(min_uint16_uint16,       uint16_t,    uint16_t,    CumMin, 4)
instantiate_scan_helper(min_uint32_uint32,       uint32_t,    uint32_t,    CumMin, 4)
instantiate_scan_helper(min_uint64_uint64,       uint64_t,    uint64_t,    CumMin, 2)
instantiate_scan_helper(min_int8_int8,           int8_t,      int8_t,      CumMin, 4)
instantiate_scan_helper(min_int16_int16,         int16_t,     int16_t,     CumMin, 4)
instantiate_scan_helper(min_int32_int32,         int32_t,     int32_t,     CumMin, 4)
instantiate_scan_helper(min_int64_int64,         int64_t,     int64_t,     CumMin, 2)
instantiate_scan_helper(min_float16_float16,     half,        half,        CumMin, 4)
instantiate_scan_helper(min_float32_float32,     float,       float,       CumMin, 4)
instantiate_scan_helper(min_bfloat16_bfloat16,   bfloat16_t,  bfloat16_t,  CumMin, 4)
instantiate_scan_helper(min_complex64_complex64, complex64_t, complex64_t, CumMin, 2)
instantiate_scan_helper(logaddexp_float16_float16,     half,        half,        CumLogaddexp, 4)
instantiate_scan_helper(logaddexp_float32_float32,     float,       float,       CumLogaddexp, 4)
instantiate_scan_helper(logaddexp_bfloat16_bfloat16,   bfloat16_t,  bfloat16_t,  CumLogaddexp, 4)
instantiate_scan_helper(logaddexp_complex64_complex64, complex64_t, complex64_t, CumLogaddexp, 2) // clang-format on
// ===== end:   /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/scan.metal =====
