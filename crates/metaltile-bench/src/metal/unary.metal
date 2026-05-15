// ===== begin: /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/unary.metal =====
#line 1 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/unary.metal"
// Copyright © 2024 Apple Inc.

// clang-format off
// ----- expanded "mlx/backend/metal/kernels/utils.h" from /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/unary.metal:4 -----
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
#line 5 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/unary.metal"
// ----- expanded "mlx/backend/metal/kernels/unary_ops.h" from /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/unary.metal:5 -----
// ===== begin: /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/unary_ops.h =====
#line 1 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/unary_ops.h"
// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <metal_integer>
#include <metal_math>

// ----- expanded "mlx/backend/metal/kernels/cexpf.h" from /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/unary_ops.h:8 -----
// ===== begin: /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/cexpf.h =====
#line 1 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/cexpf.h"
// Copyright © 2025 Apple Inc.
// Copyright © 2008-2013 NVIDIA Corporation
// Copyright © 2013 Filipe RNC Maia
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Forked from
// https://github.com/NVIDIA/cccl/blob/main/thrust/thrust/detail/complex/cexpf.h

// TODO: We should use thrust::exp but the thrust header in old CUDA versions
// can not be used in JIT.

#pragma once

#include <metal_math>

using ieee_float_shape_type = union {
  float value;
  uint32_t word;
};

inline void get_float_word(thread uint32_t& i, float d) {
  ieee_float_shape_type gf_u;
  gf_u.value = (d);
  (i) = gf_u.word;
}

inline void get_float_word(thread int32_t& i, float d) {
  ieee_float_shape_type gf_u;
  gf_u.value = (d);
  (i) = gf_u.word;
}

inline void set_float_word(thread float& d, uint32_t i) {
  ieee_float_shape_type sf_u;
  sf_u.word = (i);
  (d) = sf_u.value;
}

inline float frexp_expf(float x, thread int* expt) {
  const uint32_t k = 235;
  const float kln2 = 162.88958740F;

  float exp_x;
  uint32_t hx;

  exp_x = metal::exp(x - kln2);
  get_float_word(hx, exp_x);
  *expt = (hx >> 23) - (0x7f + 127) + k;
  set_float_word(exp_x, (hx & 0x7fffff) | ((0x7f + 127) << 23));
  return exp_x;
}

inline complex64_t ldexp_cexpf(complex64_t z, int expt) {
  float x, y, exp_x, scale1, scale2;
  int ex_expt, half_expt;

  x = z.real;
  y = z.imag;
  exp_x = frexp_expf(x, &ex_expt);
  expt += ex_expt;

  half_expt = expt / 2;
  set_float_word(scale1, (0x7f + half_expt) << 23);
  half_expt = expt - half_expt;
  set_float_word(scale2, (0x7f + half_expt) << 23);

  return complex64_t{
      metal::cos(y) * exp_x * scale1 * scale2,
      metal::sin(y) * exp_x * scale1 * scale2};
}

inline complex64_t cexpf(const thread complex64_t& z) {
  float x, y, exp_x;
  uint32_t hx, hy;

  const uint32_t exp_ovfl = 0x42b17218, cexp_ovfl = 0x43400074;

  x = z.real;
  y = z.imag;

  get_float_word(hy, y);
  hy &= 0x7fffffff;

  /* cexp(x + I 0) = exp(x) + I 0 */
  if (hy == 0) {
    return complex64_t{metal::exp(x), y};
  }
  get_float_word(hx, x);
  /* cexp(0 + I y) = cos(y) + I sin(y) */
  if ((hx & 0x7fffffff) == 0) {
    return complex64_t{metal::cos(y), metal::sin(y)};
  }
  if (hy >= 0x7f800000) {
    if ((hx & 0x7fffffff) != 0x7f800000) {
      /* cexp(finite|NaN +- I Inf|NaN) = NaN + I NaN */
      return complex64_t{y - y, y - y};
    } else if (hx & 0x80000000) {
      /* cexp(-Inf +- I Inf|NaN) = 0 + I 0 */
      return complex64_t{0.0, 0.0};
    } else {
      /* cexp(+Inf +- I Inf|NaN) = Inf + I NaN */
      return complex64_t{x, y - y};
    }
  }

  if (hx >= exp_ovfl && hx <= cexp_ovfl) {
    /*
     * x is between 88.7 and 192, so we must scale to avoid
     * overflow in expf(x).
     */
    return ldexp_cexpf(z, 0);
  } else {
    /*
     * Cases covered here:
     *  -  x < exp_ovfl and exp(x) won't overflow (common case)
     *  -  x > cexp_ovfl, so exp(x) * s overflows for all s > 0
     *  -  x = +-Inf (generated by exp())
     *  -  x = NaN (spurious inexact exception from y)
     */
    exp_x = metal::exp(x);
    return complex64_t{exp_x * metal::cos(y), exp_x * metal::sin(y)};
  }
}
// ===== end:   /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/cexpf.h =====
#line 9 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/unary_ops.h"
// ----- expanded "mlx/backend/metal/kernels/erf.h" from /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/unary_ops.h:9 -----
// ===== begin: /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/erf.h =====
#line 1 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/erf.h"
// Copyright © 2023 Apple Inc.

#pragma once
#include <metal_math>
// ----- expanded "mlx/backend/metal/kernels/expm1f.h" from /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/erf.h:5 -----
// ===== begin: /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/expm1f.h =====
#line 1 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/expm1f.h"
// Copyright © 2023 Apple Inc.

#pragma once

#include <metal_math>

// Original license copied below:
//  Copyright (c) 2015-2023 Norbert Juffa
//  All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions
//  are met:
//
//  1. Redistributions of source code must retain the above copyright
//     notice, this list of conditions and the following disclaimer.
//
//  2. Redistributions in binary form must reproduce the above copyright
//     notice, this list of conditions and the following disclaimer in the
//     documentation and/or other materials provided with the distribution.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
//  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
//  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
//  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
//  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
//  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
//  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/* Compute exponential base e minus 1. Maximum ulp error = 0.997458

   i = rint(a/log(2)), f = a-i*log(2). Then expm1(a) = 2**i * (expm1(f)+1) - 1.
   Compute r = expm1(f). Then expm1(a)= 2 * (0.5 * 2**i * r + 0.5 * 2**i - 0.5).
   With t = 0.5*2**i, expm1(a) = 2*(r * t + t-0.5). However, for best accuracy,
   when i == 1, expm1(a)= 2*(r + 0.5), and when i == 0, expm1(a) = r.

   NOTE: Scale factor b is only applied if i < 0 or i > 1 (should be power of 2)
*/
float expm1f_scaled_unchecked(float a, float b) {
  float f, j, r, s, t, u, v, x, y;
  int i;

  // exp(a) = 2**i * exp(f); i = rintf (a / log(2))
  j = fma(1.442695f, a, 12582912.f); // 0x1.715476p0, 0x1.8p23
  j = j - 12582912.0f; // 0x1.8p23
  i = (int)j;
  f = fma(j, -6.93145752e-1f, a);

  // approximate r = exp(f)-1 on interval [-log(2)/2, +log(2)/2]
  s = f * f;
  if (a == 0.0f)
    s = a; // ensure -0 is passed through
  // err = 0.997458  ulp1 = 11081805
  r = 1.97350979e-4f; // 0x1.9de000p-13
  r = fma(r, f, 1.39309070e-3f); // 0x1.6d30bcp-10
  r = fma(r, f, 8.33343994e-3f); // 0x1.1111f6p-7
  r = fma(r, f, 4.16668020e-2f); // 0x1.55559ep-5
  r = fma(r, f, 1.66666716e-1f); // 0x1.55555cp-3
  r = fma(r, f, 4.99999970e-1f); // 0x1.fffffep-2
  u = (j == 1) ? (f + 0.5f) : f;
  v = fma(r, s, u);
  s = 0.5f * b;
  t = ldexp(s, i);
  y = t - s;
  x = (t - y) - s; // double-float canonicalization of difference
  r = fma(v, t, x) + y;
  r = r + r;
  if (j == 0)
    r = v;
  if (j == 1)
    r = v + v;
  return r;
}

/* Compute exponential base e minus 1. max ulp err = 0.99746 */
float expm1f(float a) {
  float r;

  r = expm1f_scaled_unchecked(a, 1.0f);
  /* handle severe overflow and underflow */
  if (abs(a - 1.0f) > 88.0f) {
    r = pow(2, a);
    r = fma(r, r, -1.0f);
  }
  return r;
}
// ===== end:   /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/expm1f.h =====
#line 6 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/erf.h"

/*
 * Approximation to the error function.
 * Based on code from:
 * https://stackoverflow.com/questions/35148198/efficient-faithfully-rounded-implementation-of-error-function-erff#answer-35148199
 */
float erf(float a) {
  float r, s, t, u;
  t = metal::abs(a);
  s = a * a;
  if (t > 0.927734375f) {
    // maximum error 0.99527 ulp
    r = metal::fma(
        -1.72853470e-5f, t, 3.83197126e-4f); // -0x1.220000p-16,0x1.91cfb2p-12
    u = metal::fma(
        -3.88396438e-3f, t, 2.42546219e-2f); // -0x1.fd1438p-9, 0x1.8d6342p-6
    r = metal::fma(r, s, u);
    r = metal::fma(r, t, -1.06777877e-1f); // -0x1.b55cb8p-4
    r = metal::fma(r, t, -6.34846687e-1f); // -0x1.450aa0p-1
    r = metal::fma(r, t, -1.28717512e-1f); // -0x1.079d0cp-3
    r = metal::fma(r, t, -t);
    r = -expm1f(r);
    r = metal::copysign(r, a);
  } else {
    // maximum error 0.98929 ulp
    r = -5.96761703e-4f; // -0x1.38e000p-11
    r = metal::fma(r, s, 4.99119423e-3f); //  0x1.471a58p-8
    r = metal::fma(r, s, -2.67681349e-2f); // -0x1.b691b2p-6
    r = metal::fma(r, s, 1.12819925e-1f); //  0x1.ce1c44p-4
    r = metal::fma(r, s, -3.76125336e-1f); // -0x1.812700p-2
    r = metal::fma(r, s, 1.28379166e-1f); //  0x1.06eba8p-3
    r = metal::fma(r, a, a);
  }
  return r;
}

float erfinv(float a) {
  auto t = metal::fma(a, 0.0f - a, 1.0f);
  t = metal::log(t);
  float p;
  if (metal::abs(t) > 6.125f) { // maximum ulp error = 2.35793
    p = 3.03697567e-10f; //  0x1.4deb44p-32
    p = metal::fma(p, t, 2.93243101e-8f); //  0x1.f7c9aep-26
    p = metal::fma(p, t, 1.22150334e-6f); //  0x1.47e512p-20
    p = metal::fma(p, t, 2.84108955e-5f); //  0x1.dca7dep-16
    p = metal::fma(p, t, 3.93552968e-4f); //  0x1.9cab92p-12
    p = metal::fma(p, t, 3.02698812e-3f); //  0x1.8cc0dep-9
    p = metal::fma(p, t, 4.83185798e-3f); //  0x1.3ca920p-8
    p = metal::fma(p, t, -2.64646143e-1f); // -0x1.0eff66p-2
    p = metal::fma(p, t, 8.40016484e-1f); //  0x1.ae16a4p-1
  } else { // maximum ulp error = 2.35002
    p = 5.43877832e-9f; //  0x1.75c000p-28
    p = metal::fma(p, t, 1.43285448e-7f); //  0x1.33b402p-23
    p = metal::fma(p, t, 1.22774793e-6f); //  0x1.499232p-20
    p = metal::fma(p, t, 1.12963626e-7f); //  0x1.e52cd2p-24
    p = metal::fma(p, t, -5.61530760e-5f); // -0x1.d70bd0p-15
    p = metal::fma(p, t, -1.47697632e-4f); // -0x1.35be90p-13
    p = metal::fma(p, t, 2.31468678e-3f); //  0x1.2f6400p-9
    p = metal::fma(p, t, 1.15392581e-2f); //  0x1.7a1e50p-7
    p = metal::fma(p, t, -2.32015476e-1f); // -0x1.db2aeep-3
    p = metal::fma(p, t, 8.86226892e-1f); //  0x1.c5bf88p-1
  }
  return a * p;
}
// ===== end:   /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/erf.h =====
#line 10 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/unary_ops.h"
// ----- expanded "mlx/backend/metal/kernels/expm1f.h" from /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/unary_ops.h:10 -----
// [metal_flatten] skipped duplicate include: /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/expm1f.h
#line 11 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/unary_ops.h"
// ----- expanded "mlx/backend/metal/kernels/fp8.h" from /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/unary_ops.h:11 -----
// ===== begin: /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/fp8.h =====
#line 1 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/fp8.h"
#pragma once

struct fp8_e4m3 {
  template <typename T>
  fp8_e4m3(T f) {
    // From PyTorch
    // https://github.com/pytorch/pytorch/blob/e3643e1e0e923f0fc063dfab6f45c956d568919d/c10/util/Float8_e4m3fn.h#L148
    uint32_t fp8_max = 543 << 21;
    uint32_t denorm_mask = 141 << 23;
    uint32_t f_bits = as_type<uint32_t>(static_cast<float>(f));
    uint32_t sign = f_bits & 0x80000000;
    f_bits ^= sign;
    if (f_bits >= fp8_max) {
      // Default behavior saturates to min/max
      bits = 0x7E;
    } else {
      if (f_bits < (121 << 23)) {
        f_bits = as_type<uint32_t>(
            as_type<float>(f_bits) + as_type<float>(denorm_mask));
        bits = static_cast<uint8_t>(f_bits - denorm_mask);
      } else {
        // resulting mantissa is odd
        uint8_t mant_odd = (f_bits >> 20) & 1;
        f_bits += ((uint32_t)(7 - 127) << 23) + 0x7FFFF;
        f_bits += mant_odd;
        bits = static_cast<uint8_t>(f_bits >> 20);
      }
    }
    bits |= static_cast<uint8_t>(sign >> 24);
  }

  operator float16_t() {
    uint16_t v = (bits & 127) << 7;
    half converted = as_type<half>(v);
    converted *= 256.0;
    auto sign = bits & 128;
    return (sign ? -converted : converted);
  }

  operator bfloat16_t() {
    return static_cast<bfloat16_t>(this->operator float16_t());
  }

  operator float() {
    return static_cast<float>(this->operator float16_t());
  }

  uint8_t bits;
};

struct fp8_e8m0 {
  fp8_e8m0(float x) {
    if (!metal::isfinite(x)) {
      bits = 0xFF;
      return;
    }
    if (x < 0.0f) {
      bits = 0x00;
      return;
    }
    float le = metal::log2(x);
    int n = int(metal::round(le));

    n = n < -127 ? -127 : n;
    n = n > 127 ? 127 : n;
    bits = static_cast<uint8_t>(n + 127);
  }

  operator bfloat16_t() {
    uint16_t out = (bits == 0 ? 0x40 : (static_cast<uint16_t>(bits) << 7));
    return as_type<bfloat16_t>(out);
  }

  operator float() {
    uint32_t out = (bits == 0 ? 0x400000 : (static_cast<uint16_t>(bits) << 23));
    return as_type<float>(out);
  }

  uint8_t bits;
};
// ===== end:   /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/fp8.h =====
#line 12 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/unary_ops.h"

namespace {
constant float inf = metal::numeric_limits<float>::infinity();
}

struct Abs {
  template <typename T>
  T operator()(T x) {
    return metal::abs(x);
  };
  uint8_t operator()(uint8_t x) {
    return x;
  };
  uint16_t operator()(uint16_t x) {
    return x;
  };
  uint32_t operator()(uint32_t x) {
    return x;
  };
  uint64_t operator()(uint64_t x) {
    return x;
  };
  bool operator()(bool x) {
    return x;
  };
  complex64_t operator()(complex64_t x) {
    return {metal::precise::sqrt(x.real * x.real + x.imag * x.imag), 0};
  };
};

struct ArcCos {
  template <typename T>
  T operator()(T x) {
    return metal::precise::acos(x);
  };

  complex64_t operator()(complex64_t x);
};

struct ArcCosh {
  template <typename T>
  T operator()(T x) {
    return metal::precise::acosh(x);
  };
};

struct ArcSin {
  template <typename T>
  T operator()(T x) {
    return metal::precise::asin(x);
  };

  complex64_t operator()(complex64_t x);
};

struct ArcSinh {
  template <typename T>
  T operator()(T x) {
    return metal::precise::asinh(x);
  };
};

struct ArcTan {
  template <typename T>
  T operator()(T x) {
    return metal::precise::atan(x);
  };

  complex64_t operator()(complex64_t x);
};

struct ArcTanh {
  template <typename T>
  T operator()(T x) {
    return metal::precise::atanh(x);
  };
};

struct BitwiseInvert {
  template <typename T>
  T operator()(T x) {
    return ~x;
  };
};

struct Ceil {
  template <typename T>
  T operator()(T x) {
    return metal::ceil(x);
  };
  int8_t operator()(int8_t x) {
    return x;
  };
  int16_t operator()(int16_t x) {
    return x;
  };
  int32_t operator()(int32_t x) {
    return x;
  };
  int64_t operator()(int64_t x) {
    return x;
  };
  uint8_t operator()(uint8_t x) {
    return x;
  };
  uint16_t operator()(uint16_t x) {
    return x;
  };
  uint32_t operator()(uint32_t x) {
    return x;
  };
  uint64_t operator()(uint64_t x) {
    return x;
  };
  bool operator()(bool x) {
    return x;
  };
};

struct Cos {
  template <typename T>
  T operator()(T x) {
    return metal::precise::cos(x);
  };

  complex64_t operator()(complex64_t x) {
    return {
        metal::precise::cos(x.real) * metal::precise::cosh(x.imag),
        -metal::precise::sin(x.real) * metal::precise::sinh(x.imag)};
  };
};

struct Cosh {
  template <typename T>
  T operator()(T x) {
    return metal::precise::cosh(x);
  };

  complex64_t operator()(complex64_t x) {
    return {
        metal::precise::cosh(x.real) * metal::precise::cos(x.imag),
        metal::precise::sinh(x.real) * metal::precise::sin(x.imag)};
  };
};

struct Conjugate {
  complex64_t operator()(complex64_t x) {
    return complex64_t{x.real, -x.imag};
  }
};

struct Erf {
  template <typename T>
  T operator()(T x) {
    return static_cast<T>(erf(static_cast<float>(x)));
  };
};

struct ErfInv {
  template <typename T>
  T operator()(T x) {
    return static_cast<T>(erfinv(static_cast<float>(x)));
  };
};

struct Exp {
  template <typename T>
  T operator()(T x) {
    return metal::precise::exp(x);
  };
  complex64_t operator()(complex64_t x) {
    return cexpf(x);
  }
};

struct Expm1 {
  template <typename T>
  T operator()(T x) {
    return static_cast<T>(expm1f(static_cast<float>(x)));
  };
};

struct Floor {
  template <typename T>
  T operator()(T x) {
    return metal::floor(x);
  };
  int8_t operator()(int8_t x) {
    return x;
  };
  int16_t operator()(int16_t x) {
    return x;
  };
  int32_t operator()(int32_t x) {
    return x;
  };
  int64_t operator()(int64_t x) {
    return x;
  };
  uint8_t operator()(uint8_t x) {
    return x;
  };
  uint16_t operator()(uint16_t x) {
    return x;
  };
  uint32_t operator()(uint32_t x) {
    return x;
  };
  uint64_t operator()(uint64_t x) {
    return x;
  };
  bool operator()(bool x) {
    return x;
  };
};

struct Imag {
  float operator()(complex64_t x) {
    return x.imag;
  };
};

struct Log {
  template <typename T>
  T operator()(T x) {
    return metal::precise::log(x);
  };

  complex64_t operator()(complex64_t x) {
    auto r = metal::precise::log(Abs{}(x).real);
    auto i = metal::precise::atan2(x.imag, x.real);
    return {r, i};
  };
};

struct Log2 {
  template <typename T>
  T operator()(T x) {
    return metal::precise::log2(x);
  };

  complex64_t operator()(complex64_t x) {
    auto y = Log{}(x);
    return {y.real / M_LN2_F, y.imag / M_LN2_F};
  };
};

struct Log10 {
  template <typename T>
  T operator()(T x) {
    return metal::precise::log10(x);
  };

  complex64_t operator()(complex64_t x) {
    auto y = Log{}(x);
    return {y.real / M_LN10_F, y.imag / M_LN10_F};
  };
};

struct Log1p {
  template <typename T>
  T operator()(T x) {
    return log1p(x);
  };
};

struct LogicalNot {
  template <typename T>
  T operator()(T x) {
    return !x;
  };
};

struct Negative {
  template <typename T>
  T operator()(T x) {
    return -x;
  };
};

struct Real {
  float operator()(complex64_t x) {
    return x.real;
  };
};

struct Round {
  template <typename T>
  T operator()(T x) {
    return metal::rint(x);
  };
  complex64_t operator()(complex64_t x) {
    return {metal::rint(x.real), metal::rint(x.imag)};
  };
};

struct Sigmoid {
  template <typename T>
  T operator()(T x) {
    auto y = 1 / (1 + metal::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
  }
};

struct Sign {
  template <typename T>
  T operator()(T x) {
    return (x > T(0)) - (x < T(0));
  };
  uint32_t operator()(uint32_t x) {
    return x != 0;
  };
  complex64_t operator()(complex64_t x) {
    if (x == complex64_t(0)) {
      return x;
    }
    return x /
        (complex64_t)metal::precise::sqrt(x.real * x.real + x.imag * x.imag);
  };
};

struct Sin {
  template <typename T>
  T operator()(T x) {
    return metal::precise::sin(x);
  };

  complex64_t operator()(complex64_t x) {
    return {
        metal::precise::sin(x.real) * metal::precise::cosh(x.imag),
        metal::precise::cos(x.real) * metal::precise::sinh(x.imag)};
  };
};

struct Sinh {
  template <typename T>
  T operator()(T x) {
    return metal::precise::sinh(x);
  };

  complex64_t operator()(complex64_t x) {
    return {
        metal::precise::sinh(x.real) * metal::precise::cos(x.imag),
        metal::precise::cosh(x.real) * metal::precise::sin(x.imag)};
  };
};

struct Square {
  template <typename T>
  T operator()(T x) {
    return x * x;
  };
};

struct Sqrt {
  template <typename T>
  T operator()(T x) {
    return metal::precise::sqrt(x);
  };

  complex64_t operator()(complex64_t x) {
    if (x.real == 0.0 && x.imag == 0.0) {
      return {0.0, 0.0};
    }
    auto r = Abs{}(x).real;
    auto a = metal::precise::sqrt((r + x.real) / 2.0);
    auto b_abs = metal::precise::sqrt((r - x.real) / 2.0);
    auto b = metal::copysign(b_abs, x.imag);
    return {a, b};
  }
};

struct Rsqrt {
  template <typename T>
  T operator()(T x) {
    return metal::precise::rsqrt(x);
  };

  complex64_t operator()(complex64_t x) {
    return 1.0 / Sqrt{}(x);
  }
};

struct Tan {
  template <typename T>
  T operator()(T x) {
    return metal::precise::tan(x);
  };

  complex64_t operator()(complex64_t x) {
    float tan_a = metal::precise::tan(x.real);
    float tanh_b = metal::precise::tanh(x.imag);
    float t1 = tan_a * tanh_b;
    float denom = 1. + t1 * t1;
    return {(tan_a - tanh_b * t1) / denom, (tanh_b + tan_a * t1) / denom};
  };
};

struct Tanh {
  template <typename T>
  T operator()(T x) {
    return metal::precise::tanh(x);
  };

  complex64_t operator()(complex64_t x) {
    float tanh_a = metal::precise::tanh(x.real);
    float tan_b = metal::precise::tan(x.imag);
    float t1 = tanh_a * tan_b;
    float denom = 1. + t1 * t1;
    return {(tanh_a + tan_b * t1) / denom, (tan_b - tanh_a * t1) / denom};
  };
};

complex64_t ArcCos::operator()(complex64_t x) {
  auto i = complex64_t{0.0, 1.0};
  auto y = Log{}(x + i * Sqrt{}(1.0 - x * x));
  return {y.imag, -y.real};
};

complex64_t ArcSin::operator()(complex64_t x) {
  auto i = complex64_t{0.0, 1.0};
  auto y = Log{}(i * x + Sqrt{}(1.0 - x * x));
  return {y.imag, -y.real};
};

complex64_t ArcTan::operator()(complex64_t x) {
  auto i = complex64_t{0.0, 1.0};
  auto ix = i * x;
  return (1.0 / complex64_t{0.0, 2.0}) * Log{}((1.0 + ix) / (1.0 - ix));
};

struct ToFP8 {
  template <typename T>
  uint8_t operator()(T f) {
    return fp8_e4m3(f).bits;
  }
};

struct FromFP8 {
  float operator()(uint8_t x) {
    return float(*(thread fp8_e4m3*)(&x));
  }
};
// ===== end:   /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/unary_ops.h =====
#line 6 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/unary.metal"
// ----- expanded "mlx/backend/metal/kernels/unary.h" from /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/unary.metal:6 -----
// ===== begin: /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/unary.h =====
#line 1 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/unary.h"
// Copyright © 2024 Apple Inc.

template <typename T, typename U, typename Op, int N = WorkPerThread<T>::n>
[[kernel]] void unary_v(
    device const T* in,
    device U* out,
    constant uint& size,
    uint index [[thread_position_in_grid]]) {
  index *= N;
  if (N > 1 && index + N > size) {
    for (int i = 0; index + i < size; ++i) {
      out[index + i] = static_cast<U>(Op()(in[index + i]));
    }
  } else {
    for (int i = 0; i < N; ++i) {
      out[index + i] = static_cast<U>(Op()(in[index + i]));
    }
  }
}

template <typename T, typename U, typename Op, int N = WorkPerThread<T>::n>
[[kernel]] void unary_v2(
    device const T* in,
    device U* out,
    constant int64_t& size,
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {
  int64_t offset = N * (index.x + grid_dim.x * int64_t(index.y));
  if (N > 1 && offset + N > size) {
    for (int i = 0; offset + i < size; ++i) {
      out[offset + i] = static_cast<U>(Op()(in[offset + i]));
    }
  } else {
    for (int i = 0; i < N; ++i) {
      out[offset + i] = static_cast<U>(Op()(in[offset + i]));
    }
  }
}

template <
    typename T,
    typename U,
    typename Op,
    int N = 1,
    typename IdxT = int64_t>
[[kernel]] void unary_g(
    device const T* in,
    device U* out,
    constant const int* in_shape,
    constant const int64_t* in_strides,
    device const int& ndim,
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  auto idx = elem_to_loc<IdxT>(
      {N * index.x, index.y, index.z}, in_shape, in_strides, ndim);
  auto xshape = in_shape[ndim - 1];
  IdxT xstride = in_strides[ndim - 1];
  IdxT out_idx = N * index.x + xshape * (index.y + IdxT(grid_dim.y) * index.z);
  for (int i = 0; i < N && (int(N * index.x) + i) < xshape; ++i) {
    out[out_idx++] = static_cast<U>(Op()(in[idx]));
    idx += xstride;
  }
}
// ===== end:   /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/unary.h =====
#line 7 "/Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/unary.metal"

#define instantiate_unary_work_per_thread(op, in_tname, out_tname, in_type, out_type) \
  instantiate_kernel("vn_" #op #in_tname #out_tname, unary_v, in_type, out_type, op)

#define instantiate_unary_base(op, in_tname, out_tname, in_type, out_type)             \
  instantiate_kernel("v_" #op #in_tname #out_tname, unary_v, in_type, out_type, op, 1) \
  instantiate_kernel("v2_" #op #in_tname #out_tname, unary_v2, in_type, out_type, op)  \
  instantiate_kernel(                                                                  \
      "gn1_" #op #in_tname #out_tname, unary_g, in_type, out_type, op, 1, int)         \
  instantiate_kernel(                                                                  \
      "gn4large_" #op #in_tname #out_tname, unary_g, in_type, out_type, op, 4)

#define instantiate_unary_all(op, in_tname, out_tname, in_type, out_type)       \
  instantiate_unary_base(op, in_tname, out_tname, in_type, out_type)            \
  instantiate_unary_work_per_thread(op, in_tname, out_tname, in_type, out_type)

#define instantiate_unary_all_same(op, tname, type)   \
  instantiate_unary_all(op, tname, tname, type, type)

#define instantiate_unary_base_same(op, tname, type)   \
  instantiate_unary_base(op, tname, tname, type, type)

#define instantiate_unary_float(op)                    \
  instantiate_unary_all_same(op, float16, half)        \
  instantiate_unary_all_same(op, float32, float)       \
  instantiate_unary_all_same(op, bfloat16, bfloat16_t)

#define instantiate_unary_int(op)                   \
  instantiate_unary_all_same(op, uint8, uint8_t)    \
  instantiate_unary_all_same(op, uint16, uint16_t)  \
  instantiate_unary_all_same(op, uint32, uint32_t)  \
  instantiate_unary_base_same(op, uint64, uint64_t) \
  instantiate_unary_all_same(op, int8, int8_t)      \
  instantiate_unary_all_same(op, int16, int16_t)    \
  instantiate_unary_all_same(op, int32, int32_t)    \
  instantiate_unary_base_same(op, int64, int64_t)

#define instantiate_unary_types(op)                \
  instantiate_unary_all_same(op, bool_, bool)      \
  instantiate_unary_int(op)                        \
  instantiate_unary_float(op)

instantiate_unary_types(Abs)
instantiate_unary_float(ArcCos)
instantiate_unary_float(ArcCosh)
instantiate_unary_float(ArcSin)
instantiate_unary_float(ArcSinh)
instantiate_unary_float(ArcTan)
instantiate_unary_float(ArcTanh)
instantiate_unary_types(Ceil)
instantiate_unary_float(Cos)
instantiate_unary_float(Cosh)
instantiate_unary_float(Exp)
instantiate_unary_float(Expm1)
instantiate_unary_types(Floor)
instantiate_unary_float(Log)
instantiate_unary_float(Log2)
instantiate_unary_float(Log10)
instantiate_unary_float(Log1p)
instantiate_unary_types(Negative)
instantiate_unary_float(Sigmoid)
instantiate_unary_float(Erf)
instantiate_unary_float(ErfInv)
instantiate_unary_types(Sign)
instantiate_unary_float(Sin)
instantiate_unary_float(Sinh)
instantiate_unary_types(Square)
instantiate_unary_float(Sqrt)
instantiate_unary_float(Rsqrt)
instantiate_unary_float(Tan)
instantiate_unary_float(Tanh)
instantiate_unary_float(Round)
instantiate_unary_int(BitwiseInvert)

instantiate_unary_base_same(Abs, complex64, complex64_t)
instantiate_unary_base_same(ArcCos, complex64, complex64_t)
instantiate_unary_base_same(ArcSin, complex64, complex64_t)
instantiate_unary_base_same(ArcTan, complex64, complex64_t)
instantiate_unary_base_same(Conjugate, complex64, complex64_t)
instantiate_unary_base_same(Cos, complex64, complex64_t)
instantiate_unary_base_same(Cosh, complex64, complex64_t)
instantiate_unary_base_same(Exp, complex64, complex64_t)
instantiate_unary_base_same(Log, complex64, complex64_t)
instantiate_unary_base_same(Log1p, complex64, complex64_t)
instantiate_unary_base_same(Log2, complex64, complex64_t)
instantiate_unary_base_same(Log10, complex64, complex64_t)
instantiate_unary_base_same(Negative, complex64, complex64_t)
instantiate_unary_base_same(Sign, complex64, complex64_t)
instantiate_unary_base_same(Sin, complex64, complex64_t)
instantiate_unary_base_same(Sinh, complex64, complex64_t)
instantiate_unary_base_same(Square, complex64, complex64_t)
instantiate_unary_base_same(Sqrt, complex64, complex64_t)
instantiate_unary_base_same(Rsqrt, complex64, complex64_t)
instantiate_unary_base_same(Tan, complex64, complex64_t)
instantiate_unary_base_same(Tanh, complex64, complex64_t)
instantiate_unary_base_same(Round, complex64, complex64_t)
instantiate_unary_base(Real, complex64, float32, complex64_t, float)
instantiate_unary_base(Imag, complex64, float32, complex64_t, float)

instantiate_unary_all_same(LogicalNot, bool_, bool)

instantiate_unary_all(ToFP8, float16, uint8, float16_t, uint8_t)
instantiate_unary_all(ToFP8, bfloat16, uint8, bfloat16_t, uint8_t)
instantiate_unary_all(ToFP8, float32, uint8, float, uint8_t)
instantiate_unary_all(FromFP8, uint8, float16, uint8_t, float16_t)
instantiate_unary_all(FromFP8, uint8, bfloat16, uint8_t, bfloat16_t)
instantiate_unary_all(FromFP8, uint8, float32, uint8_t, float)

    // clang-format on
// ===== end:   /Users/gg/Documents/GitHub/mlx/mlx/backend/metal/kernels/unary.metal =====
