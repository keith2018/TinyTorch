/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

// Ref: https://github.com/pytorch/pytorch/blob/main/torch/headeronly/util/BFloat16.h

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iosfwd>
#include <ostream>

#include "Macros.h"

#if defined(__CUDACC__)
#include <cuda_bf16.h>
#endif

namespace tinytorch {

struct alignas(2) BFloat16 {
  uint16_t x;

  BFloat16() = default;

  struct from_bits_t {};
  static constexpr HOST_DEVICE from_bits_t from_bits() { return {}; }

  constexpr HOST_DEVICE BFloat16(unsigned short bits, from_bits_t) : x(bits) {}

  // NOLINTNEXTLINE(google-explicit-constructor)
  inline HOST_DEVICE BFloat16(float value);

  // NOLINTNEXTLINE(google-explicit-constructor)
  inline HOST_DEVICE operator float() const;

#if defined(__CUDACC__)
  // NOLINTNEXTLINE(google-explicit-constructor)
  inline HOST_DEVICE BFloat16(const __nv_bfloat16& value);

  // NOLINTNEXTLINE(google-explicit-constructor)
  explicit inline HOST_DEVICE operator __nv_bfloat16() const;
#endif
};

inline std::ostream& operator<<(std::ostream& out, const BFloat16& value) {
  out << (float)value;
  return out;
}

namespace detail {
inline HOST_DEVICE float f32_from_bits(uint16_t src) {
  float res = 0;
  uint32_t tmp = src;
  tmp <<= 16;

  std::memcpy(&res, &tmp, sizeof(tmp));
  return res;
}

inline HOST_DEVICE uint16_t bits_from_f32(float src) {
  uint32_t res = 0;

  std::memcpy(&res, &src, sizeof(res));
  return res >> 16;
}

inline HOST_DEVICE uint16_t round_to_nearest_even(float src) {
#if defined(_MSC_VER)
  if (isnan(src)) {
#else
  if (std::isnan(src)) {
#endif
    return UINT16_C(0x7FC0);
  } else {
    uint32_t U32;
    std::memcpy(&U32, &src, sizeof(U32));
    uint32_t rounding_bias = ((U32 >> 16) & 1) + UINT32_C(0x7FFF);
    return static_cast<uint16_t>((U32 + rounding_bias) >> 16);
  }
}

}  // namespace detail

/// Constructors
inline HOST_DEVICE BFloat16::BFloat16(float value)
    :
#if defined(__CUDACC__) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
      x(__bfloat16_as_ushort(__float2bfloat16(value)))
#else
      // RNE by default
      x(detail::round_to_nearest_even(value))
#endif
{
}

/// Implicit conversions
inline HOST_DEVICE BFloat16::operator float() const {
#if defined(__CUDACC__)
  return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&x));
#else
  return detail::f32_from_bits(x);
#endif
}

#if defined(__CUDACC__)
inline HOST_DEVICE BFloat16::BFloat16(const __nv_bfloat16& value) {
  x = *reinterpret_cast<const unsigned short*>(&value);
}
inline HOST_DEVICE BFloat16::operator __nv_bfloat16() const { return *reinterpret_cast<const __nv_bfloat16*>(&x); }
#endif

// CUDA intrinsics

#if defined(__CUDACC__)
inline __device__ BFloat16 __ldg(const BFloat16* ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __ldg(reinterpret_cast<const __nv_bfloat16*>(ptr));
#else
  return *ptr;
#endif
}
#endif

/// Arithmetic

inline HOST_DEVICE BFloat16 operator+(const BFloat16& a, const BFloat16& b) {
  return static_cast<float>(a) + static_cast<float>(b);
}

inline HOST_DEVICE BFloat16 operator-(const BFloat16& a, const BFloat16& b) {
  return static_cast<float>(a) - static_cast<float>(b);
}

inline HOST_DEVICE BFloat16 operator*(const BFloat16& a, const BFloat16& b) {
  return static_cast<float>(a) * static_cast<float>(b);
}

inline HOST_DEVICE BFloat16 operator/(const BFloat16& a, const BFloat16& b) {
  return static_cast<float>(a) / static_cast<float>(b);
}

inline HOST_DEVICE BFloat16 operator-(const BFloat16& a) { return -static_cast<float>(a); }

inline HOST_DEVICE BFloat16& operator+=(BFloat16& a, const BFloat16& b) {
  a = a + b;
  return a;
}

inline HOST_DEVICE BFloat16& operator-=(BFloat16& a, const BFloat16& b) {
  a = a - b;
  return a;
}

inline HOST_DEVICE BFloat16& operator*=(BFloat16& a, const BFloat16& b) {
  a = a * b;
  return a;
}

inline HOST_DEVICE BFloat16& operator/=(BFloat16& a, const BFloat16& b) {
  a = a / b;
  return a;
}

inline HOST_DEVICE BFloat16& operator|(BFloat16& a, const BFloat16& b) {
  a.x = a.x | b.x;
  return a;
}

inline HOST_DEVICE BFloat16& operator^(BFloat16& a, const BFloat16& b) {
  a.x = a.x ^ b.x;
  return a;
}

inline HOST_DEVICE BFloat16& operator&(BFloat16& a, const BFloat16& b) {
  a.x = a.x & b.x;
  return a;
}

/// Arithmetic with floats

inline HOST_DEVICE float operator+(BFloat16 a, float b) { return static_cast<float>(a) + b; }
inline HOST_DEVICE float operator-(BFloat16 a, float b) { return static_cast<float>(a) - b; }
inline HOST_DEVICE float operator*(BFloat16 a, float b) { return static_cast<float>(a) * b; }
inline HOST_DEVICE float operator/(BFloat16 a, float b) { return static_cast<float>(a) / b; }

inline HOST_DEVICE float operator+(float a, BFloat16 b) { return a + static_cast<float>(b); }
inline HOST_DEVICE float operator-(float a, BFloat16 b) { return a - static_cast<float>(b); }
inline HOST_DEVICE float operator*(float a, BFloat16 b) { return a * static_cast<float>(b); }
inline HOST_DEVICE float operator/(float a, BFloat16 b) { return a / static_cast<float>(b); }

inline HOST_DEVICE float& operator+=(float& a, const BFloat16& b) { return a += static_cast<float>(b); }
inline HOST_DEVICE float& operator-=(float& a, const BFloat16& b) { return a -= static_cast<float>(b); }
inline HOST_DEVICE float& operator*=(float& a, const BFloat16& b) { return a *= static_cast<float>(b); }
inline HOST_DEVICE float& operator/=(float& a, const BFloat16& b) { return a /= static_cast<float>(b); }

/// Arithmetic with doubles

inline HOST_DEVICE double operator+(BFloat16 a, double b) { return static_cast<double>(a) + b; }
inline HOST_DEVICE double operator-(BFloat16 a, double b) { return static_cast<double>(a) - b; }
inline HOST_DEVICE double operator*(BFloat16 a, double b) { return static_cast<double>(a) * b; }
inline HOST_DEVICE double operator/(BFloat16 a, double b) { return static_cast<double>(a) / b; }

inline HOST_DEVICE double operator+(double a, BFloat16 b) { return a + static_cast<double>(b); }
inline HOST_DEVICE double operator-(double a, BFloat16 b) { return a - static_cast<double>(b); }
inline HOST_DEVICE double operator*(double a, BFloat16 b) { return a * static_cast<double>(b); }
inline HOST_DEVICE double operator/(double a, BFloat16 b) { return a / static_cast<double>(b); }

/// Arithmetic with ints

inline HOST_DEVICE BFloat16 operator+(BFloat16 a, int b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return a + static_cast<BFloat16>(b);
}
inline HOST_DEVICE BFloat16 operator-(BFloat16 a, int b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return a - static_cast<BFloat16>(b);
}
inline HOST_DEVICE BFloat16 operator*(BFloat16 a, int b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return a * static_cast<BFloat16>(b);
}
inline HOST_DEVICE BFloat16 operator/(BFloat16 a, int b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return a / static_cast<BFloat16>(b);
}

inline HOST_DEVICE BFloat16 operator+(int a, BFloat16 b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return static_cast<BFloat16>(a) + b;
}
inline HOST_DEVICE BFloat16 operator-(int a, BFloat16 b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return static_cast<BFloat16>(a) - b;
}
inline HOST_DEVICE BFloat16 operator*(int a, BFloat16 b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return static_cast<BFloat16>(a) * b;
}
inline HOST_DEVICE BFloat16 operator/(int a, BFloat16 b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return static_cast<BFloat16>(a) / b;
}

//// Arithmetic with int64_t

inline HOST_DEVICE BFloat16 operator+(BFloat16 a, int64_t b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return a + static_cast<BFloat16>(b);
}
inline HOST_DEVICE BFloat16 operator-(BFloat16 a, int64_t b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return a - static_cast<BFloat16>(b);
}
inline HOST_DEVICE BFloat16 operator*(BFloat16 a, int64_t b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return a * static_cast<BFloat16>(b);
}
inline HOST_DEVICE BFloat16 operator/(BFloat16 a, int64_t b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return a / static_cast<BFloat16>(b);
}

inline HOST_DEVICE BFloat16 operator+(int64_t a, BFloat16 b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return static_cast<BFloat16>(a) + b;
}
inline HOST_DEVICE BFloat16 operator-(int64_t a, BFloat16 b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return static_cast<BFloat16>(a) - b;
}
inline HOST_DEVICE BFloat16 operator*(int64_t a, BFloat16 b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return static_cast<BFloat16>(a) * b;
}
inline HOST_DEVICE BFloat16 operator/(int64_t a, BFloat16 b) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return static_cast<BFloat16>(a) / b;
}

// Compare

inline HOST_DEVICE bool operator>(BFloat16& lhs, BFloat16& rhs) { return float(lhs) > float(rhs); }

inline HOST_DEVICE bool operator<(BFloat16& lhs, BFloat16& rhs) { return float(lhs) < float(rhs); }

inline HOST_DEVICE bool operator>=(BFloat16& lhs, BFloat16& rhs) { return float(lhs) >= float(rhs); }

inline HOST_DEVICE bool operator<=(BFloat16& lhs, BFloat16& rhs) { return float(lhs) <= float(rhs); }

inline HOST_DEVICE bool operator==(BFloat16& lhs, BFloat16& rhs) { return lhs.x == rhs.x; }

inline HOST_DEVICE bool operator!=(BFloat16& lhs, BFloat16& rhs) { return lhs.x != rhs.x; }

}  // namespace tinytorch

namespace std {

template <>
struct numeric_limits<tinytorch::BFloat16> {
  static constexpr bool is_signed = true;
  static constexpr bool is_specialized = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = true;
  static constexpr auto has_denorm = numeric_limits<float>::has_denorm;
  static constexpr auto has_denorm_loss = numeric_limits<float>::has_denorm_loss;
  static constexpr auto round_style = numeric_limits<float>::round_style;
  static constexpr bool is_iec559 = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 8;
  static constexpr int digits10 = 2;
  static constexpr int max_digits10 = 4;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -125;
  static constexpr int min_exponent10 = -37;
  static constexpr int max_exponent = 128;
  static constexpr int max_exponent10 = 38;
  static constexpr auto traps = numeric_limits<float>::traps;
  static constexpr auto tinyness_before = numeric_limits<float>::tinyness_before;

  static constexpr tinytorch::BFloat16 min() { return {0x0080, tinytorch::BFloat16::from_bits()}; }
  static constexpr tinytorch::BFloat16 lowest() { return {0xFF7F, tinytorch::BFloat16::from_bits()}; }
  static constexpr tinytorch::BFloat16 max() { return {0x7F7F, tinytorch::BFloat16::from_bits()}; }
  static constexpr tinytorch::BFloat16 epsilon() { return {0x3C00, tinytorch::BFloat16::from_bits()}; }
  static constexpr tinytorch::BFloat16 round_error() { return {0x3F00, tinytorch::BFloat16::from_bits()}; }
  static constexpr tinytorch::BFloat16 infinity() { return {0x7F80, tinytorch::BFloat16::from_bits()}; }
  static constexpr tinytorch::BFloat16 quiet_NaN() { return {0x7FC0, tinytorch::BFloat16::from_bits()}; }
  static constexpr tinytorch::BFloat16 signaling_NaN() { return {0x7F80, tinytorch::BFloat16::from_bits()}; }
  static constexpr tinytorch::BFloat16 denorm_min() { return {0x0001, tinytorch::BFloat16::from_bits()}; }
};

}  // namespace std