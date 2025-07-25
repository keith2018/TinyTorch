/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include "Utils/Macros.h"

namespace tinytorch {

enum class DType : int8_t {
  Float32 = 0,
  Float16 = 1,
  BFloat16 = 2,
  Int32 = 3,
  Int64 = 4,
  Bool = 5,
  DTypeCount
};

inline size_t dtypeSize(DType t) {
  switch (t) {
    case DType::Float32:
      return 4;
    case DType::Float16:
    case DType::BFloat16:
      return 2;
    case DType::Int32:
      return 4;
    case DType::Int64:
      return 8;
    case DType::Bool:
      return 1;
    default:
      return 0;
  }
}

template <DType>
struct DTypeToType {
  using type = void;
};

template <>
struct DTypeToType<DType::Float32> {
  using type = float;
};
template <>
struct DTypeToType<DType::Float16> {
  using type = uint16_t;
};
template <>
struct DTypeToType<DType::BFloat16> {
  using type = uint16_t;
};
template <>
struct DTypeToType<DType::Int32> {
  using type = int32_t;
};
template <>
struct DTypeToType<DType::Int64> {
  using type = int64_t;
};
template <>
struct DTypeToType<DType::Bool> {
  using type = uint8_t;
};

template <DType dtype>
using DTypeToType_t = typename DTypeToType<dtype>::type;

template <typename T>
struct TypeToDType;

template <>
struct TypeToDType<float> {
  static constexpr DType value = DType::Float32;
};

template <>
struct TypeToDType<int32_t> {
  static constexpr DType value = DType::Int32;
};

template <>
struct TypeToDType<int64_t> {
  static constexpr DType value = DType::Int64;
};

template <>
struct TypeToDType<uint8_t> {
  static constexpr DType value = DType::Bool;
};

template <typename T>
constexpr DType TypeToDType_v = TypeToDType<T>::value;

template <typename T>
void CheckDTypeMatch(DType dtype) {
  if (dtype == DType::Float32) {
    ASSERT((std::is_same_v<T, float>) && "Type mismatch: expected float");
  } else if (dtype == DType::Float16 || dtype == DType::BFloat16) {
    ASSERT((std::is_same_v<T, uint16_t>) && "Type mismatch: expected uint16_t for Float16/BFloat16");
  } else if (dtype == DType::Int32) {
    ASSERT((std::is_same_v<T, int32_t>) && "Type mismatch: expected int32_t");
  } else if (dtype == DType::Int64) {
    ASSERT((std::is_same_v<T, int64_t>) && "Type mismatch: expected int64_t");
  } else if (dtype == DType::Bool) {
    ASSERT((std::is_same_v<T, uint8_t>) && "Type mismatch: expected uint8_t");
  } else {
    ASSERT(false && "Unknown DType");
  }
}

template <typename T>
using Array1d = std::vector<T>;

template <typename T>
using Array2d = std::vector<std::vector<T>>;

template <typename T>
using Array3d = std::vector<std::vector<std::vector<T>>>;

template <typename T>
Array1d<T> flatten2d(const Array2d<T>& arr2d) {
  Array1d<T> result;
  for (const auto& row : arr2d) {
    result.insert(result.end(), row.begin(), row.end());
  }
  return result;
}

template <typename T>
Array1d<T> flatten3d(const Array3d<T>& arr3d) {
  Array1d<T> result;
  for (const auto& mat : arr3d) {
    for (const auto& row : mat) {
      result.insert(result.end(), row.begin(), row.end());
    }
  }
  return result;
}

constexpr size_t MAX_TENSOR_DIM = 8;  // max tensor dimensions

template <typename T>
struct ALIGN(16) DimArray {
  T data[MAX_TENSOR_DIM];
};

struct ALIGN(16) Dim2D {
  // NOLINTNEXTLINE(google-explicit-constructor)
  Dim2D(int64_t n) : h(n), w(n) {}
  Dim2D(int64_t h, int64_t w) : h(h), w(w) {}

  int64_t h = 0;
  int64_t w = 0;
};

namespace dtype {

static std::string dtypeNames[] = {"Float32", "Float16", "BFloat16", "Int32", "Int64", "Bool"};
inline std::string toString(DType type) { return dtypeNames[static_cast<size_t>(type)]; }

}  // namespace dtype
}  // namespace tinytorch
