/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <variant>

#include "DType.h"
#include "Utils/Macros.h"

namespace tinytorch {

class Scalar {
 public:
  using ValueType = std::variant<float, uint16_t, int32_t, int64_t, bool>;

  // NOLINTNEXTLINE(google-explicit-constructor)
  Scalar(float v) : dtype_(DType::Float32), value_(v) {}
  Scalar(uint16_t v, DType dtype) {
    ASSERT(dtype == DType::Float16 || dtype == DType::BFloat16);
    dtype_ = dtype;
    value_ = v;
  }
  // NOLINTNEXTLINE(google-explicit-constructor)
  Scalar(int32_t v) : dtype_(DType::Int32), value_(v) {}
  // NOLINTNEXTLINE(google-explicit-constructor)
  Scalar(int64_t v) : dtype_(DType::Int64), value_(v) {}
  // NOLINTNEXTLINE(google-explicit-constructor)
  Scalar(bool v) : dtype_(DType::Bool), value_(v) {}

  DType dtype() const { return dtype_; }

  template <typename T>
  T to() const {
    return std::visit(
        [](auto&& val) -> T {
          using U = std::decay_t<decltype(val)>;
          if constexpr (std::is_same_v<U, T>) {
            return val;
          } else {
            return static_cast<T>(val);
          }
        },
        value_);
  }

  template <DType dtype>
  DTypeToType_t<dtype> to() const {
    return to<DTypeToType_t<dtype>>();
  }

 private:
  DType dtype_;
  ValueType value_;
};

}  // namespace tinytorch
