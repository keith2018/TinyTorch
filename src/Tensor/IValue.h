/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <string>
#include <variant>

#include "Tensor.h"

namespace tinytorch {

class IValue {
 public:
  using ValueType = std::variant<bool,                //
                                 int8_t,              //
                                 uint16_t,            //
                                 int16_t,             //
                                 int32_t,             //
                                 int64_t,             //
                                 float,               //
                                 double,              //
                                 Scalar,              //
                                 Dim2D,               //
                                 Tensor,              //
                                 std::string,         //
                                 std::vector<IValue>  //
                                 >;

  IValue() = default;

  template <typename T, typename = std::enable_if_t<!std::is_same_v<std::decay_t<T>, IValue>>>
  IValue(T&& v) {  // NOLINT(google-explicit-constructor)
    if constexpr (std::is_enum_v<std::decay_t<T>>) {
      using Underlying = std::underlying_type_t<std::decay_t<T>>;
      value_ = static_cast<Underlying>(v);
    } else {
      value_ = std::forward<T>(v);
    }
  }

  bool toBool() const { return std::get<bool>(value_); }
  bool toInt8() const { return std::get<int8_t>(value_); }
  uint16_t toUint16() const { return std::get<uint16_t>(value_); }
  int16_t toInt16() const { return std::get<int16_t>(value_); }
  int32_t toInt32() const { return std::get<int32_t>(value_); }
  int64_t toInt64() const { return std::get<int64_t>(value_); }
  float toFloat() const { return std::get<float>(value_); }
  double toDouble() const { return std::get<double>(value_); }
  Scalar toScalar() const { return std::get<Scalar>(value_); }
  Dim2D toDim2D() const { return std::get<Dim2D>(value_); }
  Tensor toTensor() const { return std::get<Tensor>(value_); }
  const std::string& toString() const { return std::get<std::string>(value_); }
  const std::vector<IValue>& toList() const { return std::get<std::vector<IValue>>(value_); }

  template <typename Enum>
  Enum toEnum() const {
    static_assert(std::is_enum_v<Enum>, "toEnum only for enum types");
    using Underlying = std::underlying_type_t<Enum>;
    return static_cast<Enum>(std::get<Underlying>(value_));
  }

 private:
  ValueType value_;
};

}  // namespace tinytorch
