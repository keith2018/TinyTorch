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
                                 uint8_t,             //
                                 int8_t,              //
                                 uint16_t,            //
                                 int16_t,             //
                                 uint32_t,            //
                                 int32_t,             //
                                 uint64_t,            //
                                 int64_t,             //
                                 float,               //
                                 Half,                //
                                 BFloat16,            //
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
    using Decayed = std::decay_t<T>;

    // enum
    if constexpr (std::is_enum_v<Decayed>) {
      using Underlying = std::underlying_type_t<Decayed>;
      value_ = static_cast<Underlying>(v);
    }
    // const char* -> std::string
    else if constexpr (std::is_convertible_v<Decayed, std::string>) {
      value_ = std::string(std::forward<T>(v));
    }
    // char -> int8_t
    else if constexpr (std::is_same_v<Decayed, char>) {
      value_ = static_cast<int8_t>(v);
    } else {
      value_ = std::forward<T>(v);
    }
  }

  template <typename T>
  bool is() const {
    return std::holds_alternative<T>(value_);
  }

  bool toBool() const { return std::get<bool>(value_); }
  uint8_t toUint8() const { return std::get<uint8_t>(value_); }
  int8_t toInt8() const { return std::get<int8_t>(value_); }
  uint16_t toUint16() const { return std::get<uint16_t>(value_); }
  int16_t toInt16() const { return std::get<int16_t>(value_); }
  uint32_t toUint32() const { return std::get<uint32_t>(value_); }
  int32_t toInt32() const { return std::get<int32_t>(value_); }
  uint64_t toUint64() const { return std::get<uint64_t>(value_); }
  int64_t toInt64() const { return std::get<int64_t>(value_); }
  float toFloat() const { return std::get<float>(value_); }
  Half toHalf() const { return std::get<Half>(value_); }
  BFloat16 toBFloat16() const { return std::get<BFloat16>(value_); }
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
