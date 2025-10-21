/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <iostream>
#include <type_traits>
#include <utility>

#include "Tensor.h"
#include "Utils/VectorUtils.h"

namespace tinytorch {

using DispatchDevice = DeviceType;
using DispatchDType = DType;

struct DispatchKey {
  DispatchDevice device;
  DispatchDType dtype;

  constexpr bool operator==(const DispatchKey& o) const { return device == o.device && dtype == o.dtype; }

  std::string toString() const {
    return std::string("{ device: ") + deviceTypeToString(device) + ", dtype: " + dtypeToString(dtype) + "}";
  }
};

constexpr size_t dispatchKeyToIndex(DispatchKey k) {
  return static_cast<size_t>(k.device) * static_cast<size_t>(DispatchDType::DTypeCount) + static_cast<size_t>(k.dtype);
}
constexpr size_t NumDispatchKeys =
    static_cast<size_t>(DispatchDevice::DeviceTypeCount) * static_cast<size_t>(DispatchDType::DTypeCount);

template <typename Tag, typename Fn>
struct OpRegistry {
  static constexpr size_t N = NumDispatchKeys;
  static Fn table[N];

  static void registerImpl(DispatchKey key, Fn fn) { table[dispatchKeyToIndex(key)] = fn; }
  static Fn lookup(DispatchKey key) { return table[dispatchKeyToIndex(key)]; }
};

template <typename Tag, typename Fn>
Fn OpRegistry<Tag, Fn>::table[NumDispatchKeys] = {nullptr};

template <typename T>
constexpr bool isOptionValue = std::is_same_v<std::decay_t<T>, Options>;

template <typename T>
constexpr bool isTensorValue = std::is_same_v<std::decay_t<T>, Tensor>;

template <typename T>
constexpr bool isTensorArrayValue = std::is_same_v<std::decay_t<T>, ArrayView<Tensor>>;

template <typename First, typename... Rest>
DispatchKey getDispatchKeyFromArgs(First&& first, Rest&&... rest) {
  if constexpr (isOptionValue<First>) {
    return {first.device_.type, first.dtype_};
  } else if constexpr (isTensorValue<First>) {
    return {first.device().type, first.dtype()};
  } else if constexpr (isTensorArrayValue<std::decay_t<First>>) {
    ASSERT(!first.empty());
    const Tensor& t = first[0];
    ASSERT(t.defined());
    return {t.device().type, t.dtype()};
  } else {
    static_assert(sizeof...(Rest) > 0, "No Tensor argument found in getDispatchKeyFromArgs");
    return getDispatchKeyFromArgs(std::forward<Rest>(rest)...);
  }
}

#define DEFINE_OP(opname, fnType)                                                                                  \
  struct opname##Tag {};                                                                                           \
  using opname##Fn = fnType;                                                                                       \
  using opname##Registry = OpRegistry<opname##Tag, opname##Fn>;                                                    \
  template <typename... Args>                                                                                      \
  inline auto opname(Args&&... args) {                                                                             \
    DispatchKey key = getDispatchKeyFromArgs(std::forward<Args>(args)...);                                         \
    auto fn = opname##Registry::lookup(key);                                                                       \
    if (!fn) {                                                                                                     \
      std::cerr << "Error call op: " << #opname << ", no impl for dispatch key = " << key.toString() << std::endl; \
    }                                                                                                              \
    ASSERT(fn&& #opname);                                                                                          \
    return fn(std::forward<Args>(args)...);                                                                        \
  }                                                                                                                \
  struct opname##Register {                                                                                        \
    opname##Register(DispatchDevice dev, DispatchDType dt, opname##Fn fn) {                                        \
      opname##Registry::registerImpl({dev, dt}, fn);                                                               \
    }                                                                                                              \
  };

const std::vector<DispatchDevice> allDevices = {
    DispatchDevice::CPU,  //
    DispatchDevice::CUDA  //
};

const std::vector<DispatchDType> allDtypes = {
    DispatchDType::Float32,   //
    DispatchDType::Float16,   //
    DispatchDType::BFloat16,  //
    DispatchDType::Int32,     //
    DispatchDType::Int64,     //
    DispatchDType::Bool       //
};

template <typename Registry, typename Fn>
void RegisterOpImpl(const std::vector<DispatchDevice>& devices, const std::vector<DispatchDType>& dtypes, Fn fn) {
  for (auto device : devices) {
    for (auto dtype : dtypes) {
      Registry::registerImpl({device, dtype}, fn);
    }
  }
}

#define REGISTER_OP_IMPL(opname, device, dtype, fn) \
  RegisterOpImpl<opname##Registry>({DispatchDevice::device}, {DispatchDType::dtype}, fn);

#define REGISTER_OP_IMPL_MULTI(opname, devices, dtypes, fn) RegisterOpImpl<opname##Registry>(devices, dtypes, fn);

#define REGISTER_OP_IMPL_ALL_DTYPES(opname, device, fn) \
  RegisterOpImpl<opname##Registry>({DispatchDevice::device}, allDtypes, fn);

#define REGISTER_OP_IMPL_ALL_DEVICES(opname, dtype, fn) \
  RegisterOpImpl<opname##Registry>(allDevices, {DispatchDType::dtype}, fn);

#define REGISTER_OP_IMPL_ALL(opname, fn) RegisterOpImpl<opname##Registry>(allDevices, allDtypes, fn);

#define REGISTER_OP_IMPL_DTYPE_TPL(opname, device, fn)                               \
  REGISTER_OP_IMPL(opname, device, Float32, &(fn<DTypeToType_t<DType::Float32>>));   \
  REGISTER_OP_IMPL(opname, device, Float16, &(fn<DTypeToType_t<DType::Float16>>));   \
  REGISTER_OP_IMPL(opname, device, BFloat16, &(fn<DTypeToType_t<DType::BFloat16>>)); \
  REGISTER_OP_IMPL(opname, device, Int32, &(fn<DTypeToType_t<DType::Int32>>));       \
  REGISTER_OP_IMPL(opname, device, Int64, &(fn<DTypeToType_t<DType::Int64>>));       \
  REGISTER_OP_IMPL(opname, device, Bool, &(fn<DTypeToType_t<DType::Bool>>));

#define REGISTER_OP_IMPL_ALL_DEVICES_DTYPE_TPL(opname, fn) \
  REGISTER_OP_IMPL_DTYPE_TPL(opname, CPU, fn)              \
  REGISTER_OP_IMPL_DTYPE_TPL(opname, CUDA, fn)

}  // namespace tinytorch
