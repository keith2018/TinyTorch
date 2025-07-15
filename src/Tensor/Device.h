/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace tinytorch {

enum class DeviceType : int8_t {
  CPU = 0,
  CUDA = 1,
  DeviceTypeCount
};

using DeviceIndex = int8_t;

struct Device {
  DeviceType type;
  DeviceIndex index;

  // NOLINTNEXTLINE(google-explicit-constructor)
  Device(DeviceType t, DeviceIndex i = -1) : type(t), index(i) {}

  bool operator==(const Device &other) const { return type == other.type && index == other.index; }

  bool isCpu() const { return type == DeviceType::CPU; }
  bool isCuda() const { return type == DeviceType::CUDA; }
};

namespace device {

static std::string deviceTypeNames[] = {"CPU", "CUDA"};
inline std::string toString(DeviceType type) { return deviceTypeNames[static_cast<size_t>(type)]; }

}  // namespace device

}  // namespace tinytorch
