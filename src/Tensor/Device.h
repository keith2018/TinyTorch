/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <cstdint>

namespace tinytorch {

enum class DeviceType : uint8_t {
  CPU = 0,
  CUDA = 1,
  DeviceTypeCount
};

inline const char* deviceTypeToString(DeviceType type) {
  switch (type) {
    case DeviceType::CPU:
      return "CPU";
    case DeviceType::CUDA:
      return "CUDA";
    case DeviceType::DeviceTypeCount:
      return "DeviceTypeCount";
    default:
      return "Unknown";
  }
}

using DeviceIndex = int8_t;

struct Device {
  DeviceType type;
  DeviceIndex index;

  // NOLINTNEXTLINE(google-explicit-constructor)
  Device(DeviceType t, DeviceIndex i = 0) : type(t), index(i) {}

  bool operator==(const Device& other) const { return type == other.type && index == other.index; }

  bool isCpu() const { return type == DeviceType::CPU; }
  bool isCuda() const { return type == DeviceType::CUDA; }

  static Device cpu() { return {DeviceType::CPU}; }
  static Device cuda(DeviceIndex i = 0) { return {DeviceType::CUDA, i}; }
};

}  // namespace tinytorch
