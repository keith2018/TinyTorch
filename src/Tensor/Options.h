/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "DType.h"
#include "Device.h"

namespace tinytorch {

namespace options {

struct Default {
  static Device& device() {
    static Device defaultDevice(DeviceType::CPU);
    return defaultDevice;
  }
  static DType& dType() {
    static DType defaultDType = DType::Float32;
    return defaultDType;
  }
};

}  // namespace options

[[maybe_unused]]
static void setDefaultDevice(Device device) {
  options::Default::device() = device;
}

[[maybe_unused]]
static void setDefaultDType(DType dtype) {
  options::Default::dType() = dtype;
}

struct Options {
  Device device_;
  DType dtype_;
  bool requiresGrad_;
  bool pinnedMemory_;

  // NOLINTNEXTLINE(google-explicit-constructor)
  Options(Device device = options::Default::device(), DType dtype = options::Default::dType(),
          bool requiresGrad = false, bool pinnedMemory = false)
      : device_(device), dtype_(dtype), requiresGrad_(requiresGrad), pinnedMemory_(pinnedMemory) {}

  Options& device(const Device& d) {
    device_ = d;
    return *this;
  }

  Options& dtype(DType t) {
    dtype_ = t;
    return *this;
  }

  Options& requiresGrad(bool rg = true) {
    requiresGrad_ = rg;
    return *this;
  }

  Options& pinnedMemory(bool pm = true) {
    pinnedMemory_ = pm;
    return *this;
  }

  Options noGrad() const {
    Options ret = *this;
    ret.requiresGrad_ = false;
    return ret;
  }

  Options indices() const {
    Options ret = *this;
    ret.dtype_ = DType::Int64;
    return ret;
  }
};

namespace options {

inline Options device(DeviceType type, DeviceIndex index = 0) {
  Options opts;
  opts.device_ = Device(type, index);
  return opts;
}

inline Options device(const Device& d) {
  Options opts;
  opts.device_ = d;
  return opts;
}

inline Options dtype(DType dt) {
  Options opts;
  opts.dtype_ = dt;
  return opts;
}

inline Options requiresGrad(bool rg = true) {
  Options opts;
  opts.requiresGrad_ = rg;
  return opts;
}

inline Options pinnedMemory(bool pm = true) {
  Options opts;
  opts.pinnedMemory_ = pm;
  return opts;
}

}  // namespace options

}  // namespace tinytorch
