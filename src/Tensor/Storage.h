/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <functional>
#include <memory>

#include "Allocator.h"
#include "Device.h"

namespace tinytorch {

class Storage {
 public:
  Storage(int64_t nbytes, Device device, Allocator* allocator);
  ~Storage() = default;

  std::shared_ptr<Storage> clone() const;

  void* data() const { return data_.get(); }
  int64_t size() const { return nbytes_; }
  Device device() const { return device_; }

  static void copyOnDevice(void* dst, const Device& dstDevice, const void* src, const Device& srcDevice,
                           int64_t nbytes);
  static void copyOnDevice(void* dst, const void* src, const Device& device, int64_t nbytes);

 private:
  std::unique_ptr<void, std::function<void(void*)>> data_;
  int64_t nbytes_;
  Device device_;
  Allocator* allocator_;
};

}  // namespace tinytorch
