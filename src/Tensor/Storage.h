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
  Storage(int64_t nbytes, Device device, Allocator* allocator = nullptr);
  ~Storage() = default;

  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;

  Storage(Storage&&) noexcept = default;
  Storage& operator=(Storage&&) noexcept = default;

  std::shared_ptr<Storage> clone() const;

  template <typename T = void>
  T* dataPtr() {
    return static_cast<T*>(data_.get());
  }

  int64_t size() const { return nbytes_; }
  Device device() const { return device_; }

  static void copyOnDevice(void* dst, const Device& dstDevice, const void* src, const Device& srcDevice,
                           int64_t nbytes);
  static void copyOnDevice(void* dst, const void* src, int64_t nbytes, const Device& device);

 private:
  int64_t nbytes_;
  Device device_;
  Allocator* allocator_;
  std::unique_ptr<void, std::function<void(void*)>> data_;
};

}  // namespace tinytorch
