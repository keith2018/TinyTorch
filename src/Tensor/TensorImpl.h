/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Options.h"
#include "Storage.h"
#include "Utils/VectorUtils.h"

namespace tinytorch {

constexpr size_t SmallVectorInlineSize = 5;
using SizeVector = SmallVector<int64_t, SmallVectorInlineSize>;

class TensorImpl {
 public:
  TensorImpl(IntArrayView shape, Options options);
  TensorImpl(IntArrayView shape, Options options, const std::shared_ptr<Storage>& storage, int64_t offset);

  template <typename T>
  TensorImpl(const std::vector<T>& values, IntArrayView shape, Options options = {});

  TensorImpl() = default;
  ~TensorImpl() = default;

  TensorImpl(const TensorImpl&) = default;
  TensorImpl& operator=(const TensorImpl&) = default;
  TensorImpl(TensorImpl&&) noexcept = default;
  TensorImpl& operator=(TensorImpl&&) noexcept = default;

  DType dtype() const { return options_.dtype_; }
  Device device() const { return options_.device_; }
  bool pinnedMemory() const { return options_.pinnedMemory_; }
  bool requiresGrad() const { return options_.requiresGrad_; }

  int64_t dim() const { return static_cast<int64_t>(shape_.size()); }
  int64_t numel() const { return numel_; }
  int64_t storageOffset() const { return storageOffset_; }
  bool isScalar() const { return shape_.empty(); }

  template <typename T = void>
  T* dataPtr() {
    ensureStorage();
    return static_cast<T*>(dataPtr_);
  }

  template <typename T = void>
  const T* dataPtr() const {
    ensureStorage();
    return static_cast<T*>(dataPtr_);
  }

  Options& options() { return options_; }
  const Options& options() const { return options_; }
  IntArrayView shape() const { return shape_; }
  IntArrayView strides() const { return strides_; }
  int64_t shape(int64_t d) { return shape_[d < 0 ? (d + dim()) : d]; }
  int64_t stride(int64_t d) { return strides_[d < 0 ? (d + dim()) : d]; }
  const std::shared_ptr<Storage>& storage() const {
    ensureStorage();
    return storage_;
  }
  void setStorage(const std::shared_ptr<Storage>& storage, int64_t offset = 0);
  void copyOnWrite() const;

  void reshape_(IntArrayView shape);
  void flatten_(int64_t startDim = 0, int64_t endDim = -1);
  void unflatten_(int64_t dim, IntArrayView shape);
  void squeeze_(int64_t dim = -1);
  void squeeze_(IntArrayView dims);
  void unsqueeze_(int64_t dim);

  template <typename T>
  std::vector<T> toList() const;

  template <typename T>
  T item() const;

 private:
  void ensureStorage() const;
  static void computeStrides(SizeVector& strides, IntArrayView shape);
  static void computeNumel(int64_t& numel, IntArrayView shape);

  int64_t numel_ = 0;
  int64_t storageOffset_ = 0;  // bytes
  mutable void* dataPtr_ = nullptr;

  Options options_;
  SizeVector shape_;
  SizeVector strides_;
  mutable std::shared_ptr<Storage> storage_;
};

template <typename T>
TensorImpl::TensorImpl(const std::vector<T>& values, const IntArrayView shape, Options options)
    : TensorImpl(shape, options) {
  ASSERT(static_cast<int64_t>(values.size()) == numel());
  CheckDTypeMatch<T>(options.dtype_);
  Storage::copyOnDevice(dataPtr(), device(), values.data(), DeviceType::CPU, sizeof(T) * numel());
}

template <typename T>
std::vector<T> TensorImpl::toList() const {
  CheckDTypeMatch<T>(dtype());
  if (device().isCpu()) {
    const T* ptr = static_cast<const T*>(dataPtr());
    return {ptr, ptr + numel()};
  }

  std::vector<T> hostData(numel());
  Storage::copyOnDevice(hostData.data(), DeviceType::CPU, dataPtr(), device(), sizeof(T) * numel());
  return hostData;
}

template <typename T>
T TensorImpl::item() const {
  ASSERT(numel() == 1);
  CheckDTypeMatch<T>(dtype());

  if (device().isCpu()) {
    const T* ptr = static_cast<const T*>(dataPtr());
    return ptr[0];
  }

  T ret;
  Storage::copyOnDevice(&ret, DeviceType::CPU, dataPtr(), device(), sizeof(T));
  return ret;
}

}  // namespace tinytorch
