/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Storage.h"

#include "Utils/CUDAUtils.h"
#include "Utils/Logger.h"

namespace tinytorch {

Storage::Storage(int64_t nbytes, Device device, Allocator* allocator)
    : nbytes_(nbytes), device_(device), allocator_(allocator) {
  void* ptr = allocator_->allocate(nbytes_);
  data_ = std::unique_ptr<void, std::function<void(void*)>>(ptr, [allocator](void* p) { allocator->deallocate(p); });
}

std::shared_ptr<Storage> Storage::clone() const {
  auto newStorage = std::make_shared<Storage>(nbytes_, device_, allocator_);
  copyOnDevice(newStorage->data_.get(), data_.get(), device_, nbytes_);
  return newStorage;
}

void Storage::copyOnDevice(void* dst, const Device& dstDevice, const void* src, const Device& srcDevice,
                           int64_t nbytes) {
  if (nbytes == 0) {
    return;
  }

  // CPU -> CPU
  if (dstDevice.isCpu() && srcDevice.isCpu()) {
    std::memcpy(dst, src, nbytes);
    return;
  }

  // CUDA -> CUDA
  if (dstDevice.isCuda() && srcDevice.isCuda()) {
    cuda::CudaDeviceGuard guard(dstDevice.index);
    CUDA_CHECK(cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToDevice));
    return;
  }

  // CPU -> CUDA
  if (dstDevice.isCuda() && srcDevice.isCpu()) {
    cuda::CudaDeviceGuard guard(dstDevice.index);
    CUDA_CHECK(cudaMemcpy(dst, src, nbytes, cudaMemcpyHostToDevice));
    return;
  }

  // CUDA -> CPU
  if (dstDevice.isCpu() && srcDevice.isCuda()) {
    cuda::CudaDeviceGuard guard(dstDevice.index);
    CUDA_CHECK(cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToHost));
    return;
  }

  LOGE("Unknown device type in deviceCopy");
}

void Storage::copyOnDevice(void* dst, const void* src, const Device& device, int64_t nbytes) {
  copyOnDevice(dst, device, src, device, nbytes);
}

}  // namespace tinytorch
