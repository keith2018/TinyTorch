/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Allocator.h"

#include <vector>

#include "CachedAllocator.h"
#include "Options.h"
#include "Utils/CUDAUtils.h"
#include "Utils/Logger.h"

namespace tinytorch {

CPUPinnedAllocator::~CPUPinnedAllocator() {
#ifndef NDEBUG
  ASSERT(allocatedPtrs_.empty() && "Memory leak detected in CPUPinnedAllocator!");
#endif
}

void* CPUPinnedAllocator::allocate(int64_t nbytes) {
  void* ptr = nullptr;
  CUDA_CHECK(cudaHostAlloc(&ptr, nbytes, cudaHostAllocDefault));
#ifndef NDEBUG
  if (ptr) {
    allocatedPtrs_.insert(ptr);
  }
#endif
  return ptr;
}

void CPUPinnedAllocator::deallocate(void* ptr) {
  if (ptr) {
    CUDA_CHECK(cudaFreeHost(ptr));
  }
#ifndef NDEBUG
  if (ptr) {
    auto it = allocatedPtrs_.find(ptr);
    if (it != allocatedPtrs_.end()) {
      allocatedPtrs_.erase(it);
    } else {
      ASSERT(false && "Double free or invalid pointer in CPUPinnedAllocator!");
    }
  }
#endif
}

CUDAAllocator::~CUDAAllocator() {
#ifndef NDEBUG
  ASSERT(allocatedPtrs_.empty() && "Memory leak detected in CUDAAllocator!");
#endif
}

void* CUDAAllocator::allocate(int64_t nbytes) {
  void* ptr = nullptr;
  CUDA_CHECK(cudaSetDevice(deviceIndex_));
  CUDA_CHECK(cudaMalloc(&ptr, nbytes));
#ifndef NDEBUG
  if (ptr) {
    allocatedPtrs_.insert(ptr);
  }
#endif
  return ptr;
}

void CUDAAllocator::deallocate(void* ptr) {
  if (ptr) {
    CUDA_CHECK(cudaSetDevice(deviceIndex_));
    CUDA_CHECK(cudaFree(ptr));
  }
#ifndef NDEBUG
  if (ptr) {
    auto it = allocatedPtrs_.find(ptr);
    if (it != allocatedPtrs_.end()) {
      allocatedPtrs_.erase(it);
    } else {
      ASSERT(false && "Double free or invalid pointer in CUDAAllocator!");
    }
  }
#endif
}

Allocator* getAllocator(const Options& options) {
  if (options.device_.isCpu()) {
    if (options.pinnedMemory_) {
      static CachedAllocator cachedPinnedAllocator(std::make_unique<CPUPinnedAllocator>());
      return &cachedPinnedAllocator;
    } else {
      static CachedAllocator cachedCpuAllocator(std::make_unique<CPUAllocator<>>());
      return &cachedCpuAllocator;
    }
  } else if (options.device_.isCuda()) {
    auto deviceCount = cuda::getDeviceCount();
    static std::vector<CachedAllocator> deviceAllocators;
    static std::once_flag flag;
    std::call_once(flag, [&]() {
      deviceAllocators.reserve(deviceCount);
      for (auto i = 0; i < deviceCount; i++) {
        deviceAllocators.emplace_back(std::make_unique<CUDAAllocator>(i));
      }
    });
    auto deviceIndex = options.device_.index;
    if (deviceIndex < 0 || static_cast<size_t>(deviceIndex) >= deviceAllocators.size()) {
      LOGE("getAllocator error: Invalid CUDA device index %d", deviceIndex);
      return nullptr;
    }
    return &deviceAllocators[deviceIndex];
  }
  LOGE("getAllocator error: Unknown device type");
  return nullptr;
}

}  // namespace tinytorch
