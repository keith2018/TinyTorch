/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Allocator.h"

#include <vector>

#include "CachedAllocator.h"
#include "Options.h"
#ifdef USE_CUDA
#include "Utils/CUDAUtils.h"
#endif
#include "Utils/Logger.h"

namespace tinytorch {

#ifdef USE_CUDA
CPUPinnedAllocator::~CPUPinnedAllocator() {
#ifndef NDEBUG
  ASSERT(allocatedPtrs_.empty() && "Memory leak detected in CPUPinnedAllocator!");
#endif
}

void* CPUPinnedAllocator::allocate(int64_t nbytes) {
  void* ptr = nullptr;
  CUDA_CHECK(cudaHostAlloc(&ptr, static_cast<size_t>(nbytes), cudaHostAllocDefault));
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
  cuda::CudaDeviceGuard guard(deviceIndex_);
  CUDA_CHECK(cudaMalloc(&ptr, static_cast<size_t>(nbytes)));
#ifndef NDEBUG
  if (ptr) {
    allocatedPtrs_.insert(ptr);
  }
#endif
  return ptr;
}

void CUDAAllocator::deallocate(void* ptr) {
  if (ptr) {
    cuda::CudaDeviceGuard guard(deviceIndex_);
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
#endif

Allocator* getAllocator(const Options& options) {
  if (options.device_.isCpu()) {
    if (options.pinnedMemory_) {
#ifdef USE_CUDA
      // leak-on-exit
      static auto* cachedPinnedAllocator = new CachedAllocator(std::make_unique<CPUPinnedAllocator>());
      return cachedPinnedAllocator;
#else
      ASSERT(false && "cuda not support");
      return nullptr;
#endif
    } else {
      static auto* cachedCpuAllocator = new CachedAllocator(std::make_unique<CPUAllocator<>>());
      return cachedCpuAllocator;
    }
  } else if (options.device_.isCuda()) {
#ifdef USE_CUDA
    return getCUDACachedAllocator(options.device_.index);
#else
    ASSERT(false && "cuda not support");
    return nullptr;
#endif
  }
  LOGE("getAllocator error: Unknown device type");
  return nullptr;
}

CachedAllocator* getCUDACachedAllocator(int device) {
#ifdef USE_CUDA
  auto deviceCount = cuda::getDeviceCount();
  // leak-on-exit
  static std::vector<CachedAllocator*>* deviceAllocators = nullptr;
  static std::once_flag flag;
  std::call_once(flag, [&]() {
    deviceAllocators = new std::vector<CachedAllocator*>();
    deviceAllocators->reserve(deviceCount);
    for (auto i = 0; i < deviceCount; i++) {
      deviceAllocators->push_back(new CachedAllocator(std::make_unique<CUDAAllocator>(i)));
    }
  });
  if (device < 0 || static_cast<size_t>(device) >= deviceAllocators->size()) {
    LOGE("getCUDACachedAllocator error: Invalid CUDA device index %d", device);
    return nullptr;
  }
  return (*deviceAllocators)[device];
#else
  (void)device;
  return nullptr;
#endif
}

}  // namespace tinytorch
