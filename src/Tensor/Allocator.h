/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <memory>

#include "Device.h"
#include "Utils/Macros.h"
#include "ankerl/unordered_dense.h"

namespace tinytorch {

constexpr size_t defaultAlignment = 32;

class Allocator {
 public:
  virtual ~Allocator() = default;

  virtual void* allocate(int64_t nbytes) = 0;
  virtual void deallocate(void* ptr) = 0;
};

template <bool aligned = true>
class CPUAllocator : public Allocator {
 public:
  explicit CPUAllocator(size_t alignment = defaultAlignment) : alignment_(alignment) {}

  ~CPUAllocator() override {
#ifndef NDEBUG
    ASSERT(allocatedPtrs_.empty() && "Memory leak detected in CPUAllocator!");
#endif
  }

  void* allocate(int64_t nbytes) override;
  void deallocate(void* ptr) override;

 private:
  size_t alignment_;
#ifndef NDEBUG
  ankerl::unordered_dense::set<void*> allocatedPtrs_;
#endif
};

template <bool aligned>
void* CPUAllocator<aligned>::allocate(int64_t nbytes) {
  void* ptr = nullptr;
  if (aligned) {
#if !defined(_MSC_VER)
    ptr = std::aligned_alloc(alignment_, nbytes);
#else
    ptr = _aligned_malloc(nbytes, alignment_);
#endif
  } else {
    ptr = std::malloc(nbytes);
  }

#ifndef NDEBUG
  if (ptr) {
    allocatedPtrs_.insert(ptr);
  }
#endif
  return ptr;
}

template <bool aligned>
void CPUAllocator<aligned>::deallocate(void* ptr) {
  if (aligned) {
#if !defined(_MSC_VER)
    std::free(ptr);
#else
    _aligned_free(ptr);
#endif
  } else {
    std::free(ptr);
  }

#ifndef NDEBUG
  if (ptr) {
    auto it = allocatedPtrs_.find(ptr);
    if (it != allocatedPtrs_.end()) {
      allocatedPtrs_.erase(it);
    } else {
      ASSERT(false && "Double free or invalid pointer in CPUAllocator!");
    }
  }
#endif
}

#ifdef USE_CUDA
class CPUPinnedAllocator : public Allocator {
 public:
  ~CPUPinnedAllocator() override;
  void* allocate(int64_t nbytes) override;
  void deallocate(void* ptr) override;

 private:
#ifndef NDEBUG
  ankerl::unordered_dense::set<void*> allocatedPtrs_;
#endif
};

class CUDAAllocator : public Allocator {
 public:
  explicit CUDAAllocator(DeviceIndex index = 0) : deviceIndex_(index) {}
  ~CUDAAllocator() override;
  void* allocate(int64_t nbytes) override;
  void deallocate(void* ptr) override;

 private:
  DeviceIndex deviceIndex_;
#ifndef NDEBUG
  ankerl::unordered_dense::set<void*> allocatedPtrs_;
#endif
};
#endif

struct Options;
Allocator* getAllocator(const Options& options);

}  // namespace tinytorch
