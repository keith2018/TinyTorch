/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <cassert>
#include <list>
#include <memory>
#include <unordered_map>

#include "Logger.h"

namespace TinyTorch {

enum class Device;

#define ALLOC_ROUND(x) (((x) + 511) & ~511)  // 512 bytes
#define ALLOC_MAX_CACHE (512 * 1024 * 1024)  /* 512 MB */

class Allocator {
 public:
  virtual ~Allocator() = default;

  virtual Device device() = 0;
  virtual void allocate(void** ptr, size_t size) = 0;
  virtual void deallocate(void* ptr) = 0;
  virtual void clear() {}
};

template <typename T>
class CachedAllocator : public Allocator {
 public:
  explicit CachedAllocator(uint64_t maxCacheSize = ALLOC_MAX_CACHE)
      : cacheEnabled_(maxCacheSize > 0),
        maxCacheSize_(maxCacheSize),
        currentCacheSize_(0) {}

  ~CachedAllocator() override { CachedAllocator::clear(); }

  Device device() override { return base_.device(); }
  void allocate(void** ptr, size_t size) override {
    size = ALLOC_ROUND(size);
    if (!cacheEnabled_) {
      base_.allocate(ptr, size);
      return;
    }
    auto it = freedList_.find(size);
    if (it != freedList_.end() && !it->second.empty()) {
      *ptr = it->second.front();
      it->second.pop_front();
      allocatedList_[*ptr] = size;
      currentCacheSize_ -= size;
      return;
    }

    base_.allocate(ptr, size);
    if (*ptr) {
      allocatedList_[*ptr] = size;
    } else {
      LOGE("base_.allocate failed with size: %lld", size);
    }
  }
  void deallocate(void* ptr) override {
    if (!cacheEnabled_) {
      base_.deallocate(ptr);
      return;
    }
    auto it = allocatedList_.find(ptr);
    if (it != allocatedList_.end()) {
      size_t size = it->second;
      allocatedList_.erase(it);

      if (currentCacheSize_ + size > maxCacheSize_) {
        base_.deallocate(ptr);
      } else {
        freedList_[size].push_back(ptr);
        currentCacheSize_ += size;
      }
    } else {
      LOGE("error: ptr not valid: %p", ptr);
    }
  }
  void clear() override {
    assert(allocatedList_.empty());
    for (auto& pair : freedList_) {
      for (void* ptr : pair.second) {
        base_.deallocate(ptr);
      }
    }
    freedList_.clear();
  }

 private:
  T base_;
  bool cacheEnabled_;
  uint64_t maxCacheSize_;
  uint64_t currentCacheSize_;
  std::unordered_map<void*, size_t> allocatedList_;
  std::unordered_map<size_t, std::list<void*>> freedList_;
};

}  // namespace TinyTorch
