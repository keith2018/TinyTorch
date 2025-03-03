/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Allocator.h"

#include <cassert>

#include "Logger.h"

#define ALLOC_ROUND(x) (((x) + 511) & ~511)  // 512 bytes

namespace TinyTorch {

CachedAllocator::CachedAllocator(size_t maxCacheSize)
    : cacheEnabled_(maxCacheSize > 0),
      base_(nullptr),
      maxCacheSize_(maxCacheSize),
      currentCacheSize_(0) {}

CachedAllocator::~CachedAllocator() { CachedAllocator::clear(); }

void CachedAllocator::allocate(void** ptr, size_t size) {
  if (!cacheEnabled_) {
    base_->allocate(ptr, size);
    return;
  }
  size = ALLOC_ROUND(size);
  auto it = freedList_.find(size);
  if (it != freedList_.end() && !it->second.empty()) {
    *ptr = it->second.front();
    it->second.pop_front();
    allocatedList_[*ptr] = size;
    currentCacheSize_ -= size;
    return;
  }

  base_->allocate(ptr, size);
  if (*ptr) {
    allocatedList_[*ptr] = size;
  } else {
    LOGE("base_.allocate failed with size: %lld", size);
  }
}

void CachedAllocator::deallocate(void* ptr) {
  if (!cacheEnabled_) {
    base_->deallocate(ptr);
    return;
  }
  auto it = allocatedList_.find(ptr);
  if (it != allocatedList_.end()) {
    size_t size = it->second;
    freedList_[size].push_back(ptr);
    allocatedList_.erase(it);
    currentCacheSize_ += size;

    shrink();
  } else {
    LOGE("error: ptr not valid: %p", ptr);
  }
}

void CachedAllocator::clear() {
  assert(allocatedList_.empty());
  for (auto& pair : freedList_) {
    for (void* ptr : pair.second) {
      base_->deallocate(ptr);
    }
  }
  freedList_.clear();
}

void CachedAllocator::shrink() {
  while (!freedList_.empty() && currentCacheSize_ > maxCacheSize_) {
    auto it = freedList_.begin();
    if (!it->second.empty()) {
      void* ptr = it->second.front();
      it->second.pop_front();
      base_->deallocate(ptr);
      currentCacheSize_ -= it->first;

      if (it->second.empty()) {
        freedList_.erase(it);
      }
    }
  }
}

}  // namespace TinyTorch
