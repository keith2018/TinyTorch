/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Allocator.h"

#include <cassert>
#include <cstdlib>

#include "Logger.h"

namespace TinyTorch {

CachedAllocator::CachedAllocator(size_t maxCacheSize)
    : maxCacheSize_(maxCacheSize), currentCacheSize_(0) {}

CachedAllocator::~CachedAllocator() {
  CachedAllocator::clear();
  for (auto& pair : allocatedList_) {
    std::free(pair.first);
  }
}

void* CachedAllocator::malloc(size_t size) {
  auto it = freedList_.find(size);
  if (it != freedList_.end() && !it->second.empty()) {
    void* ptr = it->second.front();
    it->second.pop_front();
    allocatedList_[ptr] = size;
    currentCacheSize_ -= size;
    return ptr;
  }

  void* ptr = std::malloc(size);
  if (ptr) {
    allocatedList_[ptr] = size;
  } else {
    LOGE("std::malloc failed with size: %lld", size);
  }
  return ptr;
}

void CachedAllocator::free(void* ptr) {
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
      std::free(ptr);
    }
  }
}

void CachedAllocator::shrink() {
  while (!freedList_.empty() && currentCacheSize_ > maxCacheSize_) {
    auto it = freedList_.begin();
    if (!it->second.empty()) {
      void* ptr = it->second.front();
      it->second.pop_front();
      std::free(ptr);
      currentCacheSize_ -= it->first;

      if (it->second.empty()) {
        freedList_.erase(it);
      }
    }
  }
}

}  // namespace TinyTorch
