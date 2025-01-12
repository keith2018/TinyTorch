/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <list>
#include <unordered_map>

#include "TensorImpl.h"

namespace TinyTorch {

class CachedAllocator : public Allocator {
 public:
  explicit CachedAllocator(size_t maxCacheSize = 256 * 1024 * 1024);  // 256 MB
  ~CachedAllocator() override;

  void* malloc(size_t size) override;
  void free(void* ptr) override;
  void clear() override;

 private:
  void shrink();

  size_t maxCacheSize_;
  size_t currentCacheSize_;
  std::unordered_map<void*, size_t> allocatedList_;
  std::unordered_map<size_t, std::list<void*>> freedList_;
};

}  // namespace TinyTorch
