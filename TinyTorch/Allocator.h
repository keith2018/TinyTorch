/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <list>
#include <memory>
#include <unordered_map>

namespace TinyTorch {

#define DEFAULT_ALLOC_MAX_CACHE (512 * 1024 * 1024) /* 512 MB */

class Allocator {
 public:
  virtual ~Allocator() = default;

  virtual void allocate(void** ptr, size_t size) = 0;
  virtual void deallocate(void* ptr) = 0;
  virtual void clear() {}
};

class CachedAllocator : public Allocator {
 public:
  explicit CachedAllocator(uint64_t maxCacheSize = DEFAULT_ALLOC_MAX_CACHE);
  ~CachedAllocator() override;

  void setBaseAllocator(const std::shared_ptr<Allocator>& base) {
    base_ = base;
  }
  void allocate(void** ptr, size_t size) override;
  void deallocate(void* ptr) override;
  void clear() override;

 private:
  bool cacheEnabled_;
  std::shared_ptr<Allocator> base_;
  uint64_t maxCacheSize_;
  uint64_t currentCacheSize_;
  std::unordered_map<void*, size_t> allocatedList_;
  std::unordered_map<size_t, std::list<void*>> freedList_;
};

}  // namespace TinyTorch
