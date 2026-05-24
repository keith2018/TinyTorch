/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <atomic>
#include <mutex>

#include "Allocator.h"

namespace tinytorch {

class CachedAllocator : public Allocator {
 public:
  explicit CachedAllocator(std::unique_ptr<Allocator> base);
  ~CachedAllocator() override;

  CachedAllocator(CachedAllocator&&) noexcept;
  CachedAllocator& operator=(CachedAllocator&&) noexcept;

  CachedAllocator(const CachedAllocator&) = delete;
  CachedAllocator& operator=(const CachedAllocator&) = delete;

  static void setCacheEnabled(bool enabled) { cacheEnabled_.store(enabled, std::memory_order_release); }
  static bool isCacheEnabled() { return cacheEnabled_.load(std::memory_order_acquire); }

  void* allocate(int64_t nbytes) override;
  void deallocate(void* ptr) override;

  void beginAllocateToPool(int poolId);
  void endAllocateToPool();

  void freePool(int poolId);

  int activePoolId() const;

  static int newPoolId();

 private:
  static std::atomic<bool> cacheEnabled_;
  static std::atomic<int> nextPoolId_;
  std::unique_ptr<Allocator> base_;
  std::unique_ptr<Allocator> impl_;
  mutable std::recursive_mutex mutex_;
};

}  // namespace tinytorch
