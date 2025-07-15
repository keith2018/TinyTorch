/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Allocator.h"

namespace tinytorch {

class CachedAllocator : public Allocator {
 public:
  explicit CachedAllocator(std::unique_ptr<Allocator> base);

  static void setCacheEnabled(bool enabled) { cacheEnabled_ = enabled; }

  void* allocate(int64_t nbytes) override {
    if (!cacheEnabled_) {
      return base_->allocate(nbytes);
    }
    return impl_->allocate(nbytes);
  }

  void deallocate(void* ptr) override {
    if (!cacheEnabled_) {
      base_->deallocate(ptr);
      return;
    }
    impl_->deallocate(ptr);
  }

 private:
  static bool cacheEnabled_;
  std::unique_ptr<Allocator> base_;
  std::unique_ptr<Allocator> impl_;
};

}  // namespace tinytorch
