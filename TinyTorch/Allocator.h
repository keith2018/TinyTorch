/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <memory>

namespace tinytorch {

enum class Device;

class Allocator {
 public:
  virtual ~Allocator() = default;

  virtual Device device() = 0;
  virtual void allocate(void** ptr, size_t size) = 0;
  virtual void deallocate(void* ptr) = 0;
  virtual void clear() {}
};

class CachedAllocator : public Allocator {
 public:
  explicit CachedAllocator(std::unique_ptr<Allocator> base);

  ~CachedAllocator() override { impl_->clear(); }

  Device device() override { return base_->device(); }

  void allocate(void** ptr, size_t size) override {
    if (!cacheEnabled_) {
      base_->allocate(ptr, size);
      return;
    }
    impl_->allocate(ptr, size);
  }

  void deallocate(void* ptr) override {
    if (!cacheEnabled_) {
      base_->deallocate(ptr);
      return;
    }
    impl_->deallocate(ptr);
  }

  void clear() override { impl_->clear(); }

 private:
  bool cacheEnabled_;
  std::unique_ptr<Allocator> base_;
  std::unique_ptr<Allocator> impl_;
};

}  // namespace tinytorch
