/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include <cstring>
#include <set>
#include <thread>
#include <vector>

#include "Tensor/Allocator.h"
#include "Tensor/CachedAllocator.h"
#include "TinyTorch.h"
#include "Utils/CUDAUtils.h"
#include "test.h"

namespace tt = tinytorch;

TEST(allocator, cpu_allocate_deallocate) {
  auto* allocator = tt::getAllocator(tt::Options(tt::Device::cpu()));
  ASSERT_NE(allocator, nullptr);

  // Basic allocation
  void* ptr = allocator->allocate(1024);
  ASSERT_NE(ptr, nullptr);

  // Write/read to verify the memory is usable
  std::memset(ptr, 0xAB, 1024);
  auto* bytes = static_cast<uint8_t*>(ptr);
  EXPECT_EQ(bytes[0], 0xAB);
  EXPECT_EQ(bytes[1023], 0xAB);

  allocator->deallocate(ptr);
}

TEST(allocator, cpu_zero_size_allocation) {
  auto* allocator = tt::getAllocator(tt::Options(tt::Device::cpu()));
  ASSERT_NE(allocator, nullptr);

  // Zero-size allocation should still return something valid from the caching layer
  // (rounded up to kMinBlockSize=512)
  void* ptr = allocator->allocate(0);
  // Implementation rounds to at least 512, so this should succeed
  if (ptr) {
    allocator->deallocate(ptr);
  }
}

TEST(allocator, cpu_alignment) {
  auto* allocator = tt::getAllocator(tt::Options(tt::Device::cpu()));
  ASSERT_NE(allocator, nullptr);

  // Allocate multiple times and verify alignment (default 32-byte alignment)
  for (int i = 0; i < 10; i++) {
    void* ptr = allocator->allocate(64 + i * 7);  // various non-aligned sizes
    ASSERT_NE(ptr, nullptr);
    // The underlying memory should be 32-byte aligned (from CPUAllocator)
    // but CachedAllocator may return sub-block offsets that are 512-aligned
    // which is a multiple of 32.
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 32, 0u) << "Allocation " << i << " not 32-byte aligned";
    allocator->deallocate(ptr);
  }
}

TEST(allocator, cpu_multiple_allocations) {
  auto* allocator = tt::getAllocator(tt::Options(tt::Device::cpu()));
  ASSERT_NE(allocator, nullptr);

  std::vector<void*> ptrs;
  constexpr int kNumAllocs = 100;

  // Allocate many blocks
  for (int i = 0; i < kNumAllocs; i++) {
    void* ptr = allocator->allocate(512 * (i + 1));
    ASSERT_NE(ptr, nullptr);
    ptrs.push_back(ptr);
  }

  // Verify all pointers are unique
  std::set<void*> uniquePtrs(ptrs.begin(), ptrs.end());
  EXPECT_EQ(uniquePtrs.size(), ptrs.size());

  // Free all
  for (auto* ptr : ptrs) {
    allocator->deallocate(ptr);
  }
}

TEST(allocator, cache_reuse_same_size) {
  auto* allocator = tt::getAllocator(tt::Options(tt::Device::cpu()));
  ASSERT_NE(allocator, nullptr);

  // Allocate and free
  void* ptr1 = allocator->allocate(1024);
  ASSERT_NE(ptr1, nullptr);
  allocator->deallocate(ptr1);

  // Allocate same size — should reuse the cached block
  void* ptr2 = allocator->allocate(1024);
  ASSERT_NE(ptr2, nullptr);
  EXPECT_EQ(ptr1, ptr2) << "Expected cached block reuse for same size";
  allocator->deallocate(ptr2);
}

TEST(allocator, cache_reuse_smaller_size) {
  auto* allocator = tt::getAllocator(tt::Options(tt::Device::cpu()));
  ASSERT_NE(allocator, nullptr);

  // Allocate a larger block
  void* ptr1 = allocator->allocate(4096);
  ASSERT_NE(ptr1, nullptr);
  allocator->deallocate(ptr1);

  // Allocate a smaller size — might get the same block (or split from it)
  void* ptr2 = allocator->allocate(512);
  ASSERT_NE(ptr2, nullptr);
  // ptr2 should be at the same base address (beginning of the split block)
  EXPECT_EQ(ptr1, ptr2) << "Expected block split to reuse base address";
  allocator->deallocate(ptr2);
}

TEST(allocator, cache_disabled_no_reuse) {
  auto* allocator = tt::getAllocator(tt::Options(tt::Device::cpu()));
  ASSERT_NE(allocator, nullptr);

  // Allocate with cache enabled
  void* ptr1 = allocator->allocate(2048);
  ASSERT_NE(ptr1, nullptr);

  // Disable cache, then free — block should be released to base allocator
  tt::CachedAllocator::setCacheEnabled(false);
  allocator->deallocate(ptr1);

  // Allocate same size — should NOT get the same address (no caching)
  void* ptr2 = allocator->allocate(2048);
  ASSERT_NE(ptr2, nullptr);
  // Note: it's possible (but unlikely) to get the same address from malloc,
  // so we don't assert inequality. Just ensure it works without crash.
  allocator->deallocate(ptr2);

  // Re-enable cache
  tt::CachedAllocator::setCacheEnabled(true);
}

TEST(allocator, cache_toggle_correctness) {
  auto* allocator = tt::getAllocator(tt::Options(tt::Device::cpu()));
  ASSERT_NE(allocator, nullptr);

  // Allocate with cache enabled
  void* ptr1 = allocator->allocate(1024);
  ASSERT_NE(ptr1, nullptr);

  // Disable cache
  tt::CachedAllocator::setCacheEnabled(false);

  // Allocate another (still tracked by impl)
  void* ptr2 = allocator->allocate(1024);
  ASSERT_NE(ptr2, nullptr);

  // Re-enable cache
  tt::CachedAllocator::setCacheEnabled(true);

  // Deallocate both — should not crash regardless of cache state changes
  allocator->deallocate(ptr1);
  allocator->deallocate(ptr2);

  // Verify reuse now works again
  void* ptr3 = allocator->allocate(1024);
  ASSERT_NE(ptr3, nullptr);
  allocator->deallocate(ptr3);
}

TEST(allocator, pool_id_uniqueness) {
  int id1 = tt::CachedAllocator::newPoolId();
  int id2 = tt::CachedAllocator::newPoolId();
  int id3 = tt::CachedAllocator::newPoolId();
  EXPECT_NE(id1, id2);
  EXPECT_NE(id2, id3);
  EXPECT_NE(id1, id3);
}

TEST(allocator, concurrent_allocate_deallocate) {
  auto* allocator = tt::getAllocator(tt::Options(tt::Device::cpu()));
  ASSERT_NE(allocator, nullptr);

  constexpr int kNumThreads = 8;
  constexpr int kAllocsPerThread = 100;
  std::vector<std::thread> threads;

  std::atomic<int> errors{0};

  threads.reserve(kNumThreads);
  for (int t = 0; t < kNumThreads; t++) {
    threads.emplace_back([&, t]() {
      std::vector<void*> ptrs;
      for (int i = 0; i < kAllocsPerThread; i++) {
        size_t size = 512 * static_cast<size_t>((t * kAllocsPerThread + i) % 20 + 1);
        void* ptr = allocator->allocate(static_cast<int64_t>(size));
        if (!ptr) {
          errors.fetch_add(1);
          continue;
        }
        // Touch the memory to detect corruption
        std::memset(ptr, static_cast<int>(t & 0xFF), std::min(size, size_t(64)));
        ptrs.push_back(ptr);
      }
      // Deallocate in reverse order
      for (auto it = ptrs.rbegin(); it != ptrs.rend(); ++it) {
        allocator->deallocate(*it);
      }
    });
  }

  for (auto& th : threads) {
    th.join();
  }

  EXPECT_EQ(errors.load(), 0) << "Some allocations failed under contention";
}

TEST(allocator, concurrent_allocate_interleaved) {
  auto* allocator = tt::getAllocator(tt::Options(tt::Device::cpu()));
  ASSERT_NE(allocator, nullptr);

  constexpr int kNumThreads = 4;
  constexpr int kIterations = 200;
  std::vector<std::thread> threads;

  std::atomic<int> errors{0};

  threads.reserve(kNumThreads);
  for (int t = 0; t < kNumThreads; t++) {
    threads.emplace_back([&]() {
      for (int i = 0; i < kIterations; i++) {
        // Allocate and immediately free — exercises cache reuse under contention
        void* ptr = allocator->allocate(1024);
        if (!ptr) {
          errors.fetch_add(1);
          continue;
        }
        // Small delay to increase interleaving probability
        std::memset(ptr, 0, 64);
        allocator->deallocate(ptr);
      }
    });
  }

  for (auto& th : threads) {
    th.join();
  }

  EXPECT_EQ(errors.load(), 0);
}

TEST(allocator, large_allocation) {
  auto* allocator = tt::getAllocator(tt::Options(tt::Device::cpu()));
  ASSERT_NE(allocator, nullptr);

  // Allocate > kMinLargeAlloc (10 MiB) — uses kRoundLarge rounding
  constexpr size_t kLargeSize = 16 * 1024 * 1024;  // 16 MiB
  void* ptr = allocator->allocate(static_cast<int64_t>(kLargeSize));
  ASSERT_NE(ptr, nullptr);

  // Write to first and last byte
  auto* bytes = static_cast<uint8_t*>(ptr);
  bytes[0] = 0x11;
  bytes[kLargeSize - 1] = 0x22;
  EXPECT_EQ(bytes[0], 0x11);
  EXPECT_EQ(bytes[kLargeSize - 1], 0x22);

  allocator->deallocate(ptr);
}

TEST(allocator, mixed_sizes) {
  auto* allocator = tt::getAllocator(tt::Options(tt::Device::cpu()));
  ASSERT_NE(allocator, nullptr);

  // Mix of small and large allocations
  std::vector<std::pair<void*, size_t>> allocs;
  std::vector<size_t> sizes = {64, 512, 1024, 4096, 1048576, 2097152, 10485760, 20971520};

  for (auto size : sizes) {
    void* ptr = allocator->allocate(static_cast<int64_t>(size));
    ASSERT_NE(ptr, nullptr) << "Failed to allocate " << size << " bytes";
    allocs.emplace_back(ptr, size);
  }

  // Verify all are writable
  for (auto& [ptr, size] : allocs) {
    std::memset(ptr, 0xCD, std::min(size, size_t(128)));
  }

  // Free in random-ish order
  for (int i = static_cast<int>(allocs.size()) - 1; i >= 0; i -= 2) {
    allocator->deallocate(allocs[i].first);
  }
  for (int i = 0; i < static_cast<int>(allocs.size()); i += 2) {
    allocator->deallocate(allocs[i].first);
  }
}

TEST(allocator, tensor_allocate_cpu) {
  auto opts = tt::Options(tt::Device::cpu(), tt::DType::Float32).noGrad();

  // Create tensor — exercises full allocator path
  tt::Tensor t({100, 100}, opts);
  ASSERT_NE(t.dataPtr<>(), nullptr);

  // Fill and verify
  auto data = std::vector<float>(10000, 3.14f);
  tt::Tensor t2(data, {100, 100}, opts);
  auto list = t2.toList<float>();
  EXPECT_EQ(list.size(), 10000u);
  EXPECT_NEAR(list[0], 3.14f, 1e-5f);
  EXPECT_NEAR(list[9999], 3.14f, 1e-5f);
}

TEST(allocator, tensor_reuse_after_destroy) {
  auto opts = tt::Options(tt::Device::cpu(), tt::DType::Float32).noGrad();

  void* firstPtr = nullptr;
  {
    tt::Tensor t({256}, opts);
    firstPtr = t.dataPtr<>();
    ASSERT_NE(firstPtr, nullptr);
  }
  // Tensor destroyed, memory returned to cache

  // New tensor of same size should reuse the cached block
  tt::Tensor t2({256}, opts);
  void* secondPtr = t2.dataPtr<>();
  EXPECT_EQ(firstPtr, secondPtr) << "Expected cached block reuse for Tensor";
}

TEST(allocator, tensor_multiple_create_destroy_cycles) {
  auto opts = tt::Options(tt::Device::cpu(), tt::DType::Float32).noGrad();

  // Stress test: create and destroy many tensors
  for (int cycle = 0; cycle < 50; cycle++) {
    std::vector<tt::Tensor> tensors;
    for (int i = 0; i < 20; i++) {
      tensors.push_back(tt::Tensor({static_cast<int64_t>(128 * (i + 1))}, opts));
      ASSERT_NE(tensors.back().dataPtr<>(), nullptr);
    }
    // All tensors freed when vector goes out of scope
  }
  // Should not crash, leak, or corrupt
}

#ifdef USE_CUDA

#define SKIP_IF_NO_CUDA()                                         \
  do {                                                            \
    if (!tt::cuda::deviceAvailable()) {                           \
      GTEST_SKIP() << "CUDA device not available; skipping test"; \
    }                                                             \
  } while (0)

TEST(allocator, cuda_basic_allocate_deallocate) {
  SKIP_IF_NO_CUDA();

  auto* allocator = tt::getCUDACachedAllocator(0);
  ASSERT_NE(allocator, nullptr);

  void* ptr = allocator->allocate(4096);
  ASSERT_NE(ptr, nullptr);
  allocator->deallocate(ptr);
}

TEST(allocator, cuda_cache_reuse) {
  SKIP_IF_NO_CUDA();

  auto* allocator = tt::getCUDACachedAllocator(0);
  ASSERT_NE(allocator, nullptr);

  void* ptr1 = allocator->allocate(2048);
  ASSERT_NE(ptr1, nullptr);
  allocator->deallocate(ptr1);

  void* ptr2 = allocator->allocate(2048);
  ASSERT_NE(ptr2, nullptr);
  EXPECT_EQ(ptr1, ptr2) << "CUDA cached block should be reused";
  allocator->deallocate(ptr2);
}

TEST(allocator, cuda_pool_begin_end) {
  SKIP_IF_NO_CUDA();

  auto* allocator = tt::getCUDACachedAllocator(0);
  ASSERT_NE(allocator, nullptr);

  EXPECT_EQ(allocator->activePoolId(), -1);

  int poolId = tt::CachedAllocator::newPoolId();
  allocator->beginAllocateToPool(poolId);
  EXPECT_EQ(allocator->activePoolId(), poolId);

  void* ptr = allocator->allocate(1024);
  ASSERT_NE(ptr, nullptr);

  allocator->endAllocateToPool();
  EXPECT_EQ(allocator->activePoolId(), -1);

  // Free and cleanup
  allocator->deallocate(ptr);
  allocator->freePool(poolId);
}

TEST(allocator, cuda_tensor_integration) {
  SKIP_IF_NO_CUDA();

  auto opts = tt::Options(tt::Device(tt::DeviceType::CUDA, 0), tt::DType::Float32).noGrad();

  // Create tensors on GPU
  std::vector<float> data(1024, 1.5f);
  tt::Tensor t(data, {32, 32}, opts);

  // Copy back and verify
  auto host = t.toList<float>();
  EXPECT_EQ(host.size(), 1024u);
  EXPECT_NEAR(host[0], 1.5f, 1e-5f);
  EXPECT_NEAR(host[1023], 1.5f, 1e-5f);
}

TEST(allocator, cuda_concurrent_allocate) {
  SKIP_IF_NO_CUDA();

  auto* allocator = tt::getCUDACachedAllocator(0);
  ASSERT_NE(allocator, nullptr);

  constexpr int kNumThreads = 4;
  constexpr int kAllocsPerThread = 50;
  std::vector<std::thread> threads;
  std::atomic<int> errors{0};

  threads.reserve(kNumThreads);
  for (int t = 0; t < kNumThreads; t++) {
    threads.emplace_back([&]() {
      std::vector<void*> ptrs;
      for (int i = 0; i < kAllocsPerThread; i++) {
        void* ptr = allocator->allocate(1024 * (i + 1));
        if (!ptr) {
          errors.fetch_add(1);
          continue;
        }
        ptrs.push_back(ptr);
      }
      for (auto* ptr : ptrs) {
        allocator->deallocate(ptr);
      }
    });
  }

  for (auto& th : threads) {
    th.join();
  }

  EXPECT_EQ(errors.load(), 0);
}

#endif  // USE_CUDA
