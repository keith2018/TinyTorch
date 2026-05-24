/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "CachedAllocator.h"

#include <array>
#include <set>
#include <unordered_map>

#include "Utils/Logger.h"

namespace tinytorch {

std::atomic<bool> CachedAllocator::cacheEnabled_{true};
std::atomic<int> CachedAllocator::nextPoolId_{0};

// Ref:
// https://github.com/pytorch/pytorch/blob/main/c10/cuda/CUDACachingAllocator.cpp

// clang-format off
constexpr size_t kMinBlockSize = 512;        // all sizes are rounded to at least 512 bytes
constexpr size_t kSmallSize = 1048576;       // largest "small" allocation is 1 MiB
constexpr size_t kSmallBuffer = 2097152;     // "small" allocations are packed in 2 MiB blocks
constexpr size_t kLargeBuffer = 20971520;    // "large" allocations may be packed in 20 MiB blocks
constexpr size_t kMinLargeAlloc = 10485760;  // allocations between 1 and 10 MiB may use kLargeBuffer
constexpr size_t kRoundLarge = 2097152;      // round up large allocations to 2 MiB
// clang-format on

struct Block;
typedef bool (*Comparison)(const Block*, const Block*);

struct BlockPool {
  BlockPool(Comparison comparator, bool small) : blocks(comparator), isSmall(small) {}
  std::set<Block*, Comparison> blocks;
  const bool isSmall;
};

struct Block {
  size_t size;      // block size in bytes
  BlockPool* pool;  // owning memory pool
  void* ptr;        // memory address
  bool allocated;   // in-use flag
  Block* prev;      // prev block if split from a larger allocation
  Block* next;      // next block if split from a larger allocation

  Block(size_t size, BlockPool* pool, void* ptr)
      : size(size), pool(pool), ptr(ptr), allocated(false), prev(nullptr), next(nullptr) {}

  // constructor for search key
  explicit Block(size_t size)
      : size(size), pool(nullptr), ptr(nullptr), allocated(false), prev(nullptr), next(nullptr) {}

  bool isSplit() const { return (prev != nullptr) || (next != nullptr); }
};

static bool BlockComparator(const Block* a, const Block* b) {
  if (a->size != b->size) {
    return a->size < b->size;
  }
  return reinterpret_cast<uintptr_t>(a->ptr) < reinterpret_cast<uintptr_t>(b->ptr);
}

struct AllocParams {
  Block searchKey;
  BlockPool* pool;
  size_t allocSize;
  Block* retBlock;

  AllocParams(size_t size, BlockPool* pool, size_t allocSize)
      : searchKey(size), pool(pool), allocSize(allocSize), retBlock(nullptr) {}
};

struct PoolState {
  BlockPool largeBlocks{BlockComparator, false};
  BlockPool smallBlocks{BlockComparator, true};
  int refCount{0};

  PoolState() = default;

  PoolState(PoolState&&) noexcept = default;
  PoolState& operator=(PoolState&&) noexcept = delete;

  PoolState(const PoolState&) = delete;
  PoolState& operator=(const PoolState&) = delete;
};

class CachedAllocatorImpl : public Allocator {
 public:
  explicit CachedAllocatorImpl(Allocator* base) : base_(base), totalAllocatedSize_(0), activePoolId_(-1) {}

  static size_t roundSize(size_t size) {
    if (size < kMinBlockSize) {
      return kMinBlockSize;
    }
    return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
  }

  static size_t getAllocationSize(size_t size) {
    if (size <= kSmallSize) {
      return kSmallBuffer;
    }
    if (size < kMinLargeAlloc) {
      return kLargeBuffer;
    }
    return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
  }

  static bool getFreeBlock(AllocParams& p) {
    BlockPool& pool = *p.pool;

    // set-container search, return minimum satisfied value.
    const auto it = pool.blocks.lower_bound(&p.searchKey);
    if (it == pool.blocks.end()) {
      return false;
    }
    p.retBlock = *it;
    pool.blocks.erase(it);
    return true;
  }

  static BlockPool& getPool(size_t size, PoolState& poolState) {
    if (size <= kSmallSize) {
      return poolState.smallBlocks;
    }
    return poolState.largeBlocks;
  }

  bool allocBlock(AllocParams& p) {
    size_t size = p.allocSize;
    void* ptr = base_->allocate(static_cast<int64_t>(size));
    if (!ptr) {
      return false;
    }
    totalAllocatedSize_ += size;
    p.retBlock = new Block(size, p.pool, ptr);
    ASSERT(p.retBlock != nullptr && p.retBlock->ptr != nullptr);
    return true;
  }

  static bool shouldSplit(const Block* block, size_t size) {
    size_t remaining = block->size - size;
    if (block->pool->isSmall) {
      return remaining >= kMinBlockSize;
    }
    return remaining > kSmallSize;
  }

  void releaseCachedBlocks(PoolState& poolState) {
    releaseBlocks(poolState.largeBlocks);
    releaseBlocks(poolState.smallBlocks);
  }

  void releaseBlock(Block* block) {
    base_->deallocate(block->ptr);
    totalAllocatedSize_ -= block->size;
    auto* pool = block->pool;
    pool->blocks.erase(block);
    delete block;
  }

  void releaseBlocks(BlockPool& pool) {
    // frees all non-split blocks
    auto it = pool.blocks.begin();
    while (it != pool.blocks.end()) {
      Block* block = *it;
      ++it;
      if (!block->prev && !block->next) {
        releaseBlock(block);
      }
    }
  }

  void releaseAllBlocks(BlockPool& pool) {
    auto it = pool.blocks.begin();
    while (it != pool.blocks.end()) {
      Block* block = *it;
      ++it;
      if (!block->prev) {
        base_->deallocate(block->ptr);
        totalAllocatedSize_ -= block->size;
      }
      pool.blocks.erase(block);
      delete block;
    }
  }

  Block* mallocImpl(size_t origSize, PoolState& poolState) {
    size_t size = roundSize(origSize);
    auto& pool = getPool(size, poolState);
    const size_t allocSize = getAllocationSize(size);
    AllocParams params(size, &pool, allocSize);

    bool blockFound = getFreeBlock(params);

    if (!blockFound) {
      blockFound = allocBlock(params);
      if (!blockFound) {
        // retry after release caches from the same pool
        releaseCachedBlocks(poolState);
        blockFound = allocBlock(params);
      }

      if (!blockFound) {
        // last resort: try releasing default pool caches
        if (activePoolId_ >= 0) {
          releaseCachedBlocks(defaultPool_);
          blockFound = allocBlock(params);
        }
      }

      if (!blockFound) {
        LOGE("Out of memory. failed to allocate size: %zu", allocSize);
        return nullptr;
      }
    }

    ASSERT(params.retBlock != nullptr && params.retBlock->ptr != nullptr);
    Block* block = params.retBlock;

    if (shouldSplit(block, size)) {
      Block* remaining = block;

      block = new Block(size, &pool, block->ptr);
      block->prev = remaining->prev;
      if (block->prev) {
        block->prev->next = block;
      }
      block->next = remaining;

      remaining->prev = block;
      remaining->ptr = static_cast<char*>(remaining->ptr) + size;
      remaining->size -= size;
      pool.blocks.insert(remaining);
    }

    block->allocated = true;
    return block;
  }

  static void freeBlock(Block* block) {
    ASSERT(!block->allocated);
    auto& pool = *block->pool;

    const std::array<Block*, 2> mergeCandidates = {block->prev, block->next};
    for (Block* candidate : mergeCandidates) {
      (void)tryMergeBlocks(block, candidate, pool);
    }
    pool.blocks.insert(block);
  }

  static size_t tryMergeBlocks(Block* dst, Block* src, BlockPool& pool) {
    if (!src || src->allocated) {
      return 0;
    }
    ASSERT(dst->isSplit() && src->isSplit());

    if (dst->prev == src) {
      // [src dst]
      dst->ptr = src->ptr;
      dst->prev = src->prev;
      if (dst->prev) {
        dst->prev->next = dst;
      }
    } else {
      // [dest src]
      dst->next = src->next;
      if (dst->next) {
        dst->next->prev = dst;
      }
    }
    const size_t subsumedSize = src->size;
    dst->size += subsumedSize;
    pool.blocks.erase(src);
    delete src;

    return subsumedSize;
  }

  void* allocate(int64_t nbytes) override {
    auto& poolState = getActivePoolState();
    Block* block = mallocImpl(static_cast<size_t>(nbytes), poolState);
    if (block) {
      activeBlocks_[block->ptr] = block;
      return block->ptr;
    }

    LOGE("allocate error, size: %lld", static_cast<long long>(nbytes));
    return nullptr;
  }

  void deallocate(void* ptr) override {
    auto it = activeBlocks_.find(ptr);
    if (it != activeBlocks_.end()) {
      Block* block = it->second;
      activeBlocks_.erase(it);

      block->allocated = false;
      freeBlock(block);

      if (!CachedAllocator::isCacheEnabled()) {
        if (!block->isSplit()) {
          releaseBlock(block);
        }
      }
    } else {
      LOGE("deallocate error, ptr not valid: %p", ptr);
    }
  }

  void beginAllocateToPool(int poolId) {
    ASSERT(activePoolId_ < 0 && "Nested pool allocation not supported");
    activePoolId_ = poolId;
    // lazy-create pool and increment ref count
    auto it = graphPools_.find(poolId);
    if (it == graphPools_.end()) {
      graphPools_.emplace(poolId, PoolState{});
      it = graphPools_.find(poolId);
    }
    it->second.refCount++;
  }

  void endAllocateToPool() {
    ASSERT(activePoolId_ >= 0 && "endAllocateToPool called without matching begin");
    auto it = graphPools_.find(activePoolId_);
    if (it != graphPools_.end()) {
      it->second.refCount--;
    }
    activePoolId_ = -1;
  }

  void freePool(int poolId) {
    auto it = graphPools_.find(poolId);
    if (it == graphPools_.end()) {
      return;
    }

    // do not free a pool that is still actively referenced
    if (it->second.refCount > 0) {
      LOGE("freePool warning: pool %d still has %d active references, forcing release", poolId, it->second.refCount);
    }

    // release all blocks in the pool back to the base allocator.
    releaseAllBlocks(it->second.largeBlocks);
    releaseAllBlocks(it->second.smallBlocks);
    graphPools_.erase(it);
  }

  int activePoolId() const { return activePoolId_; }

  ~CachedAllocatorImpl() override {
    // release all graph pools
    for (auto& [id, pool] : graphPools_) {
      releaseAllBlocks(pool.largeBlocks);
      releaseAllBlocks(pool.smallBlocks);
    }
    graphPools_.clear();

    // release any remaining active blocks
    for (auto& [ptr, block] : activeBlocks_) {
      base_->deallocate(block->ptr);
      delete block;
    }
    activeBlocks_.clear();

    // release default pool cached blocks
    releaseAllBlocks(defaultPool_.largeBlocks);
    releaseAllBlocks(defaultPool_.smallBlocks);
  }

 private:
  PoolState& getActivePoolState() {
    if (activePoolId_ < 0) {
      return defaultPool_;
    }
    return graphPools_[activePoolId_];
  }

  Allocator* base_;
  uint64_t totalAllocatedSize_;

  PoolState defaultPool_;

  std::unordered_map<int, PoolState> graphPools_;
  int activePoolId_;  // -1 = default pool

  ankerl::unordered_dense::map<void*, Block*> activeBlocks_;
};

CachedAllocator::CachedAllocator(std::unique_ptr<Allocator> base)
    : base_(std::move(base)), impl_(std::make_unique<CachedAllocatorImpl>(base_.get())) {}

CachedAllocator::~CachedAllocator() = default;

CachedAllocator::CachedAllocator(CachedAllocator&& other) noexcept
    : base_(std::move(other.base_)), impl_(std::move(other.impl_)) {}

CachedAllocator& CachedAllocator::operator=(CachedAllocator&& other) noexcept {
  if (this != &other) {
    impl_ = std::move(other.impl_);
    base_ = std::move(other.base_);
  }
  return *this;
}

void* CachedAllocator::allocate(int64_t nbytes) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return impl_->allocate(nbytes);
}

void CachedAllocator::deallocate(void* ptr) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  impl_->deallocate(ptr);
}

void CachedAllocator::beginAllocateToPool(int poolId) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
  static_cast<CachedAllocatorImpl*>(impl_.get())->beginAllocateToPool(poolId);
}

void CachedAllocator::endAllocateToPool() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
  static_cast<CachedAllocatorImpl*>(impl_.get())->endAllocateToPool();
}

void CachedAllocator::freePool(int poolId) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
  static_cast<CachedAllocatorImpl*>(impl_.get())->freePool(poolId);
}

int CachedAllocator::activePoolId() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
  return static_cast<CachedAllocatorImpl*>(impl_.get())->activePoolId();
}

int CachedAllocator::newPoolId() { return nextPoolId_.fetch_add(1, std::memory_order_relaxed); }

}  // namespace tinytorch
