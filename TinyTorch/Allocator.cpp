/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Allocator.h"

#include <array>
#include <cassert>
#include <memory>
#include <set>
#include <unordered_map>

#include "Logger.h"

namespace TinyTorch {

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
  BlockPool(Comparison comparator, bool small)
      : blocks(comparator), isSmall(small) {}
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
      : size(size),
        pool(pool),
        ptr(ptr),
        allocated(false),
        prev(nullptr),
        next(nullptr) {}

  // constructor for search key
  explicit Block(size_t size)
      : size(size),
        pool(nullptr),
        ptr(nullptr),
        allocated(false),
        prev(nullptr),
        next(nullptr) {}

  bool isSplit() const { return (prev != nullptr) || (next != nullptr); }
};

static bool BlockComparator(const Block* a, const Block* b) {
  if (a->size != b->size) {
    return a->size < b->size;
  }
  return reinterpret_cast<uintptr_t>(a->ptr) <
         reinterpret_cast<uintptr_t>(b->ptr);
}

struct AllocParams {
  Block searchKey;
  BlockPool* pool;
  size_t allocSize;
  Block* retBlock;

  AllocParams(size_t size, BlockPool* pool, size_t allocSize)
      : searchKey(size), pool(pool), allocSize(allocSize), retBlock(nullptr) {}

  size_t size() const { return searchKey.size; }
};

class CachedAllocatorImpl : public Allocator {
 public:
  explicit CachedAllocatorImpl(Allocator* base)
      : base_(base),
        totalAllocatedSize_(0),
        largeBlocks(BlockComparator, false),
        smallBlocks(BlockComparator, true) {}

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

    // set-container search, return minium satisfied value.
    const auto it = pool.blocks.lower_bound(&p.searchKey);
    if (it == pool.blocks.end()) {
      return false;
    }
    p.retBlock = *it;
    pool.blocks.erase(it);
    return true;
  }

  BlockPool& getPool(size_t size) {
    if (size <= kSmallSize) {
      return smallBlocks;
    }
    return largeBlocks;
  }

  bool allocBlock(AllocParams& p) {
    size_t size = p.allocSize;
    void* ptr = nullptr;
    base_->allocate(&ptr, size);
    if (!ptr) {
      return false;
    }
    totalAllocatedSize_ += size;
    p.retBlock = new Block(size, p.pool, ptr);
    assert(p.retBlock != nullptr && p.retBlock->ptr != nullptr);
    return true;
  }

  static bool shouldSplit(const Block* block, size_t size) {
    size_t remaining = block->size - size;
    if (block->pool->isSmall) {
      return remaining >= kMinBlockSize;
    }
    return remaining > kSmallSize;
  }

  void releaseCachedBlocks() {
    releaseBlocks(largeBlocks);
    releaseBlocks(smallBlocks);
  }

  void releaseBlock(Block* block) {
    base_->deallocate(block->ptr);
    totalAllocatedSize_ -= block->size;
    auto* pool = block->pool;
    pool->blocks.erase(block);
    delete block;
  }

  void releaseBlocks(BlockPool& pool) {
    // Frees all non-split blocks
    auto it = pool.blocks.begin();
    while (it != pool.blocks.end()) {
      Block* block = *it;
      ++it;
      if (!block->prev && !block->next) {
        releaseBlock(block);
      }
    }
  }

  Block* mallocImpl(size_t origSize) {
    size_t size = roundSize(origSize);
    auto& pool = getPool(size);
    const size_t allocSize = getAllocationSize(size);
    AllocParams params(size, &pool, allocSize);

    bool blockFound = getFreeBlock(params);

    if (!blockFound) {
      blockFound = allocBlock(params);
      if (!blockFound) {
        // retry after release caches
        releaseCachedBlocks();
        blockFound = allocBlock(params);
      }

      if (!blockFound) {
        LOGE("Out of memory. failed to allocate size: %lld", allocSize);
        return nullptr;
      }
    }

    assert(params.retBlock != nullptr && params.retBlock->ptr != nullptr);
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
    }

    block->allocated = true;
    activeBlocks[block->ptr] = block;
    return block;
  }

  void freeImpl(Block* block) {
    block->allocated = false;
    freeBlock(block);
  }

  void freeBlock(Block* block) {
    assert(!block->allocated);
    auto& pool = *block->pool;

    const std::array<Block*, 2> mergeCandidates = {block->prev, block->next};
    for (Block* candidate : mergeCandidates) {
      tryMergeBlocks(block, candidate, pool);
    }

    activeBlocks.erase(block->ptr);
    pool.blocks.insert(block);
  }

  static size_t tryMergeBlocks(Block* dst, Block* src, BlockPool& pool) {
    if (!src || src->allocated) {
      return 0;
    }
    assert(dst->isSplit() && src->isSplit());

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

  Device device() override { return base_->device(); }

  void allocate(void** ptr, size_t size) override {
    Block* block = mallocImpl(size);
    if (block) {
      *ptr = block->ptr;
    } else {
      LOGE("allocate error, size: %lld", size);
    }
  }

  void deallocate(void* ptr) override {
    auto it = activeBlocks.find(ptr);
    if (it != activeBlocks.end()) {
      freeImpl(it->second);
    } else {
      LOGE("deallocate error, ptr not valid: %p", ptr);
    }
  }

  void clear() override {
    releaseCachedBlocks();
    assert(activeBlocks.empty());
    assert(largeBlocks.blocks.empty());
    assert(smallBlocks.blocks.empty());
  }

 private:
  Allocator* base_;
  uint64_t totalAllocatedSize_;

  BlockPool largeBlocks;
  BlockPool smallBlocks;
  std::unordered_map<void*, Block*> activeBlocks;
};

CachedAllocator::CachedAllocator(std::unique_ptr<Allocator> base)
    : cacheEnabled_(true),
      base_(std::move(base)),
      impl_(std::make_unique<CachedAllocatorImpl>(base_.get())) {}

}  // namespace TinyTorch
