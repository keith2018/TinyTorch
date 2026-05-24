/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "CUDAGraph.h"

#ifdef USE_CUDA

#include "Tensor/Allocator.h"
#include "Tensor/CachedAllocator.h"

namespace tinytorch::cuda {

CUDAGraphCaptureGuard::CUDAGraphCaptureGuard(CUDAGraph& graph, CUDAStream& stream, int poolId)
    : graph_(graph), stream_(stream), poolId_(poolId) {
  CachedAllocator* allocator = getCUDACachedAllocator(stream.deviceIdx());
  ASSERT(allocator != nullptr && "Failed to get CUDA CachedAllocator");
  allocator->beginAllocateToPool(poolId_);

  graph_.beginCapture(stream_);
}

CUDAGraphCaptureGuard::~CUDAGraphCaptureGuard() {
  CachedAllocator* allocator = getCUDACachedAllocator(stream_.deviceIdx());
  if (allocator) {
    allocator->endAllocateToPool();
  }

  graph_.endCapture(stream_);
}

}  // namespace tinytorch::cuda

#endif  // USE_CUDA
