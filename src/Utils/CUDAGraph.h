/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#ifdef USE_CUDA

#include "CUDAUtils.h"

namespace tinytorch::cuda {

class CUDAGraph {
 public:
  CUDAGraph() = default;
  ~CUDAGraph() { reset(); }

  CUDAGraph(CUDAGraph&& other) noexcept : graph_(other.graph_), graphExec_(other.graphExec_) {
    other.graph_ = nullptr;
    other.graphExec_ = nullptr;
  }

  CUDAGraph& operator=(CUDAGraph&& other) noexcept {
    if (this != &other) {
      reset();
      graph_ = other.graph_;
      graphExec_ = other.graphExec_;
      other.graph_ = nullptr;
      other.graphExec_ = nullptr;
    }
    return *this;
  }

  CUDAGraph(const CUDAGraph&) = delete;
  CUDAGraph& operator=(const CUDAGraph&) = delete;

  void beginCapture(CUDAStream& stream, cudaStreamCaptureMode mode = cudaStreamCaptureModeGlobal) {
    if (graph_ || graphExec_) {
      // auto-reset to avoid resource leaks in release mode
      reset();
    }
    ASSERT(stream.valid() && "Stream must be valid for capture");
    CUDA_CHECK(cudaStreamBeginCapture(stream.stream(), mode));
  }

  void endCapture(CUDAStream& stream) {
    ASSERT(!graph_ && !graphExec_ && "Graph already instantiated");
    CUDA_CHECK(cudaStreamEndCapture(stream.stream(), &graph_));
    ASSERT(graph_ != nullptr && "cudaStreamEndCapture returned null graph");

    // instantiate: creates the executable graph from the captured graph
#if CUDART_VERSION >= 12000
    CUDA_CHECK(cudaGraphInstantiate(&graphExec_, graph_, 0ULL));
#else
    CUDA_CHECK(cudaGraphInstantiate(&graphExec_, graph_, nullptr, nullptr, 0));
#endif
    ASSERT(graphExec_ != nullptr);
  }

  void replay(CUDAStream& stream) {
    ASSERT(graphExec_ != nullptr && "No graph captured; call beginCapture/endCapture first");
    CUDA_CHECK(cudaGraphLaunch(graphExec_, stream.stream()));
  }

  void reset() {
    if (graphExec_) {
      CUDA_CHECK(cudaGraphExecDestroy(graphExec_));
      graphExec_ = nullptr;
    }
    if (graph_) {
      CUDA_CHECK(cudaGraphDestroy(graph_));
      graph_ = nullptr;
    }
  }

  bool valid() const { return graphExec_ != nullptr; }

 private:
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t graphExec_ = nullptr;
};

class CUDAGraphCaptureGuard {
 public:
  CUDAGraphCaptureGuard(CUDAGraph& graph, CUDAStream& stream, int poolId);
  ~CUDAGraphCaptureGuard();

  CUDAGraphCaptureGuard(const CUDAGraphCaptureGuard&) = delete;
  CUDAGraphCaptureGuard& operator=(const CUDAGraphCaptureGuard&) = delete;
  CUDAGraphCaptureGuard(CUDAGraphCaptureGuard&&) = delete;
  CUDAGraphCaptureGuard& operator=(CUDAGraphCaptureGuard&&) = delete;

 private:
  CUDAGraph& graph_;
  CUDAStream& stream_;
  int poolId_;
};

}  // namespace tinytorch::cuda

#endif  // USE_CUDA
