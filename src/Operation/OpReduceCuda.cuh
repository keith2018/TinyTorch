/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "OpElemWise.h"
#include "OpReduce.h"
#include "OpTransformCuda.cuh"
#include "Tensor/Storage.h"
#include "Utils/CUDAUtils.h"

namespace tinytorch::op {

struct OpCudaReduceMin {
  template <typename T>
  __device__ static T apply(const T& a, const T& b) {
    return ::min(a, b);
  }

  template <typename T>
  __device__ static bool compare(const T& a, const T& b) {
    return a < b;
  }

  template <typename T>
  __device__ static T defaultVal() {
    return cuda::Inf<T>();
  }
};

struct OpCudaReduceMax {
  template <typename T>
  __device__ static T apply(const T& a, const T& b) {
    return ::max(a, b);
  }

  template <typename T>
  __device__ static bool compare(const T& a, const T& b) {
    return a > b;
  }

  template <typename T>
  __device__ static T defaultVal() {
    return -cuda::Inf<T>();
  }
};

struct OpCudaReduceSum {
  template <typename T>
  __device__ static T apply(const T& a, const T& b) {
    return a + b;
  }

  template <typename T>
  __device__ static T defaultVal() {
    return 0;
  }
};

__device__ __forceinline__ int64_t cudaGetReduceSrcIndex(const int64_t* retShape, const int64_t* srcStrides,
                                                         int64_t idx, int64_t dim, int64_t retDimCount) {
  int64_t outIndex = idx;
  int64_t inIndex = 0;
#pragma unroll
  for (int64_t d = retDimCount - 1; d >= 0; d--) {
    const int64_t coord = outIndex % retShape[d];
    outIndex /= retShape[d];
    inIndex += coord * srcStrides[d < dim ? d : d + 1];
  }
  return inIndex;
}

__device__ __forceinline__ int64_t cudaGetReduceDstIndex(const int64_t* shape, const int64_t* strides, int64_t idx,
                                                         int64_t dim, int64_t dimCount) {
  int64_t retIdx = 0;
  int64_t stride = 1;
#pragma unroll
  for (int64_t d = dimCount - 1; d >= 0; d--) {
    if (d != dim) {
      retIdx += (idx / strides[d] % shape[d]) * stride;
      stride *= shape[d];
    }
  }
  return retIdx;
}

__device__ __forceinline__ int64_t cudaGetReduceDstIndex(const int64_t* shape, const int64_t* strides,
                                                         const int64_t* inAxis, int64_t idx, int64_t dimCount) {
  int64_t retIdx = 0;
  int64_t stride = 1;
#pragma unroll
  for (int64_t d = dimCount - 1; d >= 0; d--) {
    if (0 == inAxis[d]) {
      retIdx += (idx / strides[d] % shape[d]) * stride;
      stride *= shape[d];
    }
  }
  return retIdx;
}

struct CudaReduceIdx {
  int64_t inIdx;
  int64_t outIdx;
};

struct CudaReduceIndexAll {
  __device__ CudaReduceIdx operator()(const int64_t index, const int64_t segLength, const int64_t segCount) const {
    const int64_t idx = (index < segLength) ? index : -1;
    return {idx, idx};
  }
};

struct CudaReduceIndexFirstDim {
  __device__ CudaReduceIdx operator()(const int64_t index, const int64_t segLength, const int64_t segCount) const {
    const auto segDim = gridDim.x * blockDim.x / segCount;
    const int64_t segIdx = index / segDim;
    const int64_t segTid = index % segDim;
    const int64_t idx = (segTid < segLength) ? segIdx + segTid * segCount : -1;
    return {idx, segTid};
  }
};

struct CudaReduceIndexLastDim {
  __device__ CudaReduceIdx operator()(const int64_t index, const int64_t segLength, const int64_t segCount) const {
    const auto segDim = gridDim.x * blockDim.x / segCount;
    const int64_t segIdx = index / segDim;
    const int64_t segTid = index % segDim;
    const int64_t idx = (segTid < segLength) ? segTid + segIdx * segLength : -1;
    return {idx, segTid};
  }
};

template <typename T, typename OP>
__device__ __forceinline__ void cudaWarpReduce(T& val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val = OP::template apply(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
  }
}

template <typename T, typename OP>
__device__ __forceinline__ void cudaWarpReduceIdx(T& val, int64_t& idx) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    T otherVal = __shfl_down_sync(0xFFFFFFFF, val, offset);
    int64_t otherIdx = __shfl_down_sync(0xFFFFFFFF, idx, offset);
    if (OP::template apply(otherVal, val) == otherVal) {
      val = otherVal;
      idx = otherIdx;
    }
  }
}

template <typename T, typename OP, typename IndexFunc>
__global__ void kReduceMerge(T* outValues, const T* inValues, const int64_t segLength, const int64_t segCount) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ T shared[CUDA_WARP_SIZE];
  T val = OP::template defaultVal<T>();

  const IndexFunc indexer;
  const auto [inIdx, outIdx] = indexer(index, segLength, segCount);
  if (inIdx >= 0) {
    val = inValues[inIdx];
  }
  cudaWarpReduce<T, OP>(val);

  if (threadIdx.x % warpSize == 0) {
    shared[threadIdx.x / warpSize] = val;
  }
  __syncthreads();

  if (threadIdx.x < warpSize) {
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[threadIdx.x] : OP::template defaultVal<T>();
    cudaWarpReduce<T, OP>(val);
  }

  if (threadIdx.x == 0) {
    outValues[blockIdx.x] = val;
  }
}

template <typename T, typename OP, typename IndexFunc>
__global__ void kReduceIdxMerge(T* outValues, int64_t* outIndices, const T* inValues, const int64_t* inIndices,
                                const int64_t segLength, const int64_t segCount) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ T sharedVal[CUDA_WARP_SIZE];
  __shared__ int64_t sharedIdx[CUDA_WARP_SIZE];

  T val = OP::template defaultVal<T>();
  int64_t idx = -1;

  const IndexFunc indexer;
  const auto [inIdx, outIdx] = indexer(index, segLength, segCount);
  if (inIdx >= 0) {
    val = inValues[inIdx];
    idx = inIndices ? inIndices[inIdx] : outIdx;
  }
  cudaWarpReduceIdx<T, OP>(val, idx);

  if (threadIdx.x % warpSize == 0) {
    sharedVal[threadIdx.x / warpSize] = val;
    sharedIdx[threadIdx.x / warpSize] = idx;
  }
  __syncthreads();

  if (threadIdx.x < warpSize) {
    val = (threadIdx.x < blockDim.x / warpSize) ? sharedVal[threadIdx.x] : OP::template defaultVal<T>();
    idx = (threadIdx.x < blockDim.x / warpSize) ? sharedIdx[threadIdx.x] : -1;

    cudaWarpReduceIdx<T, OP>(val, idx);
  }

  if (threadIdx.x == 0) {
    if (outIndices) outIndices[blockIdx.x] = idx;
    if (outValues) outValues[blockIdx.x] = val;
  }
}

template <typename T, typename OP, bool isFirstDim>
__global__ void kReduceDimFirstOrLast(T* outValues, const T* inValues, const int64_t segLength,
                                      const int64_t segCount) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < segCount) {
    auto targetVal = OP::template defaultVal<T>();
    auto srcIdx = isFirstDim ? index : index * segLength;
#pragma unroll
    for (int64_t j = 0; j < segLength; j++) {
      const auto val = inValues[srcIdx];
      srcIdx += isFirstDim ? segCount : 1;
      targetVal = OP::template apply(val, targetVal);
    }
    outValues[index] = targetVal;
  }
}

template <typename T, typename OP>
__global__ void kReduceDim(cuda::TensorCudaCtx values, const cuda::TensorCudaCtx t, const int64_t dim,
                           const int64_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;

  const auto dimSize = t.shape[dim];
  const auto stride = t.strides[dim];

  if (index < n) {
    const auto* tPtr = static_cast<T*>(t.data);
    auto* valuesPtr = static_cast<T*>(values.data);

    auto targetVal = OP::template defaultVal<T>();
    int64_t srcIdx = cudaGetReduceSrcIndex(values.shape, t.strides, index, dim, values.ndim);
#pragma unroll
    for (int64_t j = 0; j < dimSize; j++) {
      const auto val = tPtr[srcIdx];
      srcIdx += stride;
      targetVal = OP::template apply(val, targetVal);
    }
    valuesPtr[index] = targetVal;
  }
}

template <typename T, typename OP, bool isFirstDim>
__global__ void kReduceIdxDimFirstOrLast(T* outValues, int64_t* outIndices, const T* inValues, const int64_t segLength,
                                         const int64_t segCount) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < segCount) {
    auto targetVal = OP::template defaultVal<T>();
    int64_t targetIdx = 0;
    auto srcIdx = isFirstDim ? index : index * segLength;
#pragma unroll
    for (int64_t j = 0; j < segLength; j++) {
      const auto val = inValues[srcIdx];
      srcIdx += isFirstDim ? segCount : 1;
      if (OP::template apply(val, targetVal) == val) {
        targetVal = val;
        targetIdx = j;
      }
    }
    outValues[index] = targetVal;
    outIndices[index] = targetIdx;
  }
}

template <typename T, typename OP>
__global__ void kReduceIdxDim(cuda::TensorCudaCtx values, int64_t* indices, const cuda::TensorCudaCtx t,
                              const int64_t dim, const int64_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;

  const auto dimSize = t.shape[dim];
  const auto stride = t.strides[dim];

  if (index < n) {
    const auto* tPtr = static_cast<T*>(t.data);
    auto* valuesPtr = static_cast<T*>(values.data);

    auto targetVal = OP::template defaultVal<T>();
    int64_t targetIdx = 0;
    int64_t srcIdx = cudaGetReduceSrcIndex(values.shape, t.strides, index, dim, values.ndim);
#pragma unroll
    for (int64_t j = 0; j < dimSize; j++) {
      const auto val = tPtr[srcIdx];
      srcIdx += stride;
      if (OP::template apply(val, targetVal) == val) {
        targetVal = val;
        targetIdx = j;
      }
    }
    valuesPtr[index] = targetVal;
    indices[index] = targetIdx;
  }
}

// TODO optimize
template <typename T>
__global__ void kReduceMultiDimSum(T* retPtr, const cuda::TensorCudaCtx t, const DimArray<int64_t> inAxis,
                                   const int64_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    const int64_t retIdx = cudaGetReduceDstIndex(t.shape, t.strides, inAxis.data, index, t.ndim);
    const auto* tPtr = static_cast<T*>(t.data);
    atomicAdd(&retPtr[retIdx], tPtr[index]);
  }
}

// TODO optimize
template <typename T>
__global__ void kReduceMultiDimVar(T* retPtr, const cuda::TensorCudaCtx t, const T* meanValues,
                                   const DimArray<int64_t> inAxis, const int64_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    const int64_t retIdx = cudaGetReduceDstIndex(t.shape, t.strides, inAxis.data, index, t.ndim);
    const auto* tPtr = static_cast<T*>(t.data);
    const T diff = tPtr[index] - meanValues[retIdx];
    atomicAdd(&retPtr[retIdx], diff * diff);
  }
}

template <typename T>
__global__ void kSquaredDiff(T* output, const T* input, const T* mean, const int64_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    const T diff = input[index] - *mean;
    output[index] = diff * diff;
  }
}

class ReducerCuda {
 public:
  template <typename T, typename OP, typename IndexFunc>
  static void reduceMerge(T* values, const T* input, const Device& device, int64_t n, int64_t m = 1);
  template <typename T, typename OP, typename IndexFunc>
  static void reduceIdxMerge(T* values, int64_t* indices, const T* input, const Device& device, int64_t n,
                             int64_t m = 1);

  template <typename T, typename OP>
  static void reduceDimFirst(T* values, const T* input, const Device& device, int64_t n, int64_t m = 1);
  template <typename T, typename OP>
  static void reduceDimLast(T* values, const T* input, const Device& device, int64_t n, int64_t m = 1);
  template <typename T, typename OP>
  static void reduceIdxDimFirst(T* values, int64_t* indices, const T* input, const Device& device, int64_t n,
                                int64_t m = 1);
  template <typename T, typename OP>
  static void reduceIdxDimLast(T* values, int64_t* indices, const T* input, const Device& device, int64_t n,
                               int64_t m = 1);

  template <typename T, typename OP>
  static Tensor reduceDim(const Tensor& t, int64_t dim, bool keepDims);
  template <typename T, typename OP>
  static TensorPair reduceIdxDim(const Tensor& t, int64_t dim, bool keepDims);
};

template <typename T, typename OP, typename IndexFunc>
void ReducerCuda::reduceMerge(T* values, const T* input, const Device& device, int64_t n, int64_t m) {
  auto* allocator = getAllocator(options::device(device));
  const auto blockSize = cuda::getKernelBlockSize(device.index);
  const auto stream = cuda::getCurrentCUDAStream(device.index).stream;

  auto blocks = cuda::getKernelGridSize(blockSize, n);
  auto* dTmp = static_cast<T*>(allocator->allocate(m * blocks * sizeof(T)));

  // first round
  kReduceMerge<T, OP, IndexFunc><<<m * blocks, blockSize, 0, stream>>>(dTmp, input, n, m);
  CUDA_KERNEL_CHECK();

  // merge
  while (blocks > 1) {
    const auto currBlocks = blocks;
    blocks = cuda::getKernelGridSize(blockSize, currBlocks);
    kReduceMerge<T, OP, IndexFunc><<<m * blocks, blockSize, 0, stream>>>(dTmp, dTmp, currBlocks, m);
    CUDA_KERNEL_CHECK();
  }
  Storage::copyOnDevice(values, dTmp, device, m * sizeof(T));
  allocator->deallocate(dTmp);
}

template <typename T, typename OP, typename IndexFunc>
void ReducerCuda::reduceIdxMerge(T* values, int64_t* indices, const T* input, const Device& device, int64_t n,
                                 int64_t m) {
  auto* allocator = getAllocator(options::device(device));
  const auto blockSize = cuda::getKernelBlockSize(device.index);
  const auto stream = cuda::getCurrentCUDAStream(device.index).stream;

  auto blocks = cuda::getKernelGridSize(blockSize, n);

  auto* tmpValues = static_cast<T*>(allocator->allocate(m * blocks * sizeof(T)));
  auto* tmpIndices = static_cast<int64_t*>(allocator->allocate(m * blocks * static_cast<int64_t>(sizeof(int64_t))));

  // first round
  kReduceIdxMerge<T, OP, IndexFunc><<<m * blocks, blockSize, 0, stream>>>(tmpValues, tmpIndices, input, nullptr, n, m);
  CUDA_KERNEL_CHECK();

  // merge
  while (blocks > 1) {
    const auto currBlocks = blocks;
    blocks = cuda::getKernelGridSize(blockSize, currBlocks);
    kReduceIdxMerge<T, OP, IndexFunc>
        <<<m * blocks, blockSize, 0, stream>>>(tmpValues, tmpIndices, tmpValues, tmpIndices, currBlocks, m);
    CUDA_KERNEL_CHECK();
  }
  if (values) {
    Storage::copyOnDevice(values, tmpValues, device, m * sizeof(T));
  }
  if (indices) {
    Storage::copyOnDevice(indices, tmpIndices, device, m * static_cast<int64_t>(sizeof(int64_t)));
  }
  allocator->deallocate(tmpValues);
  allocator->deallocate(tmpIndices);
}

template <typename T, typename OP>
void ReducerCuda::reduceDimFirst(T* values, const T* input, const Device& device, int64_t n, int64_t m) {
  // faster
  auto tmp = Tensor::empty({m * n}, options::device(device));
  cudaTranspose2d(tmp.dataPtr<T>(), input, m, n, device);
  reduceMerge<T, OP, CudaReduceIndexLastDim>(values, tmp.dataPtr<T>(), device, n, m);

  // slower than transpose
  // reduceMerge<T, OP, CudaReduceIndexFirstDim>(values, input, device, n, m);
}

template <typename T, typename OP>
void ReducerCuda::reduceDimLast(T* values, const T* input, const Device& device, int64_t n, int64_t m) {
  reduceMerge<T, OP, CudaReduceIndexLastDim>(values, input, device, n, m);
}

template <typename T, typename OP>
void ReducerCuda::reduceIdxDimFirst(T* values, int64_t* indices, const T* input, const Device& device, int64_t n,
                                    int64_t m) {
  // faster
  auto tmp = Tensor::empty({m * n}, options::device(device));
  cudaTranspose2d(tmp.dataPtr<T>(), input, m, n, device);
  reduceIdxMerge<T, OP, CudaReduceIndexLastDim>(values, indices, tmp.dataPtr<T>(), device, n, m);

  // slower than transpose
  // reduceIdxMerge<T, OP, CudaReduceIndexFirstDim>(values, indices, input, device, n, m);
}

template <typename T, typename OP>
void ReducerCuda::reduceIdxDimLast(T* values, int64_t* indices, const T* input, const Device& device, int64_t n,
                                   int64_t m) {
  reduceIdxMerge<T, OP, CudaReduceIndexLastDim>(values, indices, input, device, n, m);
}

template <typename T, typename OP>
Tensor ReducerCuda::reduceDim(const Tensor& t, int64_t dim, bool keepDims) {
  if (dim < 0) {
    dim += t.dim();
  }
  if (dim < 0 || dim >= t.dim()) {
    LOGE("Invalid axis: %lld", dim);
    ASSERT(false);
    return {};
  }

  const auto retShape = getReduceShape(t, dim, false);
  auto values = Tensor::empty(retShape.view(), t.options().noGrad());

  const auto blockSize = cuda::getKernelBlockSize(t.device().index);

  // first dim
  if (dim == 0) {
    const auto dimSize = t.shape().front();
    if (static_cast<uint64_t>(dimSize) < blockSize) {
      auto params = cuda::getKernelLaunchParams(t.device().index, t.numel());
      CUDA_LAUNCH_KERNEL((kReduceDimFirstOrLast<T, OP, true>), params, values.dataPtr<T>(), t.dataPtr<T>(), dimSize,
                         values.numel());
    } else {
      reduceDimFirst<T, OpCudaReduceSum>(values.dataPtr<T>(), t.dataPtr<T>(), t.device(), dimSize, values.numel());
    }
  } else if (dim == t.dim() - 1) {
    // last dim
    const auto dimSize = t.shape().back();
    if (static_cast<uint64_t>(dimSize) < blockSize) {
      auto params = cuda::getKernelLaunchParams(t.device().index, t.numel());
      CUDA_LAUNCH_KERNEL((kReduceDimFirstOrLast<T, OP, false>), params, values.dataPtr<T>(), t.dataPtr<T>(), dimSize,
                         values.numel());
    } else {
      reduceDimLast<T, OpCudaReduceSum>(values.dataPtr<T>(), t.dataPtr<T>(), t.device(), dimSize, values.numel());
    }
  } else {
    // other dim
    auto ctxT = cuda::getTensorCudaCtx(t);
    auto ctxValues = cuda::getTensorCudaCtx(values);

    auto params = cuda::getKernelLaunchParams(t.device().index, t.numel());
    CUDA_LAUNCH_KERNEL((kReduceDim<T, OP>), params, ctxValues, ctxT, dim, ctxValues.numel);
  }

  if (keepDims) {
    const auto shapeKeepDims = getReduceShape(t, dim, true);
    values.reshape(shapeKeepDims.view());
  }
  return values;
}

template <typename T, typename OP>
TensorPair ReducerCuda::reduceIdxDim(const Tensor& t, int64_t dim, bool keepDims) {
  if (dim < 0) {
    dim += t.dim();
  }
  if (dim < 0 || dim >= t.dim()) {
    LOGE("Invalid axis: %lld", dim);
    ASSERT(false);
    return {};
  }

  const auto retShape = getReduceShape(t, dim, false);
  auto values = Tensor::empty(retShape.view(), t.options().noGrad());
  auto indices = Tensor::empty(retShape.view(), getIndicesOptions(t));

  const auto blockSize = cuda::getKernelBlockSize(t.device().index);

  if (dim == 0) {
    // first dim
    const auto dimSize = t.shape().front();
    if (static_cast<uint64_t>(dimSize) < blockSize) {
      auto params = cuda::getKernelLaunchParams(t.device().index, t.numel());
      CUDA_LAUNCH_KERNEL((kReduceIdxDimFirstOrLast<T, OP, true>), params, values.dataPtr<T>(),
                         indices.dataPtr<int64_t>(), t.dataPtr<T>(), dimSize, values.numel());
    } else {
      reduceIdxDimFirst<T, OP>(values.dataPtr<T>(), indices.dataPtr<int64_t>(), t.dataPtr<T>(), t.device(), dimSize,
                               values.numel());
    }
  } else if (dim == t.dim() - 1) {
    // last dim
    const auto dimSize = t.shape().back();
    if (static_cast<uint64_t>(dimSize) < blockSize) {
      auto params = cuda::getKernelLaunchParams(t.device().index, t.numel());
      CUDA_LAUNCH_KERNEL((kReduceIdxDimFirstOrLast<T, OP, false>), params, values.dataPtr<T>(),
                         indices.dataPtr<int64_t>(), t.dataPtr<T>(), dimSize, values.numel());
    } else {
      reduceIdxDimLast<T, OP>(values.dataPtr<T>(), indices.dataPtr<int64_t>(), t.dataPtr<T>(), t.device(), dimSize,
                              values.numel());
    }
  } else {
    // other dim
    auto ctxT = cuda::getTensorCudaCtx(t);
    auto ctxValues = cuda::getTensorCudaCtx(values);

    auto params = cuda::getKernelLaunchParams(t.device().index, t.numel());
    CUDA_LAUNCH_KERNEL((kReduceIdxDim<T, OP>), params, ctxValues, indices.dataPtr<int64_t>(), ctxT, dim,
                       ctxValues.numel);
  }

  if (keepDims) {
    const auto shapeKeepDims = getReduceShape(t, dim, true);
    values.reshape(shapeKeepDims.view());
    indices.reshape(shapeKeepDims.view());
  }
  return {values, indices};
}

template <typename T, typename OP>
Tensor reduceOpAllCudaImpl(const Tensor& t) {
  if (t.isScalar()) {
    return t;
  }
  auto ret = Tensor::scalar(0, t.options().noGrad());
  const T* tPtr = t.dataPtr<T>();
  T* retPtr = ret.dataPtr<T>();
  ReducerCuda::reduceMerge<T, OP, CudaReduceIndexAll>(retPtr, tPtr, t.device(), t.numel());
  return ret;
}

template <typename T, typename OP>
Tensor reduceOpArgMinMaxCudaImpl(const Tensor& t) {
  auto ret = Tensor::scalar(0, getIndicesOptions(t));
  if (t.isScalar()) {
    return ret;
  }
  const T* tPtr = t.dataPtr<T>();
  auto* retPtr = ret.dataPtr<int64_t>();
  ReducerCuda::reduceIdxMerge<T, OP, CudaReduceIndexAll>(nullptr, retPtr, tPtr, t.device(), t.numel());
  return ret;
}

template <typename T, typename OP>
TensorPair reduceOpMinMaxDimCudaImpl(const Tensor& t, int64_t dim, bool keepDims = false) {
  if (t.isScalar()) {
    return {t, Tensor::scalar(0, getIndicesOptions(t))};
  }
  return ReducerCuda::reduceIdxDim<T, OP>(t, dim, keepDims);
}

template <typename T>
Tensor reduceOpSumDimsCudaImpl(const Tensor& t, const IntArrayView dims, bool keepDims = false) {
  if (t.isScalar()) {
    return t;
  }
  ASSERT(!dims.empty());

  if (dims.size() == 1) {
    return ReducerCuda::reduceDim<T, OpCudaReduceSum>(t, dims[0], keepDims);
  }

  DimArray<int64_t> inAxis{};
  for (int64_t d : dims) {
    if (d < 0) {
      d += t.dim();
    }
    if (d < 0 || d >= t.dim()) {
      LOGE("Invalid reduce dim: %lld", d);
      ASSERT(false);
      return {};
    }
    inAxis.data[d] = 1;
  }

  const auto retShape = getReduceShape(t, inAxis, keepDims);
  auto ret = Tensor::zeros(retShape.view(), t.options().noGrad());

  auto ctxT = cuda::getTensorCudaCtx(t);

  auto params = cuda::getKernelLaunchParams(t.device().index, t.numel());
  CUDA_LAUNCH_KERNEL((kReduceMultiDimSum<T>), params, ret.dataPtr<T>(), ctxT, inAxis, t.numel());
  return ret;
}

template <typename T>
Tensor reduceOpSumDimCudaImpl(const Tensor& t, const int64_t dim, bool keepDims = false) {
  return reduceOpSumDimsCudaImpl<T>(t, {dim}, keepDims);
}

template <typename T>
Tensor reduceOpSumCudaImpl(const Tensor& t) {
  return reduceOpAllCudaImpl<T, OpCudaReduceSum>(t);
}

template <typename T>
Tensor reduceOpMeanCudaImpl(const Tensor& t) {
  if (t.isScalar()) {
    return t;
  }
  auto ret = reduceOpAllCudaImpl<T, OpCudaReduceSum>(t);
  const auto r = 1.f / static_cast<float>(t.numel());
  op::mulInplace(ret, Tensor::scalar(r, ret.options().noGrad()));
  return ret;
}

template <typename T>
Tensor reduceOpMeanDimsCudaImpl(const Tensor& t, const IntArrayView dims, bool keepDims = false) {
  if (t.isScalar()) {
    return t;
  }
  ASSERT(!dims.empty());
  auto ret = reduceOpSumDimsCudaImpl<T>(t, dims, keepDims);
  if (ret.defined()) {
    auto r = static_cast<float>(ret.numel()) / static_cast<float>(t.numel());
    op::mulInplace(ret, Tensor::scalar(r, ret.options().noGrad()));
  }
  return ret;
}

template <typename T>
Tensor reduceOpMeanDimCudaImpl(const Tensor& t, const int64_t dim, bool keepDims = false) {
  return reduceOpMeanDimsCudaImpl<T>(t, {dim}, keepDims);
}

template <typename T>
TensorPair reduceOpVarMeanCudaImpl(const Tensor& t, bool unbiased = true) {
  if (t.isScalar()) {
    return {Tensor::scalar(0, t.options().noGrad()), t};
  }
  const auto meanVal = op::mean(t);
  auto squaredDiff = Tensor::empty({t.numel()}, t.options().noGrad());

  const T* tPtr = t.dataPtr<T>();
  const T* meanPtr = meanVal.dataPtr<T>();
  T* squaredDiffPtr = squaredDiff.dataPtr<T>();

  auto params = cuda::getKernelLaunchParams(t.device().index, t.numel());
  CUDA_LAUNCH_KERNEL((kSquaredDiff<T>), params, squaredDiffPtr, tPtr, meanPtr, t.numel());

  auto varVal = Tensor::empty({}, t.options().noGrad());
  T* varPtr = varVal.dataPtr<T>();
  ReducerCuda::reduceMerge<T, OpCudaReduceSum, CudaReduceIndexAll>(varPtr, squaredDiffPtr, t.device(), t.numel());

  const auto n = static_cast<float>(t.numel());
  auto r = 1.f / n;
  if (unbiased) {
    r *= n / (n - 1.f);
  }
  op::mulInplace(varVal, Tensor::scalar(r, varVal.options().noGrad()));
  return {varVal, meanVal};
}

template <typename T>
TensorPair reduceOpVarMeanDimsCudaImpl(const Tensor& t, IntArrayView dims, bool unbiased = true,
                                       bool keepDims = false) {
  if (t.isScalar()) {
    return {Tensor::scalar(0, t.options().noGrad()), t};
  }
  ASSERT(!dims.empty());

  DimArray<int64_t> inAxis{};
  for (int64_t d : dims) {
    if (d < 0) {
      d += t.dim();
    }
    if (d < 0 || d >= t.dim()) {
      LOGE("Invalid reduce dim: %lld", d);
      ASSERT(false);
      return {};
    }
    inAxis.data[d] = 1;
  }

  auto meanVal = op::meanOnDims(t, dims, true);

  auto retShape = getReduceShape(t, inAxis, keepDims);
  auto varVal = Tensor::zeros(retShape.view(), t.options().noGrad());

  auto ctxT = cuda::getTensorCudaCtx(t);
  const T* meanPtr = meanVal.dataPtr<T>();
  T* varPtr = varVal.dataPtr<T>();

  auto params = cuda::getKernelLaunchParams(t.device().index, t.numel());
  CUDA_LAUNCH_KERNEL((kReduceMultiDimVar<T>), params, varPtr, ctxT, meanPtr, inAxis, t.numel());

  auto reduceSize = static_cast<float>(t.numel()) / static_cast<float>(varVal.numel());
  auto r = 1.f / reduceSize;
  if (unbiased) {
    r *= reduceSize / (reduceSize - 1.f);
  }
  op::mulInplace(varVal, Tensor::scalar(r, varVal.options().noGrad()));
  return {varVal, meanVal};
}

template <typename T>
TensorPair reduceOpVarMeanDimCudaImpl(const Tensor& t, const int64_t dim, bool unbiased = true, bool keepDims = false) {
  return reduceOpVarMeanDimsCudaImpl<T>(t, {dim}, unbiased, keepDims);
}

}  // namespace tinytorch::op
