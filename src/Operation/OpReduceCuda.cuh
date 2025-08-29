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
#include "Utils/CUDAMath.h"
#include "Utils/CUDAUtils.h"

namespace tinytorch::op {

struct OpCudaReduceMin {
  template <typename T>
  __device__ static T apply(const T& a, const T& b) {
    return cuda::min(a, b);
  }

  template <typename T>
  __device__ static bool compare(const T& a, const T& b) {
    return a < b;
  }

  template <typename T>
  __device__ static T defaultVal() {
    return std::numeric_limits<T>::max();
  }
};

struct OpCudaReduceMax {
  template <typename T>
  __device__ static T apply(const T& a, const T& b) {
    return cuda::max(a, b);
  }

  template <typename T>
  __device__ static bool compare(const T& a, const T& b) {
    return a > b;
  }

  template <typename T>
  __device__ static T defaultVal() {
    return std::numeric_limits<T>::lowest();
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

template <typename OutType, typename InType, typename OP, typename IndexFunc>
__global__ void kReduceMerge(OutType* outValues, const InType* inValues, const int64_t segLength,
                             const int64_t segCount) {
  using ComputeT = typename cuda::CudaComputeType<InType>::type;
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ ComputeT shared[CUDA_WARP_SIZE];
  ComputeT val = OP::template defaultVal<ComputeT>();

  const IndexFunc indexer;
  const auto [inIdx, outIdx] = indexer(index, segLength, segCount);
  if (inIdx >= 0) {
    val = static_cast<ComputeT>(inValues[inIdx]);
  }
  cudaWarpReduce<ComputeT, OP>(val);

  if (threadIdx.x % warpSize == 0) {
    shared[threadIdx.x / warpSize] = val;
  }
  __syncthreads();

  if (threadIdx.x < warpSize) {
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[threadIdx.x] : OP::template defaultVal<ComputeT>();
    cudaWarpReduce<ComputeT, OP>(val);
  }

  if (threadIdx.x == 0) {
    outValues[blockIdx.x] = static_cast<OutType>(val);
  }
}

template <typename OutType, typename InType, typename OP, typename IndexFunc>
__global__ void kReduceIdxMerge(OutType* outValues, int64_t* outIndices, const InType* inValues,
                                const int64_t* inIndices, const int64_t segLength, const int64_t segCount) {
  using ComputeT = typename cuda::CudaComputeType<InType>::type;
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ ComputeT sharedVal[CUDA_WARP_SIZE];
  __shared__ int64_t sharedIdx[CUDA_WARP_SIZE];

  ComputeT val = OP::template defaultVal<ComputeT>();
  int64_t idx = -1;

  const IndexFunc indexer;
  const auto [inIdx, outIdx] = indexer(index, segLength, segCount);
  if (inIdx >= 0) {
    val = static_cast<ComputeT>(inValues[inIdx]);
    idx = inIndices ? inIndices[inIdx] : outIdx;
  }
  cudaWarpReduceIdx<ComputeT, OP>(val, idx);

  if (threadIdx.x % warpSize == 0) {
    sharedVal[threadIdx.x / warpSize] = val;
    sharedIdx[threadIdx.x / warpSize] = idx;
  }
  __syncthreads();

  if (threadIdx.x < warpSize) {
    val = (threadIdx.x < blockDim.x / warpSize) ? sharedVal[threadIdx.x] : OP::template defaultVal<ComputeT>();
    idx = (threadIdx.x < blockDim.x / warpSize) ? sharedIdx[threadIdx.x] : -1;

    cudaWarpReduceIdx<ComputeT, OP>(val, idx);
  }

  if (threadIdx.x == 0) {
    if (outIndices) outIndices[blockIdx.x] = idx;
    if (outValues) outValues[blockIdx.x] = static_cast<OutType>(val);
  }
}

template <typename T, typename OP, bool isFirstDim>
__global__ void kReduceDimFirstOrLast(T* outValues, const T* inValues, const int64_t segLength,
                                      const int64_t segCount) {
  using ComputeT = typename cuda::CudaComputeType<T>::type;
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < segCount) {
    ComputeT targetVal = OP::template defaultVal<ComputeT>();
    int64_t srcIdx = isFirstDim ? index : index * segLength;
#pragma unroll
    for (int64_t j = 0; j < segLength; j++) {
      const auto val = static_cast<ComputeT>(inValues[srcIdx]);
      srcIdx += isFirstDim ? segCount : 1;
      targetVal = OP::template apply(val, targetVal);
    }
    outValues[index] = static_cast<T>(targetVal);
  }
}

template <typename T, typename OP>
__global__ void kReduceDim(cuda::TensorCudaCtx values, const cuda::TensorCudaCtx t, const int64_t dim,
                           const int64_t n) {
  using ComputeT = typename cuda::CudaComputeType<T>::type;
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;

  const auto dimSize = t.shape[dim];
  const auto stride = t.strides[dim];

  if (index < n) {
    const auto* tPtr = static_cast<T*>(t.data);
    auto* valuesPtr = static_cast<T*>(values.data);

    ComputeT targetVal = OP::template defaultVal<ComputeT>();
    int64_t srcIdx = cudaGetReduceSrcIndex(values.shape, t.strides, index, dim, values.ndim);
#pragma unroll
    for (int64_t j = 0; j < dimSize; j++) {
      const auto val = static_cast<ComputeT>(tPtr[srcIdx]);
      srcIdx += stride;
      targetVal = OP::template apply(val, targetVal);
    }
    valuesPtr[index] = static_cast<T>(targetVal);
  }
}

template <typename T, typename OP, bool isFirstDim>
__global__ void kReduceIdxDimFirstOrLast(T* outValues, int64_t* outIndices, const T* inValues, const int64_t segLength,
                                         const int64_t segCount) {
  using ComputeT = typename cuda::CudaComputeType<T>::type;
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < segCount) {
    ComputeT targetVal = OP::template defaultVal<ComputeT>();
    int64_t targetIdx = 0;
    auto srcIdx = isFirstDim ? index : index * segLength;
#pragma unroll
    for (int64_t j = 0; j < segLength; j++) {
      const auto val = static_cast<ComputeT>(inValues[srcIdx]);
      srcIdx += isFirstDim ? segCount : 1;
      if (OP::template apply(val, targetVal) == val) {
        targetVal = val;
        targetIdx = j;
      }
    }
    outValues[index] = static_cast<T>(targetVal);
    outIndices[index] = targetIdx;
  }
}

template <typename T, typename OP>
__global__ void kReduceIdxDim(cuda::TensorCudaCtx values, int64_t* indices, const cuda::TensorCudaCtx t,
                              const int64_t dim, const int64_t n) {
  using ComputeT = typename cuda::CudaComputeType<T>::type;
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;

  const auto dimSize = t.shape[dim];
  const auto stride = t.strides[dim];

  if (index < n) {
    const auto* tPtr = static_cast<T*>(t.data);
    auto* valuesPtr = static_cast<T*>(values.data);

    ComputeT targetVal = OP::template defaultVal<ComputeT>();
    int64_t targetIdx = 0;
    int64_t srcIdx = cudaGetReduceSrcIndex(values.shape, t.strides, index, dim, values.ndim);
#pragma unroll
    for (int64_t j = 0; j < dimSize; j++) {
      const auto val = static_cast<ComputeT>(tPtr[srcIdx]);
      srcIdx += stride;
      if (OP::template apply(val, targetVal) == val) {
        targetVal = val;
        targetIdx = j;
      }
    }
    valuesPtr[index] = static_cast<T>(targetVal);
    indices[index] = targetIdx;
  }
}

template <typename T, int BLOCK_SIZE>
__global__ void kReduceMultiDimSum(T* outPtr, const cuda::TensorCudaCtx t, const DimArray<int64_t> inAxis,
                                   const int64_t reduceCnt, const int64_t n) {
  using ComputeT = typename cuda::CudaComputeType<T>::type;
  __shared__ ComputeT sdata[BLOCK_SIZE];

  const int64_t outIdx = blockIdx.x;
  if (outIdx >= n) {
    return;
  }

  const auto* inPtr = static_cast<const T*>(t.data);
  ComputeT localSum = 0;

  for (int64_t r = threadIdx.x; r < reduceCnt; r += BLOCK_SIZE) {
    int64_t inIndex = 0;
    int64_t tmpOut = outIdx;
    int64_t tmpR = r;

    for (int64_t d = t.ndim - 1; d >= 0; d--) {
      int64_t coord;
      if (inAxis.data[d] == 0) {
        coord = tmpOut % t.shape[d];
        tmpOut /= t.shape[d];
      } else {
        coord = tmpR % t.shape[d];
        tmpR /= t.shape[d];
      }
      inIndex += coord * t.strides[d];
    }

    localSum += static_cast<ComputeT>(inPtr[inIndex]);
  }

  sdata[threadIdx.x] = localSum;
  __syncthreads();

  for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    outPtr[outIdx] = static_cast<T>(sdata[0]);
  }
}

template <typename T>
__global__ void kReduceMultiDimVar(T* retPtr, const cuda::TensorCudaCtx t, const T* meanValues,
                                   const DimArray<int64_t> inAxis, const int64_t n) {
  using ComputeT = typename cuda::CudaComputeType<T>::type;
  const int64_t outIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (outIdx >= n) {
    return;
  }

  const auto* tPtr = static_cast<const T*>(t.data);
  const ComputeT mean = static_cast<ComputeT>(meanValues[outIdx]);

  ComputeT sum = 0;
  int64_t count = 0;

  for (int64_t inIdx = 0; inIdx < t.numel; inIdx++) {
    const auto dstIdx = cudaGetReduceDstIndex(t.shape, t.strides, inAxis.data, inIdx, t.ndim);
    if (dstIdx == outIdx) {
      ComputeT diff = static_cast<ComputeT>(tPtr[inIdx]) - mean;
      sum += diff * diff;
      count++;
    }
  }

  retPtr[outIdx] = static_cast<T>(sum);
}

template <typename T>
__global__ void kSquaredDiff(T* output, const T* input, const T* mean, const int64_t n) {
  using ComputeT = typename cuda::CudaComputeType<T>::type;
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    const ComputeT diff = static_cast<ComputeT>(input[index]) - static_cast<ComputeT>(*mean);
    output[index] = static_cast<T>(diff * diff);
  }
}

class ReducerCuda {
 public:
  template <typename T, typename OP, typename IndexFunc>
  static void reduceMerge(T* values, const T* input, const Options& options, int64_t n, int64_t m = 1);
  template <typename T, typename OP, typename IndexFunc>
  static void reduceIdxMerge(T* values, int64_t* indices, const T* input, const Options& options, int64_t n,
                             int64_t m = 1);

  template <typename T, typename OP>
  static void reduceDimFirst(T* values, const T* input, const Options& options, int64_t n, int64_t m = 1);
  template <typename T, typename OP>
  static void reduceDimLast(T* values, const T* input, const Options& options, int64_t n, int64_t m = 1);
  template <typename T, typename OP>
  static void reduceIdxDimFirst(T* values, int64_t* indices, const T* input, const Options& options, int64_t n,
                                int64_t m = 1);
  template <typename T, typename OP>
  static void reduceIdxDimLast(T* values, int64_t* indices, const T* input, const Options& options, int64_t n,
                               int64_t m = 1);

  template <typename T, typename OP>
  static Tensor reduceDim(const Tensor& t, int64_t dim, bool keepDims);
  template <typename T, typename OP>
  static TensorPair reduceIdxDim(const Tensor& t, int64_t dim, bool keepDims);
};

template <typename T, typename OP, typename IndexFunc>
void ReducerCuda::reduceMerge(T* values, const T* input, const Options& options, int64_t n, int64_t m) {
  using ComputeT = typename cuda::CudaComputeType<T>::type;
  const auto blockSize = cuda::getKernelBlockSize(options.device_.index);
  const auto stream = cuda::getCurrentCUDAStream(options.device_.index).stream;

  auto blocks = cuda::getKernelGridSize(blockSize, n);
  Tensor tmp({m * blocks}, Options(options.device_, TypeToDType_v<ComputeT>));
  ComputeT* tmpPtr = tmp.dataPtr<ComputeT>();

  // first round
  kReduceMerge<ComputeT, T, OP, IndexFunc><<<m * blocks, blockSize, 0, stream>>>(tmpPtr, input, n, m);
  CUDA_KERNEL_CHECK();

  // merge
  while (blocks > 1) {
    const auto currBlocks = blocks;
    blocks = cuda::getKernelGridSize(blockSize, currBlocks);
    kReduceMerge<ComputeT, ComputeT, OP, IndexFunc>
        <<<m * blocks, blockSize, 0, stream>>>(tmpPtr, tmpPtr, currBlocks, m);
    CUDA_KERNEL_CHECK();
  }
  tmp = tmp.to(options.dtype_);
  Storage::copyOnDevice(values, tmpPtr, m * sizeof(T), options.device_);
}

template <typename T, typename OP, typename IndexFunc>
void ReducerCuda::reduceIdxMerge(T* values, int64_t* indices, const T* input, const Options& options, int64_t n,
                                 int64_t m) {
  using ComputeT = typename cuda::CudaComputeType<T>::type;
  const auto blockSize = cuda::getKernelBlockSize(options.device_.index);
  const auto stream = cuda::getCurrentCUDAStream(options.device_.index).stream;

  auto blocks = cuda::getKernelGridSize(blockSize, n);

  Tensor tmpValues({m * blocks}, Options(options.device_, TypeToDType_v<ComputeT>));
  Tensor tmpIndices({m * blocks}, Options(options.device_, DType::Int64));
  ComputeT* tmpValuesPtr = tmpValues.dataPtr<ComputeT>();
  int64_t* tmpIndicesPtr = tmpIndices.dataPtr<int64_t>();

  // first round
  kReduceIdxMerge<ComputeT, T, OP, IndexFunc>
      <<<m * blocks, blockSize, 0, stream>>>(tmpValuesPtr, tmpIndicesPtr, input, nullptr, n, m);
  CUDA_KERNEL_CHECK();

  // merge
  while (blocks > 1) {
    const auto currBlocks = blocks;
    blocks = cuda::getKernelGridSize(blockSize, currBlocks);
    kReduceIdxMerge<ComputeT, ComputeT, OP, IndexFunc>
        <<<m * blocks, blockSize, 0, stream>>>(tmpValuesPtr, tmpIndicesPtr, tmpValuesPtr, tmpIndicesPtr, currBlocks, m);
    CUDA_KERNEL_CHECK();
  }
  if (values) {
    tmpValues = tmpValues.to(options.dtype_);
    Storage::copyOnDevice(values, tmpValuesPtr, m * sizeof(T), options.device_);
  }
  if (indices) {
    Storage::copyOnDevice(indices, tmpIndicesPtr, m * static_cast<int64_t>(sizeof(int64_t)), options.device_);
  }
}

template <typename T, typename OP>
void ReducerCuda::reduceDimFirst(T* values, const T* input, const Options& options, int64_t n, int64_t m) {
  // faster
  auto tmp = Tensor::empty({m * n}, options);
  cudaTranspose2d(tmp.dataPtr<T>(), input, m, n, options.device_);
  reduceMerge<T, OP, CudaReduceIndexLastDim>(values, tmp.dataPtr<T>(), options, n, m);

  // slower than transpose
  // reduceMerge<T, OP, CudaReduceIndexFirstDim>(values, input, options, n, m);
}

template <typename T, typename OP>
void ReducerCuda::reduceDimLast(T* values, const T* input, const Options& options, int64_t n, int64_t m) {
  reduceMerge<T, OP, CudaReduceIndexLastDim>(values, input, options, n, m);
}

template <typename T, typename OP>
void ReducerCuda::reduceIdxDimFirst(T* values, int64_t* indices, const T* input, const Options& options, int64_t n,
                                    int64_t m) {
  // faster
  auto tmp = Tensor::empty({m * n}, options);
  cudaTranspose2d(tmp.dataPtr<T>(), input, m, n, options.device_);
  reduceIdxMerge<T, OP, CudaReduceIndexLastDim>(values, indices, tmp.dataPtr<T>(), options, n, m);

  // slower than transpose
  // reduceIdxMerge<T, OP, CudaReduceIndexFirstDim>(values, indices, input, options, n, m);
}

template <typename T, typename OP>
void ReducerCuda::reduceIdxDimLast(T* values, int64_t* indices, const T* input, const Options& options, int64_t n,
                                   int64_t m) {
  reduceIdxMerge<T, OP, CudaReduceIndexLastDim>(values, indices, input, options, n, m);
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
  auto values = Tensor::empty(retShape, t.options().noGrad());

  const auto blockSize = cuda::getKernelBlockSize(t.device().index);

  // first dim
  if (dim == 0) {
    const auto dimSize = t.shape().front();
    if (static_cast<uint64_t>(dimSize) < blockSize) {
      auto params = cuda::getKernelLaunchParams(t.device().index, t.numel());
      CUDA_LAUNCH_KERNEL((kReduceDimFirstOrLast<T, OP, true>), params, values.dataPtr<T>(), t.dataPtr<T>(), dimSize,
                         values.numel());
    } else {
      reduceDimFirst<T, OpCudaReduceSum>(values.dataPtr<T>(), t.dataPtr<T>(), t.options(), dimSize, values.numel());
    }
  } else if (dim == t.dim() - 1) {
    // last dim
    const auto dimSize = t.shape().back();
    if (static_cast<uint64_t>(dimSize) < blockSize) {
      auto params = cuda::getKernelLaunchParams(t.device().index, t.numel());
      CUDA_LAUNCH_KERNEL((kReduceDimFirstOrLast<T, OP, false>), params, values.dataPtr<T>(), t.dataPtr<T>(), dimSize,
                         values.numel());
    } else {
      reduceDimLast<T, OpCudaReduceSum>(values.dataPtr<T>(), t.dataPtr<T>(), t.options(), dimSize, values.numel());
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
    values.reshape_(shapeKeepDims);
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
  auto values = Tensor::empty(retShape, t.options().noGrad());
  auto indices = Tensor::empty(retShape, t.options().noGrad().indices());

  const auto blockSize = cuda::getKernelBlockSize(t.device().index);

  if (dim == 0) {
    // first dim
    const auto dimSize = t.shape().front();
    if (static_cast<uint64_t>(dimSize) < blockSize) {
      auto params = cuda::getKernelLaunchParams(t.device().index, t.numel());
      CUDA_LAUNCH_KERNEL((kReduceIdxDimFirstOrLast<T, OP, true>), params, values.dataPtr<T>(),
                         indices.dataPtr<int64_t>(), t.dataPtr<T>(), dimSize, values.numel());
    } else {
      reduceIdxDimFirst<T, OP>(values.dataPtr<T>(), indices.dataPtr<int64_t>(), t.dataPtr<T>(), t.options(), dimSize,
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
      reduceIdxDimLast<T, OP>(values.dataPtr<T>(), indices.dataPtr<int64_t>(), t.dataPtr<T>(), t.options(), dimSize,
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
    values.reshape_(shapeKeepDims);
    indices.reshape_(shapeKeepDims);
  }
  return {values, indices};
}

template <typename T, typename OP>
Tensor reduceOpAllCudaImpl(const Tensor& t) {
  using CudaT = typename cuda::CudaTypeCast<T>::type;
  if (t.isScalar()) {
    return t;
  }
  auto ret = Tensor::scalar(0, t.options().noGrad());
  const CudaT* tPtr = t.dataPtr<CudaT>();
  CudaT* retPtr = ret.dataPtr<CudaT>();
  ReducerCuda::reduceMerge<CudaT, OP, CudaReduceIndexAll>(retPtr, tPtr, t.options(), t.numel());
  return ret;
}

template <typename T, typename OP>
Tensor reduceOpArgMinMaxCudaImpl(const Tensor& t) {
  using CudaT = typename cuda::CudaTypeCast<T>::type;
  auto ret = Tensor::scalar(0, t.options().noGrad().indices());
  if (t.isScalar()) {
    return ret;
  }
  const CudaT* tPtr = t.dataPtr<CudaT>();
  auto* retPtr = ret.dataPtr<int64_t>();
  ReducerCuda::reduceIdxMerge<CudaT, OP, CudaReduceIndexAll>(nullptr, retPtr, tPtr, t.options(), t.numel());
  return ret;
}

template <typename T, typename OP>
TensorPair reduceOpMinMaxDimCudaImpl(const Tensor& t, int64_t dim, bool keepDims = false) {
  using CudaT = typename cuda::CudaTypeCast<T>::type;
  if (t.isScalar()) {
    return {t, Tensor::scalar(0, t.options().noGrad().indices())};
  }
  return ReducerCuda::reduceIdxDim<CudaT, OP>(t, dim, keepDims);
}

template <typename T>
Tensor reduceOpSumDimsCudaImpl(const Tensor& t, const IntArrayView dims, bool keepDims = false) {
  using CudaT = typename cuda::CudaTypeCast<T>::type;
  if (t.isScalar()) {
    return t;
  }
  ASSERT(!dims.empty());

  if (dims.size() == 1) {
    return ReducerCuda::reduceDim<CudaT, OpCudaReduceSum>(t, dims[0], keepDims);
  }

  DimArray<int64_t> inAxis{};
  int64_t reduceCnt = 1;
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
    reduceCnt *= t.shape(d);
  }

  const auto retShape = getReduceShape(t, inAxis, keepDims);
  auto ret = Tensor::zeros(retShape, t.options().noGrad());

  auto ctxT = cuda::getTensorCudaCtx(t);

  constexpr int blockSize = 512;
  ASSERT(blockSize <= cuda::getMaxThreadsPerBlock(t.device().index));

  dim3 grid(t.numel());
  dim3 block(blockSize);
  const auto stream = cuda::getCurrentCUDAStream(t.device().index).stream;
  kReduceMultiDimSum<CudaT, blockSize>
      <<<grid, block, 0, stream>>>(ret.dataPtr<CudaT>(), ctxT, inAxis, reduceCnt, t.numel());
  CUDA_KERNEL_CHECK();
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
  using CudaT = typename cuda::CudaTypeCast<T>::type;
  if (t.isScalar()) {
    return t;
  }
  auto ret = reduceOpAllCudaImpl<CudaT, OpCudaReduceSum>(t);
  const auto r = 1.f / static_cast<float>(t.numel());
  op::mulInplace(ret, Tensor::scalar(r, ret.options().noGrad()));
  return ret;
}

template <typename T>
Tensor reduceOpMeanDimsCudaImpl(const Tensor& t, const IntArrayView dims, bool keepDims = false) {
  using CudaT = typename cuda::CudaTypeCast<T>::type;
  if (t.isScalar()) {
    return t;
  }
  ASSERT(!dims.empty());
  auto ret = reduceOpSumDimsCudaImpl<CudaT>(t, dims, keepDims);
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
  using CudaT = typename cuda::CudaTypeCast<T>::type;
  if (t.isScalar()) {
    return {Tensor::scalar(0, t.options().noGrad()), t};
  }
  const auto meanVal = op::mean(t);
  auto squaredDiff = Tensor::empty({t.numel()}, t.options().noGrad());

  const CudaT* tPtr = t.dataPtr<CudaT>();
  const CudaT* meanPtr = meanVal.dataPtr<CudaT>();
  CudaT* squaredDiffPtr = squaredDiff.dataPtr<CudaT>();

  auto params = cuda::getKernelLaunchParams(t.device().index, t.numel());
  CUDA_LAUNCH_KERNEL((kSquaredDiff<CudaT>), params, squaredDiffPtr, tPtr, meanPtr, t.numel());

  auto varVal = Tensor::empty({}, t.options().noGrad());
  CudaT* varPtr = varVal.dataPtr<CudaT>();
  ReducerCuda::reduceMerge<CudaT, OpCudaReduceSum, CudaReduceIndexAll>(varPtr, squaredDiffPtr, t.options(), t.numel());

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
  using CudaT = typename cuda::CudaTypeCast<T>::type;
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
  auto varVal = Tensor::zeros(retShape, t.options().noGrad());

  auto ctxT = cuda::getTensorCudaCtx(t);
  const CudaT* meanPtr = meanVal.dataPtr<CudaT>();
  CudaT* varPtr = varVal.dataPtr<CudaT>();

  auto params = cuda::getKernelLaunchParams(t.device().index, varVal.numel());
  CUDA_LAUNCH_KERNEL((kReduceMultiDimVar<CudaT>), params, varPtr, ctxT, meanPtr, inAxis, varVal.numel());

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
