/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sort.h>

#include <numeric>

#include "OpTransform.h"
#include "Utils/CUDAUtils.h"
#include "Utils/RandomGenerator.h"

namespace tinytorch::op {

#define TRANSPOSE_TILE_DIM 32

__device__ __forceinline__ void cudaGetSubIndices(int64_t* subIndices, const int64_t* shape,
                                                  const int64_t* const* indices, const int64_t idx, const int64_t len) {
#pragma unroll
  for (int64_t i = 0; i < len; i++) {
    const auto ind = indices[i][idx];
    subIndices[i] = ind >= 0 ? ind : ind + shape[i];
  }
}

__device__ __forceinline__ int64_t cudaIndicesToOffset(const int64_t* strides, const int64_t* indices,
                                                       const int64_t dimCount) {
  int64_t offset = 0;
#pragma unroll
  for (int64_t i = 0; i < dimCount; i++) {
    offset += indices[i] * strides[i];
  }
  return offset;
}

template <typename T>
__global__ void kPermute(const cuda::TensorCudaCtx ret, const cuda::TensorCudaCtx self, const DimArray<int64_t> dims,
                         const int64_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    int64_t srcIndex = 0;
    auto offset = static_cast<int64_t>(index);
#pragma unroll
    for (int64_t d = 0; d < self.ndim; d++) {
      srcIndex += (offset / ret.strides[d]) * self.strides[dims.data[d]];
      offset %= ret.strides[d];
    }

    const auto* selfPtr = static_cast<T*>(self.data);
    auto* retPtr = static_cast<T*>(ret.data);
    retPtr[index] = selfPtr[srcIndex];
  }
}

template <typename T>
__global__ void kTranspose(T* out, const T* in, const int64_t width, const int64_t height) {
  __shared__ T tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM + 1];  // +1 to avoid bank conflicts

  auto x = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.x;
  auto y = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.y;

  if (x < width && y < height) {
    tile[threadIdx.y][threadIdx.x] = in[y * width + x];
  }
  __syncthreads();

  x = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.x;
  y = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.y;

  if (x < height && y < width) {
    out[y * height + x] = tile[threadIdx.x][threadIdx.y];
  }
}

template <typename T>
__global__ void kIndex(T* ret, const cuda::TensorCudaCtx self, const DimArray<const int64_t*> indices,
                       const int64_t dimStride, const int64_t len, const int64_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    DimArray<int64_t> subIndices{};
    cudaGetSubIndices(subIndices.data, self.shape, indices.data, index, len);
    const int64_t dataIdx = cudaIndicesToOffset(self.strides, subIndices.data, self.ndim);
    const auto* tPtr = static_cast<T*>(self.data);
    memcpy(&ret[dimStride * index], &tPtr[dataIdx], dimStride * sizeof(T));
  }
}

template <typename T>
__global__ void kIndex2D(T* ret, const T* self, const int64_t* indices0, const int64_t* indices1, const int64_t dim0,
                         const int64_t dim1, const int64_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    auto idx0 = indices0[index];
    auto idx1 = indices1[index];
    if (idx0 < 0) idx0 += dim0;
    if (idx1 < 0) idx1 += dim1;
    const auto dataIdx = idx0 * dim1 + idx1;
    ret[index] = self[dataIdx];
  }
}

template <typename T>
__global__ void kIndexPutScalar(cuda::TensorCudaCtx self, DimArray<const int64_t*> indices, const int64_t dimStride,
                                const int64_t len, const T* val, const int64_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    DimArray<int64_t> subIndices{};
    cudaGetSubIndices(subIndices.data, self.shape, indices.data, index, len);
    const int64_t dataIdx = cudaIndicesToOffset(self.strides, subIndices.data, self.ndim);
    auto* tPtr = static_cast<T*>(self.data);
#pragma unroll
    for (int64_t i = 0; i < dimStride; i++) {
      tPtr[dataIdx + i] = *val;
    }
  }
}

template <typename T>
__global__ void kIndexPutScalar2D(T* self, const int64_t* indices0, const int64_t* indices1, const int64_t dim0,
                                  const int64_t dim1, const T* val, const int64_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    auto idx0 = indices0[index];
    auto idx1 = indices1[index];
    if (idx0 < 0) idx0 += dim0;
    if (idx1 < 0) idx1 += dim1;
    const auto dataIdx = idx0 * dim1 + idx1;
    self[dataIdx] = *val;
  }
}

template <typename T>
__global__ void kIndexPut(cuda::TensorCudaCtx self, DimArray<const int64_t*> indices, const int64_t dimStride,
                          const int64_t len, const T* val, const int64_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    DimArray<int64_t> subIndices{};
    cudaGetSubIndices(subIndices.data, self.shape, indices.data, index, len);
    const int64_t dataIdx = cudaIndicesToOffset(self.strides, subIndices.data, self.ndim);
    auto* tPtr = static_cast<T*>(self.data);
    memcpy(&tPtr[dataIdx], &val[dimStride * index], dimStride * sizeof(T));
  }
}

template <typename T>
__global__ void kIndexPut2D(T* self, const int64_t* indices0, const int64_t* indices1, const int64_t dim0,
                            const int64_t dim1, const T* val, const int64_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    auto idx0 = indices0[index];
    auto idx1 = indices1[index];
    if (idx0 < 0) idx0 += dim0;
    if (idx1 < 0) idx1 += dim1;
    const auto dataIdx = idx0 * dim1 + idx1;
    self[dataIdx] = val[index];
  }
}

template <typename T, bool LOWER>
__global__ void kTriangle(T* ret, const T* t, const int64_t rows, const int64_t cols, const int64_t diagonal) {
  auto i = blockIdx.y * blockDim.y + threadIdx.y;
  auto j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < rows && j < cols) {
    const auto index = i * cols + j;
    if ((LOWER && j <= i + diagonal) || (!LOWER && j >= i + diagonal)) {
      ret[index] = t[index];
    } else {
      ret[index] = 0;
    }
  }
}

template <typename T>
__global__ void kPrepareProbabilities(const T* prob, float* floatProb, int64_t n, int64_t batch) {
  int64_t batchIdx = blockIdx.x;
  if (batchIdx >= batch) {
    return;
  }

  const T* batchProb = prob + batchIdx * n;
  float* batchFloatProb = floatProb + batchIdx * n;

  for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
    batchFloatProb[i] = static_cast<float>(batchProb[i]);
  }
}

inline __global__ void kMultinomialWithReplacement(const float* cdf, int64_t* output, int64_t batch, int64_t n,
                                                   int64_t numSamples, uint64_t seed, uint64_t seq) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t totalSamples = batch * numSamples;
  if (idx >= totalSamples) {
    return;
  }

  int64_t batchIdx = idx / numSamples;

  curandState state;
  curand_init(seed, idx, seq, &state);

  const float* batchCdf = cdf + batchIdx * n;
  float total = batchCdf[n - 1];

  if (total > 0) {
    float r = curand_uniform(&state) * total;
    int64_t left = 0, right = n - 1;
    while (left < right) {
      int64_t mid = left + (right - left) / 2;
      if (batchCdf[mid] < r)
        left = mid + 1;
      else
        right = mid;
    }
    output[idx] = left;
  } else {
    output[idx] = 0;
  }
}

template <typename T>
__global__ void kMultinomialNoReplacement(const T* prob, int64_t* output, float* workspace, int64_t batch, int64_t n,
                                          int64_t numSamples, uint64_t seed, uint64_t seq) {
  int64_t batchIdx = blockIdx.x;
  if (batchIdx >= batch) {
    return;
  }

  const T* batchProb = prob + batchIdx * n;
  float* batchWorkspace = workspace + batchIdx * n;

  for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
    batchWorkspace[i] = static_cast<float>(batchProb[i]);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    curandState state;
    curand_init(seed, batchIdx, seq, &state);

    for (int64_t s = 0; s < numSamples; s++) {
      float total = 0.f;
      for (int64_t i = 0; i < n; ++i) total += batchWorkspace[i];

      int64_t outputIdx = batchIdx * numSamples + s;
      if (total > 0) {
        float r = curand_uniform(&state) * total;
        float cumSum = 0.f;
        int64_t selectedIdx = 0;
        for (int64_t i = 0; i < n; i++) {
          cumSum += batchWorkspace[i];
          if (cumSum >= r) {
            selectedIdx = i;
            break;
          }
        }
        output[outputIdx] = selectedIdx;
        batchWorkspace[selectedIdx] = 0.f;
      } else {
        output[outputIdx] = 0;
      }
    }
  }
}

struct OpCudaGather {
  template <typename T>
  __device__ static void apply(const cuda::TensorCudaCtx& out, const cuda::TensorCudaCtx& self,
                               const cuda::TensorCudaCtx&, int64_t i, int64_t offset) {
    const T* selfPtr = static_cast<const T*>(self.data);
    T* outPtr = static_cast<T*>(out.data);
    outPtr[i] = selfPtr[offset];
  }
};

struct OpCudaScatter {
  template <typename T>
  __device__ static void apply(const cuda::TensorCudaCtx& out, const cuda::TensorCudaCtx&,
                               const cuda::TensorCudaCtx& src, int64_t i, int64_t offset) {
    T* outPtr = static_cast<T*>(out.data);
    const T* srcPtr = static_cast<const T*>(src.data);
    outPtr[offset] = srcPtr[i];
  }
};

template <typename T, typename OP>
__global__ void kGatherScatter(const cuda::TensorCudaCtx out, const cuda::TensorCudaCtx self,
                               const cuda::TensorCudaCtx index, const cuda::TensorCudaCtx src,
                               const DimArray<int64_t> indexStrides, const int64_t dim, const int64_t n) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }

  int64_t remain = i;
  int64_t coord[MAX_TENSOR_DIM];
#pragma unroll
  for (int64_t d = 0; d < index.ndim; d++) {
    coord[d] = remain / indexStrides.data[d];
    remain = remain % indexStrides.data[d];
  }

  const auto* idxPtr = static_cast<const int64_t*>(index.data);
  int64_t idx = idxPtr[i];
  if (idx < 0) idx += self.shape[dim];
  coord[dim] = idx;

  int64_t offset = 0;
#pragma unroll
  for (int64_t d = 0; d < self.ndim; d++) {
    offset += coord[d] * self.strides[d];
  }

  OP::template apply<T>(out, self, src, i, offset);
}

template <typename T>
void cudaTranspose2d(T* out, const T* in, int64_t width, int64_t height, const Device& device) {
  dim3 blockSize(TRANSPOSE_TILE_DIM, TRANSPOSE_TILE_DIM);
  dim3 gridSize((width + TRANSPOSE_TILE_DIM - 1) / TRANSPOSE_TILE_DIM,
                (height + TRANSPOSE_TILE_DIM - 1) / TRANSPOSE_TILE_DIM);

  const auto stream = cuda::getCurrentCUDAStream(device.index).stream;
  kTranspose<<<gridSize, blockSize, 0, stream>>>(out, in, width, height);
  CUDA_KERNEL_CHECK();
}

template <typename T>
Tensor permuteOpCudaImpl(const Tensor& self, const IntArrayView dims) {
  ASSERT(dims.size() == self.dim());
  auto retShape = SizeVector(self.shape());
  reorderIndices(retShape.data(), self.dim(), dims);
  auto ret = Tensor::empty(retShape, self.options().noGrad());

  auto ctxSelf = cuda::getTensorCudaCtx(self);
  auto ctxRet = cuda::getTensorCudaCtx(ret);

  DimArray<int64_t> dimsData{};
  for (auto i = 0; i < self.dim(); i++) {
    dimsData.data[i] = dims[i];
  }

  auto params = cuda::getKernelLaunchParams(self.device().index, self.numel());
  CUDA_LAUNCH_KERNEL(kPermute<T>, params, ctxRet, ctxSelf, dimsData, self.numel());
  return ret;
}

template <typename T>
Tensor permuteAllOpCudaImpl(const Tensor& self) {
  SizeVector dims(self.dim());
  std::iota(dims.begin(), dims.end(), 0);
  std::reverse(dims.begin(), dims.end());
  return permuteOpCudaImpl<T>(self, dims);
}

template <typename T>
Tensor transposeOpCudaImpl(const Tensor& self, int64_t dim0, int64_t dim1) {
  if (dim0 < 0) {
    dim0 += self.dim();
  }
  if (dim1 < 0) {
    dim1 += self.dim();
  }
  if (dim0 < 0 || dim0 >= self.dim() || dim1 < 0 || dim1 >= self.dim()) {
    LOGE("Invalid axis: (%lld, %lld)", dim0, dim1);
    ASSERT(false);
    return {};
  }

  SizeVector dims(self.dim());
  std::iota(dims.begin(), dims.end(), 0);
  dims[dim0] = dim1;
  dims[dim1] = dim0;
  return permuteOpCudaImpl<T>(self, dims);
}

template <typename T>
Tensor transpose2dOpCudaImpl(const Tensor& self) {
  ASSERT(self.dim() == 2);

  SizeVector retShape = {self.shape(1), self.shape(0)};
  auto ret = Tensor::empty(retShape, self.options().noGrad());
  cudaTranspose2d(ret.dataPtr<T>(), self.dataPtr<T>(), retShape[0], retShape[1], self.device());
  return ret;
}

template <typename T>
Tensor indexAdvanceOpCudaImpl(const Tensor& self, ArrayView<Tensor> indices) {
  auto len = static_cast<int64_t>(indices.size());
  auto firstDim = indices[0].numel();
  auto dimStride = self.stride(len - 1);
  SizeVector retShape(indices[0].shape());
  for (auto i = len; i < self.dim(); i++) {
    retShape.pushBack(self.shape(i));
  }
  auto ret = Tensor::empty(retShape, self.options().noGrad());
  const auto* selfPtr = self.dataPtr<T>();
  auto* retPtr = ret.dataPtr<T>();

  // 2D
  if (self.dim() == 2 && len == 2) {
    auto dim0 = self.shape(0);
    auto dim1 = self.shape(1);
    ASSERT(indices[0].dtype() == DType::Int64);
    ASSERT(indices[1].dtype() == DType::Int64);
    auto* idx0Ptr = indices[0].dataPtr<int64_t>();
    auto* idx1Ptr = indices[1].dataPtr<int64_t>();
    auto params = cuda::getKernelLaunchParams(self.device().index, firstDim);
    CUDA_LAUNCH_KERNEL(kIndex2D<T>, params, retPtr, selfPtr, idx0Ptr, idx1Ptr, dim0, dim1, firstDim);
    return ret;
  }

  DimArray<const int64_t*> indicesData{};
  for (int64_t i = 0; i < len; i++) {
    ASSERT(indices[i].dtype() == DType::Int64);
    indicesData.data[i] = indices[i].dataPtr<int64_t>();
  }
  auto ctxSelf = cuda::getTensorCudaCtx(self);
  auto params = cuda::getKernelLaunchParams(self.device().index, firstDim);
  CUDA_LAUNCH_KERNEL(kIndex<T>, params, retPtr, ctxSelf, indicesData, dimStride, len, firstDim);
  return ret;
}

template <typename T>
void indexPutAdvanceOpCudaImpl(Tensor& self, ArrayView<Tensor> indices, const Tensor& val) {
  auto len = static_cast<int64_t>(indices.size());
  auto firstDim = indices[0].numel();
  auto dimStride = self.stride(len - 1);

  T* selfPtr = self.dataPtr<T>();
  const T* valPtr = val.dataPtr<T>();

  // 2D
  if (self.dim() == 2 && len == 2) {
    auto dim0 = self.shape(0);
    auto dim1 = self.shape(1);
    ASSERT(indices[0].dtype() == DType::Int64);
    ASSERT(indices[1].dtype() == DType::Int64);
    auto* idx0Ptr = indices[0].dataPtr<int64_t>();
    auto* idx1Ptr = indices[1].dataPtr<int64_t>();
    auto params = cuda::getKernelLaunchParams(self.device().index, firstDim);
    if (val.isScalar()) {
      CUDA_LAUNCH_KERNEL(kIndexPutScalar2D<T>, params, selfPtr, idx0Ptr, idx1Ptr, dim0, dim1, valPtr, firstDim);
    } else {
      ASSERT(val.numel() == firstDim);
      CUDA_LAUNCH_KERNEL(kIndexPut2D<T>, params, selfPtr, idx0Ptr, idx1Ptr, dim0, dim1, valPtr, firstDim);
    }
    return;
  }

  DimArray<const int64_t*> indicesData{};
  for (int64_t i = 0; i < len; i++) {
    ASSERT(indices[i].dtype() == DType::Int64);
    indicesData.data[i] = indices[i].dataPtr<int64_t>();
  }
  auto ctxSelf = cuda::getTensorCudaCtx(self);
  auto params = cuda::getKernelLaunchParams(self.device().index, firstDim);
  if (val.isScalar()) {
    CUDA_LAUNCH_KERNEL(kIndexPutScalar<T>, params, ctxSelf, indicesData, dimStride, len, valPtr, firstDim);
  } else {
    ASSERT(val.numel() == dimStride * firstDim);
    CUDA_LAUNCH_KERNEL(kIndexPut<T>, params, ctxSelf, indicesData, dimStride, len, valPtr, firstDim);
  }
}

template <typename T, bool LOWER>
Tensor triangleOpCudaImpl(const Tensor& self, int64_t diagonal) {
  auto ret = Tensor::empty(self.shape(), self.options().noGrad());
  const auto rows = self.shape(0);
  const auto cols = self.shape(1);

  const T* selfPtr = self.dataPtr<T>();
  T* retPtr = ret.dataPtr<T>();

  dim3 blockSize(CUDA_WARP_SIZE, CUDA_WARP_SIZE);
  dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

  const auto stream = cuda::getCurrentCUDAStream(self.device().index).stream;
  kTriangle<T, LOWER><<<gridSize, blockSize, 0, stream>>>(retPtr, selfPtr, rows, cols, diagonal);
  CUDA_KERNEL_CHECK();
  return ret;
}

template <typename T>
Tensor trilOpCudaImpl(const Tensor& self, int64_t diagonal = 0) {
  return triangleOpCudaImpl<T, true>(self, diagonal);
}

template <typename T>
Tensor triuOpCudaImpl(const Tensor& self, int64_t diagonal = 0) {
  return triangleOpCudaImpl<T, false>(self, diagonal);
}

template <typename T>
TensorPair topkOpCudaImpl(const Tensor& self, int64_t k, int64_t dim, bool largest, bool sorted) {
  if (dim < 0) dim += self.dim();
  ASSERT(dim >= 0 && dim < self.dim());

  int64_t n = self.shape(dim);
  ASSERT(k > 0 && k <= n);

  SizeVector retShape(self.shape());
  retShape[dim] = k;
  Tensor values = Tensor::empty(retShape, self.options().noGrad());
  Tensor indices = Tensor::empty(retShape, self.options().noGrad().indices());

  int64_t outerSize = 1;
  int64_t innerSize = 1;
  for (int64_t i = 0; i < dim; i++) {
    outerSize *= self.shape(i);
  }
  for (int64_t i = dim + 1; i < self.dim(); i++) {
    innerSize *= self.shape(i);
  }

  const auto* selfPtr = self.dataPtr<T>();
  auto* valPtr = values.dataPtr<T>();
  auto* idxPtr = indices.dataPtr<int64_t>();

  const auto stream = cuda::getCurrentCUDAStream(self.device().index).stream;
  auto policy = thrust::cuda::par.on(stream);

  Storage tmpValues(static_cast<int64_t>(n * sizeof(T)), self.device());
  Storage tmpIndices(static_cast<int64_t>(n * sizeof(int64_t)), self.device());

  auto* tmpValuesPtr = tmpValues.dataPtr<T>();
  auto* tmpIndicesPtr = tmpIndices.dataPtr<int64_t>();

  for (int64_t o = 0; o < outerSize; o++) {
    for (int64_t in = 0; in < innerSize; in++) {
      int64_t base = o * n * innerSize + in;

      // init values & indices
      thrust::device_ptr<T> dstPtr = thrust::device_pointer_cast(tmpValuesPtr);
      auto inputIter =
          thrust::make_transform_iterator(thrust::counting_iterator<int64_t>(0),
                                          [=] __host__ __device__(int64_t i) { return selfPtr[base + i * innerSize]; });
      thrust::copy(policy, inputIter, inputIter + n, dstPtr);
      thrust::sequence(policy, thrust::device_pointer_cast(tmpIndicesPtr),
                       thrust::device_pointer_cast(tmpIndicesPtr + n), 0);

      // sort
      if (largest) {
        thrust::sort_by_key(policy, thrust::device_pointer_cast(tmpValuesPtr),
                            thrust::device_pointer_cast(tmpValuesPtr + n), thrust::device_pointer_cast(tmpIndicesPtr),
                            thrust::greater<T>());
      } else {
        thrust::sort_by_key(policy, thrust::device_pointer_cast(tmpValuesPtr),
                            thrust::device_pointer_cast(tmpValuesPtr + n), thrust::device_pointer_cast(tmpIndicesPtr),
                            thrust::less<T>());
      }

      // copy results
      auto outputValIter = thrust::make_transform_iterator(
          thrust::counting_iterator<int64_t>(0),
          [=] __host__ __device__(int64_t i) -> T& { return valPtr[(o * k + i) * innerSize + in]; });

      auto outputIdxIter = thrust::make_transform_iterator(
          thrust::counting_iterator<int64_t>(0),
          [=] __host__ __device__(int64_t i) -> int64_t& { return idxPtr[(o * k + i) * innerSize + in]; });
      thrust::copy(policy, thrust::device_pointer_cast(tmpValuesPtr), thrust::device_pointer_cast(tmpValuesPtr + k),
                   outputValIter);
      thrust::copy(policy, thrust::device_pointer_cast(tmpIndicesPtr), thrust::device_pointer_cast(tmpIndicesPtr + k),
                   outputIdxIter);
    }
  }

  return {values, indices};
}

template <typename T>
Tensor multinomialOpCudaImpl(const Tensor& self, int64_t nSamples, bool replacement) {
  ASSERT(self.dim() == 1 || self.dim() == 2);
  ASSERT(self.dtype() == DType::Float32);

  int64_t batch = (self.dim() == 2) ? self.shape(0) : 1;
  int64_t n = (self.dim() == 2) ? self.shape(1) : self.shape(0);

  SizeVector retShape;
  if (self.dim() == 2) {
    retShape = {batch, nSamples};
  } else {
    retShape = {nSamples};
  }

  Tensor ret = Tensor::empty(retShape, self.options().noGrad().indices());
  if (n == 0) {
    return ret;
  }

  const T* selfPtr = self.dataPtr<T>();
  auto* retPtr = ret.dataPtr<int64_t>();

  auto seed = RandomGeneratorCUDA::getSeed();
  auto seq = RandomGeneratorCUDA::nextSequence();

  const auto stream = cuda::getCurrentCUDAStream(self.device().index).stream;
  auto blockSize = cuda::getKernelBlockSize(self.device().index);

  Storage tmpProb(static_cast<int64_t>(batch * n * sizeof(float)), self.device());
  auto* tmpPtr = tmpProb.dataPtr<float>();

  if (replacement) {
    kPrepareProbabilities<<<batch, blockSize, 0, stream>>>(selfPtr, tmpPtr, n, batch);
    CUDA_KERNEL_CHECK();

    for (int64_t b = 0; b < batch; b++) {
      thrust::device_ptr<float> batchPtr(tmpPtr + b * n);
      thrust::inclusive_scan(thrust::cuda::par.on(stream), batchPtr, batchPtr + n, batchPtr);
    }

    int64_t totalSamples = batch * nSamples;
    auto gridSize = (totalSamples + blockSize - 1) / blockSize;
    kMultinomialWithReplacement<<<gridSize, blockSize, 0, stream>>>(tmpPtr, retPtr, batch, n, nSamples, seed, seq);
    CUDA_KERNEL_CHECK();
  } else {
    kMultinomialNoReplacement<<<batch, blockSize, 0, stream>>>(selfPtr, retPtr, tmpPtr, batch, n, nSamples, seed, seq);
    CUDA_KERNEL_CHECK();
  }
  return ret;
}

template <typename T>
TensorPair sortOpCudaImpl(const Tensor& self, int64_t dim, bool descending) {
  if (dim < 0) {
    dim += self.dim();
  }
  ASSERT(dim >= 0 && dim < self.dim());

  SizeVector retShape(self.shape());
  Tensor values = Tensor::empty(retShape, self.options().noGrad());
  Tensor indices = Tensor::empty(retShape, self.options().noGrad().indices());

  int64_t outer = 1, inner = 1, n = self.shape(dim);
  for (int64_t i = 0; i < dim; i++) {
    outer *= self.shape(i);
  }
  for (int64_t i = dim + 1; i < self.dim(); i++) {
    inner *= self.shape(i);
  }

  const T* selfPtr = self.dataPtr<T>();
  T* valPtr = values.dataPtr<T>();
  auto* idxPtr = indices.dataPtr<int64_t>();

  const auto stream = cuda::getCurrentCUDAStream(self.device().index).stream;

  thrust::copy(thrust::cuda::par.on(stream), selfPtr, selfPtr + self.numel(), valPtr);

  auto initIndices = [=] __host__ __device__(int64_t idx) {
    int64_t i = (idx % (n * inner)) / inner;
    return i;
  };
  thrust::transform(thrust::cuda::par.on(stream), thrust::make_counting_iterator<int64_t>(0),
                    thrust::make_counting_iterator<int64_t>(outer * n * inner), thrust::device_pointer_cast(idxPtr),
                    initIndices);

  for (int64_t seg = 0; seg < outer * inner; seg++) {
    int64_t o = seg / inner;
    int64_t in = seg % inner;

    T* valSegmentStart = valPtr + o * n * inner + in;
    int64_t* idxSegmentStart = idxPtr + o * n * inner + in;

    auto valIter = thrust::make_permutation_iterator(
        thrust::device_pointer_cast(valSegmentStart),
        thrust::make_transform_iterator(thrust::make_counting_iterator<int64_t>(0),
                                        [inner] __host__ __device__(int64_t i) { return i * inner; }));
    auto idxIter = thrust::make_permutation_iterator(
        thrust::device_pointer_cast(idxSegmentStart),
        thrust::make_transform_iterator(thrust::make_counting_iterator<int64_t>(0),
                                        [inner] __host__ __device__(int64_t i) { return i * inner; }));

    if (descending) {
      thrust::stable_sort_by_key(thrust::cuda::par.on(stream), valIter, valIter + n, idxIter, thrust::greater<T>());
    } else {
      thrust::stable_sort_by_key(thrust::cuda::par.on(stream), valIter, valIter + n, idxIter, thrust::less<T>());
    }
  }

  return {values, indices};
}

template <typename T>
Tensor cumsumOpCudaImpl(const Tensor& self, int64_t dim) {
  if (dim < 0) {
    dim += self.dim();
  }
  ASSERT(dim >= 0 && dim < self.dim());

  Tensor ret = Tensor::empty(self.shape(), self.options().noGrad());
  const T* selfPtr = self.dataPtr<T>();
  T* retPtr = ret.dataPtr<T>();

  int64_t outer = 1, inner = 1, n = self.shape(dim);
  for (int64_t i = 0; i < dim; i++) {
    outer *= self.shape(i);
  }
  for (int64_t i = dim + 1; i < self.dim(); i++) {
    inner *= self.shape(i);
  }

  const auto stream = cuda::getCurrentCUDAStream(self.device().index).stream;

  if (inner == 1) {
    for (int64_t o = 0; o < outer; ++o) {
      int64_t offset = o * n;
      thrust::inclusive_scan(thrust::cuda::par.on(stream), selfPtr + offset, selfPtr + offset + n, retPtr + offset);
    }
  } else {
    thrust::for_each_n(thrust::cuda::par.on(stream), thrust::counting_iterator<int64_t>(0), outer * inner,
                       [=] __host__ __device__(int64_t row) {
                         int64_t o = row / inner;
                         int64_t inIdx = row % inner;
                         int64_t base = o * n * inner + inIdx;

                         T sum = 0;
                         for (int64_t i = 0; i < n; ++i) {
                           sum += selfPtr[base + i * inner];
                           retPtr[base + i * inner] = sum;
                         }
                       });
  }
  return ret;
}

template <typename T, typename OP>
void gatherScatterCudaImpl(const Tensor& self, int64_t dim, const Tensor& index, const Tensor* src, Tensor& out) {
  if (dim < 0) {
    dim += self.dim();
  }
  ASSERT(dim >= 0 && dim < self.dim());
  ASSERT(index.dim() == self.dim());

  DimArray<int64_t> indexStrides{};
  int64_t stride = 1;
  for (int64_t d = index.dim() - 1; d >= 0; d--) {
    indexStrides.data[d] = stride;
    stride *= index.shape()[d];
  }

  auto ctxOut = cuda::getTensorCudaCtx(out);
  auto ctxSelf = cuda::getTensorCudaCtx(self);
  auto ctxIndex = cuda::getTensorCudaCtx(index);
  auto ctxSrc = src ? cuda::getTensorCudaCtx(*src) : cuda::TensorCudaCtx{};

  auto params = cuda::getKernelLaunchParams(self.device().index, index.numel());
  CUDA_LAUNCH_KERNEL((kGatherScatter<T, OP>), params, ctxOut, ctxSelf, ctxIndex, ctxSrc, indexStrides, dim,
                     index.numel());
}

template <typename T>
Tensor gatherOpCudaImpl(const Tensor& self, int64_t dim, const Tensor& index) {
  Tensor ret = Tensor::empty(index.shape(), self.options().noGrad());
  gatherScatterCudaImpl<T, OpCudaGather>(self, dim, index, nullptr, ret);
  return ret;
}

template <typename T>
Tensor scatterOpCudaImpl(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
  Tensor ret = self.clone();
  ret.setRequiresGrad(false);
  ret.copyOnWrite();
  gatherScatterCudaImpl<T, OpCudaScatter>(ret, dim, index, &src, ret);
  return ret;
}

template <typename T>
void scatterOpInplaceCudaImpl(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
  self.copyOnWrite();
  gatherScatterCudaImpl<T, OpCudaScatter>(self, dim, index, &src, self);
}

}  // namespace tinytorch::op
