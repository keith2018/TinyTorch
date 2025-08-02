/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <numeric>

#include "OpTransform.h"
#include "Utils/CUDAUtils.h"

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

}  // namespace tinytorch::op
