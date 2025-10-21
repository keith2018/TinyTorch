/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <cassert>
#include <numeric>

#include "OpFilling.h"
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
__global__ void kTranspose2D(T* out, const T* in, const int64_t width, const int64_t height) {
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
__global__ void kTransposeND(T* out, const T* in, int64_t ndim, int64_t dim0, int64_t dim1, int64_t n,
                             const DimArray<int64_t> outStrides, const DimArray<int64_t> inStrides) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    int64_t inIdx = 0;

    int64_t coords[MAX_TENSOR_DIM];
    int64_t tmpIdx = idx;

#pragma unroll
    for (int d = 0; d < ndim; d++) {
      coords[d] = tmpIdx / outStrides.data[d];
      tmpIdx %= outStrides.data[d];
    }

    int64_t tmp = coords[dim0];
    coords[dim0] = coords[dim1];
    coords[dim1] = tmp;

#pragma unroll
    for (int d = 0; d < ndim; d++) {
      inIdx += coords[d] * inStrides.data[d];
    }

    out[idx] = in[inIdx];
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
__global__ void kTriangle(T* ret, const T* t, const int64_t batch, const int64_t rows, const int64_t cols,
                          const int64_t diagonal, const int64_t matrixSize) {
  const int64_t b = blockIdx.z;
  const int64_t i = blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t j = blockIdx.x * blockDim.x + threadIdx.x;

  if (b < batch && i < rows && j < cols) {
    const int64_t index = b * matrixSize + i * cols + j;
    if ((LOWER && j <= i + diagonal) || (!LOWER && j >= i + diagonal)) {
      ret[index] = t[index];
    } else {
      ret[index] = static_cast<T>(0);
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
__global__ void kExpand(T* out, const T* in, int64_t n, int64_t ndim, const DimArray<int64_t> outStrides,
                        const DimArray<int64_t> inShape, const DimArray<int64_t> inStrides) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    int64_t inIdx = 0;
    int64_t tmpIdx = idx;
    int64_t coords[MAX_TENSOR_DIM];

#pragma unroll
    for (int d = 0; d < ndim; d++) {
      coords[d] = tmpIdx / outStrides.data[d];
      tmpIdx %= outStrides.data[d];
    }

#pragma unroll
    for (int d = 0; d < ndim; d++) {
      int64_t srcIdx = (inShape.data[d] == 1) ? 0 : coords[d];
      inIdx += srcIdx * inStrides.data[d];
    }

    out[idx] = in[inIdx];
  }
}

template <typename T>
__global__ void kIndexSelect(T* out, const T* in, const int64_t* indices, int64_t n, int64_t ndim, int64_t indexDim,
                             const DimArray<int64_t> inStrides, const DimArray<int64_t> outStrides) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    int64_t tmpIdx = idx;
    int64_t coords[MAX_TENSOR_DIM];

#pragma unroll
    for (int d = 0; d < ndim; d++) {
      coords[d] = tmpIdx / outStrides.data[d];
      tmpIdx %= outStrides.data[d];
    }

    int64_t inIdx = 0;

#pragma unroll
    for (int d = 0; d < ndim; d++) {
      if (d == indexDim) {
        inIdx += indices[coords[d]] * inStrides.data[d];
      } else {
        inIdx += coords[d] * inStrides.data[d];
      }
    }

    out[idx] = in[inIdx];
  }
}

template <typename T>
__global__ void kRepeatInterleave(T* out, const T* in, int64_t n, int64_t dimSize, int64_t repeats, int64_t innerSize) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    int64_t innerIdx = idx % innerSize;
    int64_t rest = idx / innerSize;
    int64_t dimIdx = rest % (dimSize * repeats);
    int64_t outerIdx = rest / (dimSize * repeats);

    int64_t inDimIdx = dimIdx / repeats;
    int64_t inIdx = outerIdx * (dimSize * innerSize) + inDimIdx * innerSize + innerIdx;

    out[idx] = in[inIdx];
  }
}

template <typename T>
__global__ void kDataCheck(const T* selfPtr, int64_t n) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    auto v = static_cast<float>(selfPtr[idx]);
    assert(!isnan(v));
    assert(!isinf(v));
    UNUSED(v);
  }
}

template <typename T>
void cudaTranspose2d(T* out, const T* in, int64_t width, int64_t height, const Device& device) {
  dim3 blockSize(TRANSPOSE_TILE_DIM, TRANSPOSE_TILE_DIM);
  dim3 gridSize((width + TRANSPOSE_TILE_DIM - 1) / TRANSPOSE_TILE_DIM,
                (height + TRANSPOSE_TILE_DIM - 1) / TRANSPOSE_TILE_DIM);

  const auto& stream = cuda::getCurrentCUDAStream(device.index).stream();
  kTranspose2D<<<gridSize, blockSize, 0, stream>>>(out, in, width, height);
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
Tensor transpose2dOpCudaImpl(const Tensor& self) {
  ASSERT(self.dim() == 2);

  SizeVector retShape = {self.shape(1), self.shape(0)};
  auto ret = Tensor::empty(retShape, self.options().noGrad());
  using CudaT = typename cuda::CudaTypeCast<T>::type;
  cudaTranspose2d(ret.dataPtr<CudaT>(), self.dataPtr<CudaT>(), retShape[0], retShape[1], self.device());
  return ret;
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

  if (dim0 == dim1) {
    return self.clone();
  }

  if (dim0 > dim1) {
    std::swap(dim0, dim1);
  }

  SizeVector retShape(self.shape());
  std::swap(retShape[dim0], retShape[dim1]);

  if ((self.size(dim0) == 1 || self.size(dim1) == 1) && dim1 - dim0 == 1) {
    return op::view(self, retShape);
  }

  if (self.dim() == 2) {
    return transpose2dOpCudaImpl<T>(self);
  }

  SizeVector mergedShape;
  SizeVector mergedOutShape;

  int64_t preSize = 1;
  for (int64_t i = 0; i < dim0; i++) {
    preSize *= self.size(i);
  }
  if (preSize > 1) {
    mergedShape.pushBack(preSize);
    mergedOutShape.pushBack(preSize);
  }

  mergedShape.pushBack(self.size(dim0));
  mergedOutShape.pushBack(self.size(dim1));

  int64_t midSize = 1;
  for (int64_t i = dim0 + 1; i < dim1; i++) {
    midSize *= self.size(i);
  }
  if (midSize > 1) {
    mergedShape.pushBack(midSize);
    mergedOutShape.pushBack(midSize);
  }

  mergedShape.pushBack(self.size(dim1));
  mergedOutShape.pushBack(self.size(dim0));

  int64_t postSize = 1;
  for (int64_t i = dim1 + 1; i < self.dim(); i++) {
    postSize *= self.size(i);
  }
  if (postSize > 1) {
    mergedShape.pushBack(postSize);
    mergedOutShape.pushBack(postSize);
  }

  Tensor mergedInput = op::reshape(self, mergedShape);
  Tensor mergedOutput = Tensor::empty(mergedOutShape, self.options().noGrad());

  int64_t newDim0 = 0, newDim1 = 0;
  int pos = 0;
  if (preSize > 1) {
    pos++;
  }
  newDim0 = pos++;
  if (midSize > 1) {
    pos++;
  }
  newDim1 = pos;

  DimArray<int64_t> inStrides{};
  DimArray<int64_t> outStrides{};
  for (auto i = 0; i < mergedInput.dim(); i++) {
    inStrides.data[i] = mergedInput.stride(i);
    outStrides.data[i] = mergedOutput.stride(i);
  }

  auto params = cuda::getKernelLaunchParams(self.device().index, mergedOutput.numel());
  CUDA_LAUNCH_KERNEL(kTransposeND<T>, params, mergedOutput.dataPtr<T>(), mergedInput.dataPtr<T>(), mergedInput.dim(),
                     newDim0, newDim1, mergedOutput.numel(), outStrides, inStrides);

  return op::reshape(mergedOutput, retShape);
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

  const auto dims = self.dim();
  const auto rows = self.shape(dims - 2);
  const auto cols = self.shape(dims - 1);

  int64_t batch = 1;
  for (int i = 0; i < dims - 2; i++) {
    batch *= self.shape(i);
  }

  const int64_t matrixSize = rows * cols;
  const T* selfPtr = self.dataPtr<T>();
  T* retPtr = ret.dataPtr<T>();

  dim3 blockSize(16, 16);
  dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y, batch);

  const auto& stream = cuda::getCurrentCUDAStream(self.device().index).stream();
  kTriangle<T, LOWER><<<gridSize, blockSize, 0, stream>>>(retPtr, selfPtr, batch, rows, cols, diagonal, matrixSize);
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

template <typename T>
Tensor expandOpCudaImpl(const Tensor& self, IntArrayView sizes) {
  auto inDim = self.dim();
  auto outDim = static_cast<int64_t>(sizes.size());
  ASSERT(outDim >= inDim);

  SizeVector retShape(outDim, 1);
  for (auto i = 0; i < outDim; i++) {
    int64_t inputIdx = i - (outDim - inDim);
    if (sizes[i] == -1) {
      retShape[i] = (inputIdx >= 0 ? self.shape(inputIdx) : 1);
    } else {
      ASSERT(sizes[i] > 0);
      retShape[i] = sizes[i];
    }
  }

  auto ret = Tensor::empty(retShape, self.options().noGrad());
  if (self.isScalar()) {
    op::fillOffset(ret, self, 0, ret.numel());
    return ret;
  }

  SizeVector inShape(outDim, 1);
  SizeVector inStride(outDim, 0);
  for (auto i = 0; i < inDim; i++) {
    inShape[outDim - inDim + i] = self.shape(i);
    inStride[outDim - inDim + i] = self.stride(i);
  }

  T* outPtr = ret.dataPtr<T>();
  const T* selfPtr = self.dataPtr<T>();

  DimArray<int64_t> outStrideArr{};
  DimArray<int64_t> inShapeArr{};
  DimArray<int64_t> inStrideArr{};

  for (auto i = 0; i < outDim; i++) {
    outStrideArr.data[i] = ret.stride(i);
    inShapeArr.data[i] = inShape[i];
    inStrideArr.data[i] = inStride[i];
  }

  auto params = cuda::getKernelLaunchParams(self.device().index, ret.numel());
  CUDA_LAUNCH_KERNEL(kExpand<T>, params, outPtr, selfPtr, ret.numel(), outDim, outStrideArr, inShapeArr, inStrideArr);
  return ret;
}

template <typename T>
Tensor indexSelectOpCudaImpl(const Tensor& self, int64_t dim, const Tensor& index) {
  int64_t ndim = self.dim();
  if (dim < 0) {
    dim += ndim;
  }
  ASSERT(dim >= 0 && dim < ndim);
  ASSERT(index.dim() == 1);
  ASSERT(index.dtype() == DType::Int64);

  SizeVector retShape = self.shape();
  retShape[dim] = index.numel();
  auto ret = Tensor::empty(retShape, self.options().noGrad());

  if (index.numel() == 0) {
    return ret;
  }

  const auto* selfPtr = self.dataPtr<T>();
  auto* retPtr = ret.dataPtr<T>();
  const auto* indexPtr = index.dataPtr<int64_t>();

  DimArray<int64_t> outStrides{};
  DimArray<int64_t> inStrides{};

  for (int64_t i = 0; i < ndim; i++) {
    outStrides.data[i] = ret.stride(i);
    inStrides.data[i] = self.stride(i);
  }

  auto params = cuda::getKernelLaunchParams(self.device().index, ret.numel());
  CUDA_LAUNCH_KERNEL(kIndexSelect<T>, params, retPtr, selfPtr, indexPtr, ret.numel(), ndim, dim, inStrides, outStrides);
  return ret;
}

template <typename T>
Tensor repeatInterleaveOpCudaImpl(const Tensor& self, int64_t repeats, int64_t dim) {
  int64_t ndim = self.dim();
  if (dim < 0) {
    dim += ndim;
  }
  ASSERT(dim >= 0 && dim < ndim);
  ASSERT(repeats > 0);

  SizeVector retShape = self.shape();
  retShape[dim] *= repeats;
  auto ret = Tensor::empty(retShape, self.options().noGrad());

  const auto* selfPtr = self.dataPtr<T>();
  auto* retPtr = ret.dataPtr<T>();

  int64_t outerSize = 1, innerSize = 1;
  for (int64_t i = 0; i < dim; i++) {
    outerSize *= self.shape(i);
  }
  for (int64_t i = dim + 1; i < ndim; i++) {
    innerSize *= self.shape(i);
  }

  auto params = cuda::getKernelLaunchParams(self.device().index, ret.numel());
  CUDA_LAUNCH_KERNEL(kRepeatInterleave<T>, params, retPtr, selfPtr, ret.numel(), self.shape(dim), repeats, innerSize);
  return ret;
}

template <typename T>
void checkOpCudaImpl(const Tensor& self) {
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, BFloat16> || std::is_same_v<T, Half>) {
    using CudaT = typename cuda::CudaTypeCast<T>::type;
    const auto* selfPtr = self.dataPtr<CudaT>();
    auto params = cuda::getKernelLaunchParams(self.device().index, self.numel());
    CUDA_LAUNCH_KERNEL((kDataCheck<CudaT>), params, selfPtr, self.numel());
  }
}

}  // namespace tinytorch::op
