/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <numeric>

#include "OpFilling.h"
#include "OpTransform.h"
#include "Utils/Logger.h"
#include "Utils/RandomGenerator.h"

namespace tinytorch::op {

template <typename T>
Tensor permuteOpCpuImpl(const Tensor& self, const IntArrayView dims) {
  ASSERT(static_cast<int64_t>(dims.size()) == self.dim());
  auto retShape = SizeVector(self.shape());
  reorderIndices(retShape.data(), self.dim(), dims);
  auto ret = Tensor::empty(retShape, self.options().noGrad());

  const T* selfPtr = self.dataPtr<T>();
  T* retPtr = ret.dataPtr<T>();

  for (auto i = 0; i < self.numel(); i++) {
    int64_t originIndex = 0;
    int64_t offset = i;
    for (int64_t d = 0; d < self.dim(); d++) {
      originIndex += (offset / ret.stride(d)) * self.stride(dims[d]);
      offset %= ret.stride(d);
    }
    retPtr[i] = selfPtr[originIndex];
  }
  return ret;
}

template <typename T>
Tensor permuteAllOpCpuImpl(const Tensor& self) {
  SizeVector dims(self.dim());
  std::iota(dims.begin(), dims.end(), 0);
  std::reverse(dims.begin(), dims.end());
  return permuteOpCpuImpl<T>(self, dims);
}

template <typename T>
Tensor transpose2dOpCpuImpl(const Tensor& self) {
  ASSERT(self.dim() == 2);
  return permuteOpCpuImpl<T>(self, {1, 0});
}

template <typename T>
Tensor transposeOpCpuImpl(const Tensor& self, int64_t dim0, int64_t dim1) {
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

  if (self.dim() == 2) {
    return transpose2dOpCpuImpl<T>(self);
  }

  SizeVector dims(self.dim());
  std::iota(dims.begin(), dims.end(), 0);
  dims[dim0] = dim1;
  dims[dim1] = dim0;
  return permuteOpCpuImpl<T>(self, dims);
}

inline void getSubIndices(int64_t* subIndices, const Tensor& t, ArrayView<Tensor> indices, int64_t idx) {
  for (size_t i = 0; i < indices.size(); i++) {
    ASSERT(indices[i].dtype() == DType::Int64);
    auto* ptr = indices[i].dataPtr<int64_t>();
    auto ind = ptr[idx];
    subIndices[i] = ind >= 0 ? ind : ind + t.shape(static_cast<int64_t>(i));
  }
}

template <typename T, typename F>
void index2DCpuImpl(const Tensor& self, ArrayView<Tensor> indices, int64_t firstDim, F func) {
  auto dim0 = self.shape(0);
  auto dim1 = self.shape(1);
  ASSERT(indices[0].dtype() == DType::Int64);
  ASSERT(indices[1].dtype() == DType::Int64);
  auto* idx0Ptr = indices[0].dataPtr<int64_t>();
  auto* idx1Ptr = indices[1].dataPtr<int64_t>();
  for (int64_t i = 0; i < firstDim; i++) {
    int64_t idx0 = idx0Ptr[i];
    int64_t idx1 = idx1Ptr[i];
    if (idx0 < 0) idx0 += dim0;
    if (idx1 < 0) idx1 += dim1;
    int64_t dataIdx = idx0 * dim1 + idx1;
    func(i, dataIdx);
  }
}

template <typename T>
Tensor indexAdvanceOpCpuImpl(const Tensor& self, ArrayView<Tensor> indices) {
  auto len = static_cast<int64_t>(indices.size());
  auto firstDim = indices[0].numel();
  auto dimStride = self.stride(len - 1);
  SizeVector retShape(indices[0].shape());
  for (auto i = len; i < self.dim(); i++) {
    retShape.pushBack(self.shape(i));
  }
  auto ret = Tensor::empty(retShape, self.options().noGrad());
  const T* selfPtr = self.dataPtr<T>();
  T* retPtr = ret.dataPtr<T>();

  // 2D
  if (self.dim() == 2 && len == 2) {
    index2DCpuImpl<T>(self, indices, firstDim, [&](int64_t i, int64_t dataIdx) { retPtr[i] = selfPtr[dataIdx]; });
    return ret;
  }

  DimArray<int64_t> subIndices{};
  for (int64_t i = 0; i < firstDim; i++) {
    getSubIndices(subIndices.data, self, indices, i);
    int64_t dataIdx = indicesToOffset(self.strides(), subIndices.data);
    Storage::copyOnDevice(retPtr + (dimStride * i), selfPtr + dataIdx, dimStride * sizeof(T), self.device());
  }
  return ret;
}

template <typename T>
void indexPutAdvanceOpCpuImpl(Tensor& self, ArrayView<Tensor> indices, const Tensor& val) {
  auto len = static_cast<int64_t>(indices.size());
  auto firstDim = indices[0].numel();
  auto dimStride = self.stride(len - 1);

  T* selfPtr = self.dataPtr<T>();
  const T* valPtr = val.dataPtr<T>();

  // 2D
  if (self.dim() == 2 && len == 2) {
    if (val.isScalar()) {
      index2DCpuImpl<T>(self, indices, firstDim, [&](int64_t /*i*/, int64_t dataIdx) { selfPtr[dataIdx] = *valPtr; });
    } else {
      ASSERT(val.numel() == firstDim);
      index2DCpuImpl<T>(self, indices, firstDim, [&](int64_t i, int64_t dataIdx) { selfPtr[dataIdx] = valPtr[i]; });
    }
    return;
  }

  DimArray<int64_t> subIndices{};
  if (val.isScalar()) {
    for (int64_t i = 0; i < firstDim; i++) {
      getSubIndices(subIndices.data, self, indices, i);
      int64_t dataIdx = indicesToOffset(self.strides(), subIndices.data);
      op::fillOffset(self, val, dataIdx, dimStride);
    }
  } else {
    ASSERT(val.numel() == dimStride * firstDim);
    for (int64_t i = 0; i < firstDim; i++) {
      getSubIndices(subIndices.data, self, indices, i);
      int64_t dataIdx = indicesToOffset(self.strides(), subIndices.data);
      Storage::copyOnDevice(selfPtr + dataIdx, valPtr + (dimStride * i), dimStride * sizeof(T), self.device());
    }
  }
}

template <typename T, bool LOWER>
Tensor triangleOpCpuImpl(const Tensor& self, int64_t diagonal) {
  auto ret = Tensor::empty(self.shape(), self.options().noGrad());
  const auto rows = self.shape(0);
  const auto cols = self.shape(1);

  const T* selfPtr = self.dataPtr<T>();
  T* retPtr = ret.dataPtr<T>();

  int64_t idx = 0;
  for (auto i = 0; i < rows; i++) {
    idx = i * cols;
    for (auto j = 0; j < cols; j++) {
      if ((LOWER && j <= i + diagonal) || (!LOWER && j >= i + diagonal)) {
        retPtr[idx] = selfPtr[idx];
      } else {
        retPtr[idx] = 0;
      }
      idx++;
    }
  }
  return ret;
}

template <typename T>
Tensor trilOpCpuImpl(const Tensor& self, int64_t diagonal = 0) {
  return triangleOpCpuImpl<T, true>(self, diagonal);
}

template <typename T>
Tensor triuOpCpuImpl(const Tensor& self, int64_t diagonal = 0) {
  return triangleOpCpuImpl<T, false>(self, diagonal);
}

template <typename F>
void indexCoordForEach(const Tensor& self, int64_t dim, const Tensor& index, F func) {
  if (dim < 0) {
    dim += self.dim();
  }
  ASSERT(dim >= 0 && dim < self.dim());
  ASSERT(index.dim() == self.dim());

  const auto* idxPtr = index.dataPtr<int64_t>();
  auto strides = self.strides();
  auto selfShape = self.shape();
  auto indexShape = index.shape();
  auto ndim = self.dim();

  SizeVector indexStrides(ndim);
  int64_t stride = 1;
  for (auto d = ndim - 1; d >= 0; d--) {
    indexStrides[d] = stride;
    stride *= indexShape[d];
  }

  SizeVector coord(ndim);
  for (int64_t i = 0; i < index.numel(); i++) {
    int64_t remain = i;
    for (auto d = 0; d < ndim; d++) {
      coord[d] = remain / indexStrides[d];
      remain = remain % indexStrides[d];
    }
    int64_t idx = idxPtr[i];
    if (idx < 0) idx += selfShape[dim];
    ASSERT(idx >= 0 && idx < selfShape[dim]);
    coord[dim] = idx;

    int64_t offset = 0;
    for (auto d = 0; d < ndim; d++) {
      offset += coord[d] * strides[d];
    }
    func(i, offset);
  }
}

template <typename T>
Tensor gatherOpCpuImpl(const Tensor& self, int64_t dim, const Tensor& index) {
  Tensor ret = Tensor::empty(index.shape(), self.options().noGrad());
  const T* selfPtr = self.dataPtr<T>();
  T* retPtr = ret.dataPtr<T>();

  indexCoordForEach(self, dim, index, [&](int64_t i, int64_t offset) { retPtr[i] = selfPtr[offset]; });
  return ret;
}

template <typename T>
Tensor scatterOpCpuImpl(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
  Tensor ret = self.clone();
  ret.setRequiresGrad(false);
  ret.copyOnWrite();

  T* retPtr = ret.dataPtr<T>();
  const T* srcPtr = src.dataPtr<T>();

  indexCoordForEach(self, dim, index, [&](int64_t i, int64_t offset) { retPtr[offset] = srcPtr[i]; });
  return ret;
}

template <typename T>
void scatterOpInplaceCpuImpl(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
  self.copyOnWrite();
  T* selfPtr = self.dataPtr<T>();
  const T* srcPtr = src.dataPtr<T>();

  indexCoordForEach(self, dim, index, [&](int64_t i, int64_t offset) { selfPtr[offset] = srcPtr[i]; });
}

}  // namespace tinytorch::op
