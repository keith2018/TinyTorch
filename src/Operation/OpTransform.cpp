/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "OpTransform.h"

#include <cstring>

#include "OpFilling.h"
#include "Utils/Logger.h"

namespace tinytorch::op {

int64_t indicesToOffset(const IntArrayView strides, const int64_t* indices) {
  int64_t offset = 0;
  for (size_t i = 0; i < strides.size(); i++) {
    offset += indices[i] * strides[i];
  }
  return offset;
}

void offsetToIndices(int64_t* indices, const IntArrayView shape, int64_t offset) {
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; i--) {
    indices[i] = offset % shape[i];
    offset /= shape[i];
  }
}

void reorderIndices(int64_t* indices, int64_t ndim, const IntArrayView order) {
  ASSERT(ndim == static_cast<int64_t>(order.size()));
  static SizeVector temp(MAX_TENSOR_DIM);
  for (auto i = 0; i < ndim; i++) {
    auto d = order[i];
    if (d < 0) {
      d += ndim;
    }
    if (d < 0 || d >= ndim) {
      LOGE("Invalid axis: %lld at %lld", order[i], i);
      return;
    }
    temp[i] = indices[d];
  }
  std::memcpy(indices, temp.data(), sizeof(int64_t) * ndim);
}

Tensor reshapeOpImpl(const Tensor& self, const IntArrayView shape) {
  Tensor ret = self.clone();
  ret.getImpl().reshape_(shape);
  return ret;
}

void reshapeOpInplaceImpl(Tensor& self, const IntArrayView shape) {
  // reshape inplace
  self.getImpl().reshape_(shape);
}

Tensor flattenOpImpl(const Tensor& self, int64_t startDim, int64_t endDim) {
  Tensor ret = self.clone();
  ret.getImpl().flatten_(startDim, endDim);
  return ret;
}

void flattenOpInplaceImpl(Tensor& self, int64_t startDim = 0, int64_t endDim = -1) {
  self.getImpl().flatten_(startDim, endDim);
}

Tensor unflattenOpImpl(const Tensor& self, int64_t dim, const IntArrayView shape) {
  Tensor ret = self.clone();
  ret.getImpl().unflatten_(dim, shape);
  return ret;
}
void unflattenOpInplaceImpl(Tensor& self, int64_t dim, const IntArrayView shape) {
  self.getImpl().unflatten_(dim, shape);
}

Tensor squeezeOpImpl(const Tensor& self, const IntArrayView dims) {
  Tensor ret = self.clone();
  ret.getImpl().squeeze_(dims);
  return ret;
}
void squeezeOpInplaceImpl(Tensor& self, const IntArrayView dims) { self.getImpl().squeeze_(dims); }

Tensor unsqueezeOpImpl(const Tensor& self, int64_t dim) {
  Tensor ret = self.clone();
  ret.getImpl().unsqueeze_(dim);
  return ret;
}

void unsqueezeOpInplaceImpl(Tensor& self, int64_t dim) { self.getImpl().unsqueeze_(dim); }

inline std::pair<int64_t, int64_t> computeIndexAndStride(const Tensor& self, const IntArrayView indices) {
  auto len = static_cast<int64_t>(indices.size());
  int64_t dataIdx = 0;
  for (int64_t i = 0; i < len; i++) {
    auto idx = indices[i];
    dataIdx += (idx >= 0 ? idx : idx + self.shape(i)) * self.stride(i);
  }
  int64_t dimStride = self.stride(len - 1);
  return {dataIdx, dimStride};
}

template <typename T>
Tensor indexOpImpl(const Tensor& self, const IntArrayView indices) {
  auto len = static_cast<int64_t>(indices.size());
  auto [dataIdx, dimStride] = computeIndexAndStride(self, indices);
  SizeVector retShape;
  retShape.reserve(self.dim());
  for (int64_t i = len; i < self.dim(); i++) {
    retShape.pushBack(self.shape(i));
  }
  // create tensor with shared storage
  auto ret = Tensor(retShape, self.options(), self.storage(), dataIdx * sizeof(T));
  ASSERT(dimStride == ret.numel());
  return ret;
}

template <typename T>
void indexPutOpImpl(Tensor& self, const IntArrayView indices, const Tensor& val) {
  auto [dataIdx, dimStride] = computeIndexAndStride(self, indices);
  self.copyOnWrite();
  if (val.isScalar()) {
    op::fillOffset(self, val, dataIdx, dimStride);
  } else {
    ASSERT(dimStride == val.numel());
    T* selfPtr = self.dataPtr<T>();
    const T* valPtr = val.dataPtr<T>();
    Storage::copyOnDevice(selfPtr + dataIdx, valPtr, dimStride * sizeof(T), self.device());
  }
}

inline int64_t calcBlockSize(const IntArrayView shape, int64_t begin, int64_t end) {
  int64_t size = 1;
  for (int64_t d = begin; d < end; d++) {
    size *= shape[d];
  }
  return size;
}

template <typename T>
std::vector<Tensor> splitOpImpl(const Tensor& self, int64_t splitSize, int64_t dim) {
  if (dim < 0) {
    dim += self.dim();
  }
  if (dim < 0 || dim >= self.dim()) {
    LOGE("Invalid axis: %lld", dim);
    ASSERT(false);
    return {};
  }

  const auto dimSize = self.shape(dim);
  if (splitSize <= 0 || dimSize % splitSize != 0) {
    LOGE("Invalid sections: %lld", splitSize);
    ASSERT(false);
    return {};
  }

  const auto sections = dimSize / splitSize;
  std::vector<Tensor> retTensors(sections);

  // shape of result tensors
  SizeVector retShape(self.shape());
  retShape[dim] = splitSize;
  for (auto i = 0; i < sections; i++) {
    retTensors[i] = Tensor::empty(retShape, self.options().noGrad());
  }

  int64_t innerBlockSize = calcBlockSize(self.shape(), dim + 1, self.dim());
  int64_t outerBlockSize = calcBlockSize(self.shape(), 0, dim);

  const T* selfPtr = self.dataPtr<T>();
  for (auto i = 0; i < sections; i++) {
    T* retPtr = retTensors[i].dataPtr<T>();
    for (auto outerIdx = 0; outerIdx < outerBlockSize; outerIdx++) {
      int64_t srcOffset = outerIdx * dimSize * innerBlockSize + i * splitSize * innerBlockSize;
      int64_t dstOffset = outerIdx * splitSize * innerBlockSize;
      int64_t copySize = splitSize * innerBlockSize * sizeof(T);
      Storage::copyOnDevice(retPtr + dstOffset, selfPtr + srcOffset, copySize, self.device());
    }
  }
  return retTensors;
}

template <typename T>
Tensor concatOpImpl(ArrayView<Tensor> tensors, int64_t dim) {
  ASSERT(!tensors.empty());

  int64_t ndim = tensors[0].dim();
  if (dim < 0) dim += ndim;
  if (dim < 0 || dim >= ndim) {
    LOGE("Invalid axis: %lld", dim);
    ASSERT(false);
    return {};
  }

  // shape of result tensor
  SizeVector retShape(tensors[0].shape());
  int64_t concatDimSize = tensors[0].shape(dim);
  for (size_t i = 1; i < tensors.size(); i++) {
    if (tensors[i].dim() != ndim) {
      LOGE("All tensors must have the same number of dimensions");
      ASSERT(false);
      return {};
    }
    for (int64_t d = 0; d < ndim; d++) {
      if (d == dim) continue;
      if (tensors[i].shape(d) != retShape[d]) {
        LOGE("Tensor shapes must match except in concat dimension");
        ASSERT(false);
        return {};
      }
    }
    concatDimSize += tensors[i].shape(dim);
  }
  retShape[dim] = concatDimSize;

  int64_t innerBlockSize = calcBlockSize(retShape, dim + 1, ndim);
  int64_t outerBlockSize = calcBlockSize(retShape, 0, dim);

  Tensor ret = Tensor::empty(retShape, tensors[0].options().noGrad());
  T* retPtr = ret.dataPtr<T>();
  int64_t offset = 0;
  for (const auto& t : tensors) {
    int64_t currDimSize = t.shape(dim);
    const T* inPtr = t.dataPtr<T>();
    for (int64_t outerIdx = 0; outerIdx < outerBlockSize; ++outerIdx) {
      int64_t dstOffset = outerIdx * concatDimSize * innerBlockSize + offset * innerBlockSize;
      int64_t srcOffset = outerIdx * currDimSize * innerBlockSize;
      int64_t copySize = currDimSize * innerBlockSize * sizeof(T);
      Storage::copyOnDevice(retPtr + dstOffset, inPtr + srcOffset, copySize, t.device());
    }
    offset += currDimSize;
  }
  return ret;
}

template <typename T>
Tensor stackOpImpl(ArrayView<Tensor> tensors, int64_t dim) {
  ASSERT(!tensors.empty());
  // check dim
  auto& t0 = tensors[0];
  int64_t targetDim = dim >= 0 ? dim : dim + t0.dim() + 1;
  if (targetDim < 0 || targetDim > t0.dim()) {
    LOGE("Invalid axis: %lld", dim);
    ASSERT(false);
    return {};
  }

  // check device & shapes
  for (size_t i = 1; i < tensors.size(); i++) {
    auto& t = tensors[i];
    ASSERT(t.device() == t0.device());
    ASSERT(t.shape() == t0.shape());
  }

  // shape of result tensor
  SizeVector retShape(t0.shape());
  retShape.insert(retShape.begin() + targetDim, static_cast<int64_t>(tensors.size()));
  Tensor ret = Tensor::empty(retShape, t0.options().noGrad());

  int64_t innerBlockSize = calcBlockSize(t0.shape(), targetDim, t0.dim());
  int64_t outerBlockSize = calcBlockSize(t0.shape(), 0, targetDim);

  T* retPtr = ret.dataPtr<T>();
  for (size_t i = 0; i < tensors.size(); i++) {
    const auto& t = tensors[i];
    const T* srcPtr = t.dataPtr<T>();
    auto* dstPtr = retPtr + i * innerBlockSize;

    for (int64_t j = 0; j < outerBlockSize; j++) {
      Storage::copyOnDevice(dstPtr, srcPtr, innerBlockSize * sizeof(T), t0.device());
      srcPtr += innerBlockSize;
      dstPtr += tensors.size() * innerBlockSize;
    }
  }

  return ret;
}

template <typename T>
Tensor vstackOpImpl(ArrayView<Tensor> tensors) {
  ASSERT(!tensors.empty());
  if (tensors[0].dim() == 1) {
    return stack(tensors, 0);
  } else {
    return concat(tensors, 0);
  }
}

template <typename T>
Tensor hstackOpImpl(ArrayView<Tensor> tensors) {
  ASSERT(!tensors.empty());
  if (tensors[0].dim() == 1) {
    return concat(tensors, 0);
  } else {
    return concat(tensors, 1);
  }
}

template <typename T>
Tensor narrowOpImpl(const Tensor& self, int64_t dim, int64_t start, int64_t length) {
  if (dim < 0) dim += self.dim();
  ASSERT(dim >= 0 && dim < self.dim());

  if (start < 0) start += self.shape(dim);
  ASSERT(start >= 0 && start < self.shape(dim));
  ASSERT(length >= 0 && start + length <= self.shape(dim));

  SizeVector retShape(self.shape());
  retShape[dim] = length;

  auto ret = Tensor::empty(retShape, self.options().noGrad());

  const T* selfPtr = self.dataPtr<T>();
  T* retPtr = ret.dataPtr<T>();

  int64_t outerSize = 1;
  int64_t innerSize = 1;
  for (int64_t i = 0; i < dim; i++) {
    outerSize *= self.shape(i);
  }
  for (int64_t i = dim + 1; i < self.dim(); i++) {
    innerSize *= self.shape(i);
  }

  int64_t srcStride = self.shape(dim) * innerSize;
  int64_t dstStride = length * innerSize;

  for (int64_t o = 0; o < outerSize; o++) {
    const T* srcBase = selfPtr + o * srcStride + start * innerSize;
    T* dstBase = retPtr + o * dstStride;
    Storage::copyOnDevice(dstBase, srcBase, sizeof(T) * length * innerSize, self.device());
  }
  return ret;
}

void registerTransformCommon() {
  // reshape/view
  REGISTER_OP_IMPL_ALL(reshape, &reshapeOpImpl);
  REGISTER_OP_IMPL_ALL(reshapeInplace, &reshapeOpInplaceImpl);
  REGISTER_OP_IMPL_ALL(view, &reshapeOpImpl);

  // flatten
  REGISTER_OP_IMPL_ALL(flatten, &flattenOpImpl);
  REGISTER_OP_IMPL_ALL(flattenInplace, &flattenOpInplaceImpl);

  // unflatten
  REGISTER_OP_IMPL_ALL(unflatten, &unflattenOpImpl);
  REGISTER_OP_IMPL_ALL(unflattenInplace, &unflattenOpInplaceImpl);

  // squeeze
  REGISTER_OP_IMPL_ALL(squeeze, &squeezeOpImpl);
  REGISTER_OP_IMPL_ALL(squeezeInplace, &squeezeOpInplaceImpl);

  // unsqueeze
  REGISTER_OP_IMPL_ALL(unsqueeze, &unsqueezeOpImpl);
  REGISTER_OP_IMPL_ALL(unsqueezeInplace, &unsqueezeOpInplaceImpl);

  // index
  REGISTER_OP_IMPL_ALL_DEVICES_DTYPE_TPL(index, indexOpImpl);

  // indexPut
  REGISTER_OP_IMPL_ALL_DEVICES_DTYPE_TPL(indexPut, indexPutOpImpl);

  // split
  REGISTER_OP_IMPL_ALL_DEVICES_DTYPE_TPL(split, splitOpImpl);

  // concat
  REGISTER_OP_IMPL_ALL_DEVICES_DTYPE_TPL(concat, concatOpImpl);

  // stack
  REGISTER_OP_IMPL_ALL_DEVICES_DTYPE_TPL(stack, stackOpImpl);
  REGISTER_OP_IMPL_ALL_DEVICES_DTYPE_TPL(vstack, vstackOpImpl);
  REGISTER_OP_IMPL_ALL_DEVICES_DTYPE_TPL(hstack, hstackOpImpl);

  // narrow
  REGISTER_OP_IMPL_ALL_DEVICES_DTYPE_TPL(narrow, narrowOpImpl);
}

}  // namespace tinytorch::op