/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "TensorIterator.h"

namespace tinytorch {

IntArrayView TensorIteratorBase::setupBroadcast() {
  size_t nInputs = inputs_.size();
  shapes_.resize(nInputs);
  strides_.resize(nInputs);
  isScalar_.resize(nInputs);

  ndim_ = 0;
  for (auto t : inputs_) {
    ndim_ = std::max(ndim_, t->dim());
  }
  ASSERT(ndim_ <= MAX_TENSOR_DIM);

  SmallVector<IntArrayView> inputShapes;
  inputShapes.reserve(inputs_.size());
  for (auto t : inputs_) {
    inputShapes.pushBack(t->shape());
  }

  broadcastOk_ = broadcastShape(shape_, inputShapes, ndim_);
  if (!broadcastOk_) {
    ASSERT("broadcast check failed");
    return {};
  }

  total_ = 1;
  for (auto s : shape_) {
    total_ *= s;
  }

  for (size_t i = 0; i < nInputs; i++) {
    alignShape(shapes_[i], *inputs_[i], ndim_);
    alignStrides(strides_[i], *inputs_[i], ndim_);
    isScalar_[i] = inputs_[i]->isScalar();
  }

  needBroadcast_ = false;
  for (size_t i = 0; i < nInputs; i++) {
    if (inputs_[i]->shape() != shape_.view()) {
      needBroadcast_ = true;
      break;
    }
  }

  allNonScalarSameShape_ = true;
  IntArrayView refShape;
  bool found = false;
  for (size_t i = 0; i < nInputs; i++) {
    if (!isScalar_[i]) {
      if (!found) {
        refShape = inputs_[i]->shape();
        found = true;
      } else if (inputs_[i]->shape() != refShape) {
        allNonScalarSameShape_ = false;
        break;
      }
    }
  }

  // TODO coalesce dimensions
  return shape_.view();
}

bool TensorIteratorBase::broadcastShape(SizeVector& ret, const SmallVector<IntArrayView>& shapes, int64_t ndim) {
  ret.resize(ndim, 1);
  for (auto i = 0; i < ndim; i++) {
    int64_t dim = 1;
    for (const auto& shape : shapes) {
      int64_t d = i < ndim - shape.size() ? 1 : shape[i - (ndim - shape.size())];
      if (d != 1 && dim != 1 && d != dim) {
        LOGE("Incompatible shapes for broadcasting");
        return false;
      }
      dim = std::max(dim, d);
    }
    ret[i] = dim;
  }
  return true;
}

void TensorIteratorBase::alignShape(SizeVector& ret, const Tensor& t, int64_t ndim) {
  auto shape = t.shape();
  ret.resize(ndim, 1);
  for (auto i = 0; i < shape.size(); i++) {
    ret[ndim - shape.size() + i] = shape[i];
  }
}

void TensorIteratorBase::alignStrides(SizeVector& ret, const Tensor& t, int64_t ndim) {
  auto shape = t.shape();
  auto strides = t.strides();
  ret.resize(ndim, 0);
  for (auto i = 0; i < shape.size(); i++) {
    ret[ndim - shape.size() + i] = (shape[i] == 1 ? 0 : strides[i]);
  }
}

}  // namespace tinytorch