/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "TensorImpl.h"

#include "Utils/Logger.h"

namespace tinytorch {

TensorImpl::TensorImpl(const IntArrayView shape, Options options) : options_(options), shape_(shape) {
  ASSERT(shape.size() <= MAX_TENSOR_DIM);
  computeNumel(numel_, shape);
  computeStrides(strides_, shape);

  int64_t nbytes = numel_ * static_cast<int64_t>(dtypeSize(options.dtype_));
  Allocator* allocator = getAllocator(options);
  storageOffset_ = 0;
  storage_ = std::make_shared<Storage>(nbytes, options.device_, allocator);
  dataPtr_ = static_cast<uint8_t*>(storage_->data()) + storageOffset_;
}

TensorImpl::TensorImpl(const IntArrayView shape, Options options, const std::shared_ptr<Storage>& storage,
                       int64_t offset)
    : options_(options), shape_(shape) {
  ASSERT(shape.size() <= MAX_TENSOR_DIM);
  computeNumel(numel_, shape);
  computeStrides(strides_, shape);

  int64_t nbytes = numel_ * static_cast<int64_t>(dtypeSize(options.dtype_));
  ASSERT(offset + nbytes <= storage->size());
  storageOffset_ = offset;
  storage_ = storage;
  dataPtr_ = static_cast<uint8_t*>(storage_->data()) + storageOffset_;
}

void TensorImpl::copyOnWrite() {
  if (storage_ && storage_.use_count() > 1) {
    storage_ = storage_->clone();
    dataPtr_ = static_cast<uint8_t*>(storage_->data()) + storageOffset_;
  }
}

void TensorImpl::reshape(const IntArrayView shape) {
  // set as scalar
  if (shape.empty() && numel_ == 1) {
    shape_.clear();
    strides_.clear();
    return;
  }

  SizeVector retShape(shape.size());

  int64_t inferredIdx = -1;
  int64_t cnt = 1;
  for (size_t i = 0; i < shape.size(); i++) {
    if (shape[i] == -1) {
      if (inferredIdx >= 0) {
        LOGE("Invalid shape: more than one idx to infer");
        return;
      }
      inferredIdx = static_cast<int64_t>(i);
      retShape[i] = 0;
    } else {
      cnt *= shape[i];
      retShape[i] = shape[i];
    }
  }
  if (inferredIdx >= 0) {
    if (cnt == 0 || numel_ % cnt != 0) {
      LOGE("Invalid shape: cannot infer dimension");
      return;
    }
    retShape[inferredIdx] = numel_ / cnt;
  }
  // check shape
  int64_t numel = 0;
  computeNumel(numel, retShape.view());
  if (numel != numel_) {
    LOGE("Invalid shape: numel not equal");
    return;
  }

  // update shape & strides
  shape_ = std::move(retShape);
  computeStrides(strides_, shape_.view());
}

void TensorImpl::flatten(int64_t startDim, int64_t endDim) {
  SizeVector retShape;
  for (int64_t i = 0; i < startDim; i++) {
    retShape.pushBack(shape_[i]);
  }
  int64_t flattenDims = 1;
  if (endDim < 0) {
    endDim = dim() - 1;
  }
  for (int64_t i = startDim; i <= endDim; i++) {
    flattenDims *= shape_[i];
  }
  retShape.pushBack(flattenDims);
  for (int64_t i = endDim + 1; i < dim(); i++) {
    retShape.pushBack(shape_[i]);
  }

  reshape(retShape.view());
}

void TensorImpl::unflatten(int64_t d, const IntArrayView shape) {
  if (d < 0) {
    d += dim();
  }
  SizeVector retShape;
  for (int64_t i = 0; i < d; i++) {
    retShape.pushBack(shape_[i]);
  }
  int64_t unflattenDims = 1;
  int64_t inferredIdx = -1;
  for (size_t i = 0; i < shape.size(); i++) {
    if (shape[i] == -1) {
      inferredIdx = d + static_cast<int64_t>(i);
      retShape.pushBack(0);
    } else {
      unflattenDims *= shape[i];
      retShape.pushBack(shape[i]);
    }
  }
  if (inferredIdx >= 0) {
    if (unflattenDims == 0 || shape_[d] % unflattenDims != 0) {
      LOGE("Invalid axis");
      return;
    }
    retShape[inferredIdx] = shape_[d] / unflattenDims;
  } else if (unflattenDims != shape_[d]) {
    LOGE("Invalid axis");
    return;
  }
  for (int64_t i = d + 1; i < dim(); i++) {
    retShape.pushBack(shape_[i]);
  }

  reshape(retShape.view());
}

void TensorImpl::squeeze(int64_t d) {
  if (d < -dim() || d >= dim()) {
    LOGE("Invalid dim");
    return;
  }
  if (d < 0) {
    d += dim();
  }
  if (shape_[d] != 1) {
    // Ref: https://docs.pytorch.org/docs/stable/generated/torch.squeeze.html
    // If input is of shape: (A x 1 x B), squeeze(input, 0) leaves the tensor unchanged
    return;
  }

  SizeVector retShape;
  for (int64_t i = 0; i < d; i++) {
    retShape.pushBack(shape_[i]);
  }
  for (int64_t i = d + 1; i < dim(); i++) {
    retShape.pushBack(shape_[i]);
  }

  reshape(retShape.view());
}

void TensorImpl::squeeze(const IntArrayView dims) {
  if (dims.empty()) {
    // squeeze all dimensions with size 1
    SizeVector retShape;
    for (auto d : shape_) {
      if (d != 1) {
        retShape.pushBack(d);
      }
    }
    reshape(retShape.view());
  } else {
    SizeVector retShape;
    for (int64_t i = 0; i < dim(); ++i) {
      bool shouldSqueeze = false;
      for (auto d : dims) {
        int64_t dd = d;
        if (dd < 0) {
          dd += dim();
        }
        if (dd == i && shape_[i] == 1) {
          shouldSqueeze = true;
          break;
        }
      }
      if (!shouldSqueeze) {
        retShape.pushBack(shape_[i]);
      }
    }
    reshape(retShape.view());
  }
}

void TensorImpl::unsqueeze(int64_t d) {
  if (d > dim() || d < -dim() - 1) {
    LOGE("Invalid axis");
    return;
  }
  if (d < 0) {
    d += dim() + 1;
  }
  SizeVector retShape;
  for (int64_t i = 0; i < d; i++) {
    retShape.pushBack(shape_[i]);
  }
  retShape.pushBack(1);
  for (int64_t i = d; i < dim(); i++) {
    retShape.pushBack(shape_[i]);
  }

  reshape(retShape.view());
}

void TensorImpl::computeStrides(SizeVector& strides, const IntArrayView shape) {
  strides.resize(shape.size());
  int64_t stride = 1;
  for (auto i = static_cast<int64_t>(shape.size()) - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i];
  }
}

void TensorImpl::computeNumel(int64_t& numel, const IntArrayView shape) {
  numel = 1;
  for (auto s : shape) {
    numel *= s;
  }
}

}  // namespace tinytorch
