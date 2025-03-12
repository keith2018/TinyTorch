/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "TensorImpl_base.h"

#include "TensorImpl.h"

namespace TinyTorch {

int32_t TensorOperations::indicesToOffset(const Shape &strides,
                                          const int32_t *indices) {
  int32_t offset = 0;
  for (int32_t i = 0; i < strides.size(); i++) {
    offset += indices[i] * strides[i];
  }
  return offset;
}

void TensorOperations::offsetToIndices(int32_t *indices, const Shape &shape,
                                       int32_t offset) {
  for (int32_t i = (int32_t)shape.size() - 1; i >= 0; i--) {
    indices[i] = offset % shape[i];
    offset /= shape[i];
  }
}

void TensorOperations::reorderIndices(int32_t *indices,
                                      const std::vector<int32_t> &order) {
  static std::vector<int32_t> temp(TENSOR_MAX_DIMS);
  for (int i = 0; i < order.size(); i++) {
    temp[i] = indices[order[i]];
  }
  memcpy(indices, temp.data(), sizeof(int32_t) * order.size());
}

bool TensorOperations::isLeadingOnes(const Shape &shape) {
  if (shape.size() == 1) {
    return true;
  }

  int32_t notOneIdx = -1;
  for (auto i = 0; i < shape.size(); i++) {
    if (shape[i] != 1) {
      notOneIdx = i;
    } else {
      if (notOneIdx != -1) {
        return false;
      }
    }
  }
  return true;
}

bool TensorOperations::isTrailingOnes(const Shape &shape) {
  int32_t notOneIdx = -1;
  for (auto i = static_cast<int32_t>(shape.size()) - 1; i >= 0; i--) {
    if (shape[i] != 1) {
      notOneIdx = i;
    } else {
      if (notOneIdx != -1) {
        return false;
      }
    }
  }
  return true;
}

Shape TensorOperations::getReduceShape(const TensorImpl &t, int32_t dim,
                                       bool keepDims) {
  Shape retShape;
  retShape.reserve(t.dimCount_);
  for (int32_t d = 0; d < t.dimCount_; d++) {
    if (d == dim) {
      if (keepDims) {
        retShape.emplace_back(1);
      }
    } else {
      retShape.emplace_back(t.shape_[d]);
    }
  }
  return retShape;
}

Shape TensorOperations::getReduceShape(const TensorImpl &t,
                                       const FixedVector<uint8_t> &inAxis,
                                       bool keepDims) {
  Shape retShape;
  retShape.reserve(t.dimCount_);
  for (int32_t d = 0; d < t.dimCount_; d++) {
    if (inAxis.data[d] != 0) {
      if (keepDims) {
        retShape.emplace_back(1);
      }
    } else {
      retShape.emplace_back(t.shape_[d]);
    }
  }
  return retShape;
}

ShapeCompatible TensorOperations::checkShapeCompatible(const Shape &t0,
                                                       const Shape &t1,
                                                       Shape &retShape,
                                                       int32_t skipLast) {
  retShape = t0.size() > t1.size() ? t0 : t1;

  auto idxRet = (int32_t)(retShape.size() - 1 - skipLast);
  auto idx0 = (int32_t)(t0.size() - 1 - skipLast);
  auto idx1 = (int32_t)(t1.size() - 1 - skipLast);

  bool needBroadcast = false;
  while (idx0 >= 0 && idx1 >= 0) {
    auto dim0 = t0[idx0];
    auto dim1 = t1[idx1];
    if (dim0 != dim1) {
      if (dim0 == 1 || dim1 == 1) {
        retShape[idxRet] = std::max(dim0, dim1);
        needBroadcast = true;
      } else {
        return ShapeCompatible_Error;
      }
    }

    idxRet--;
    idx0--;
    idx1--;
  }

  if (!needBroadcast && t0.size() == t1.size()) {
    return ShapeCompatible_SameShape;
  }

  return ShapeCompatible_Broadcast;
}

void TensorOperations::error(const char *where, TensorError error) {
  switch (error) {
    case TensorError_EmptyTensor:
      LOGE("[%s] Tensor error: empty tensor", where);
      break;
    case TensorError_InvalidShape:
      LOGE("[%s] Tensor error: invalid shape", where);
      break;
    case TensorError_InvalidAxis:
      LOGE("[%s] Tensor error: invalid axis", where);
      break;
    case TensorError_InvalidSections:
      LOGE("[%s] Tensor error: invalid sections", where);
      break;
    case TensorError_ShapeNotAligned:
      LOGE("[%s] Tensor error: shapes not aligned", where);
      break;
    case TensorError_DeviceNotAligned:
      LOGE("[%s] Tensor error: device not aligned", where);
      break;
    case TensorError_NotSupport:
      LOGE("[%s] Tensor error: function not support", where);
      break;
    default:
      break;
  }

#ifdef DEBUG
  abort();
#endif
}

}  // namespace TinyTorch
