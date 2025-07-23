/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "OpLinalg.h"

namespace tinytorch::op {

SizeVector broadcastShape(const IntArrayView t0, const IntArrayView t1, int64_t skipLast) {
  SizeVector retShape = SizeVector(t0.size() > t1.size() ? t0 : t1);

  auto idxRet = static_cast<int64_t>(retShape.size()) - 1 - skipLast;
  auto idx0 = static_cast<int64_t>(t0.size()) - 1 - skipLast;
  auto idx1 = static_cast<int64_t>(t1.size()) - 1 - skipLast;

  while (idx0 >= 0 && idx1 >= 0) {
    auto dim0 = t0[idx0];
    auto dim1 = t1[idx1];
    if (dim0 != dim1) {
      if (dim0 == 1 || dim1 == 1) {
        retShape[idxRet] = std::max(dim0, dim1);  // needBroadcast
      } else {
        return {};
      }
    }

    idxRet--;
    idx0--;
    idx1--;
  }

  return retShape;
}

template <typename T>
Tensor matmulOpImpl(const Tensor &self, const Tensor &other) {
  if (self.dim() == 0 || other.dim() == 0) {
    ASSERT(false && "matmul error: invalid shape");
    return {};
  }

  SizeVector shapeA(self.shape());
  SizeVector shapeB(other.shape());
  bool prependA = false;
  bool appendB = false;
  if (shapeA.size() == 1) {
    shapeA.insert(shapeA.begin(), 1);
    prependA = true;
  }
  if (shapeB.size() == 1) {
    shapeB.insert(shapeB.end(), 1);
    appendB = true;
  }

  // check matrix multiplication compatible
  if (shapeA.back() != shapeB[shapeB.size() - 2]) {
    ASSERT(false && "matmul error: shape not aligned");
    return {};
  }

  // check shape broadcast compatible
  SizeVector retShape = broadcastShape(shapeA, shapeB, 2);
  if (retShape.empty()) {
    ASSERT(false && "matmul error: shape not aligned");
    return {};
  }

  auto retDimCnt = static_cast<int64_t>(retShape.size());
  auto m = shapeA[shapeA.size() - 2];
  auto k = shapeA.back();
  auto n = shapeB.back();

  retShape[retDimCnt - 2] = m;
  retShape[retDimCnt - 1] = n;
  Tensor retTensor = Tensor::empty(retShape, self.options().noGrad());

  const T *selfPtr = self.dataPtr<T>();
  const T *otherPtr = other.dataPtr<T>();
  T *retPtr = retTensor.dataPtr<T>();

  auto gemm = getGemmFunc<T>(self.device().type);

  if (retDimCnt > 2) {
    // batched matrix multiply with broadcasting
    int64_t batchSize = retTensor.numel() / (m * n);

    SizeVector aStrides(self.strides());
    SizeVector bStrides(other.strides());
    while (static_cast<int64_t>(aStrides.size()) < retTensor.dim()) {
      aStrides.insert(aStrides.begin(), 0);
    }
    while (static_cast<int64_t>(bStrides.size()) < retTensor.dim()) {
      bStrides.insert(bStrides.begin(), 0);
    }

    for (int64_t batch = 0; batch < batchSize; batch++) {
      int64_t aOffset = 0;
      int64_t bOffset = 0;
      int64_t tmp = batch;
      for (auto i = retDimCnt - 3; i >= 0; i--) {
        int64_t index = tmp % retShape[i];
        tmp /= retShape[i];
        if (static_cast<int64_t>(self.shape().size()) > i && self.shape()[i] != 1) {
          aOffset += index * aStrides[i];
        }
        if (static_cast<int64_t>(other.shape().size()) > i && other.shape()[i] != 1) {
          bOffset += index * bStrides[i];
        }
      }

      gemm(retPtr + batch * m * n, selfPtr + aOffset, otherPtr + bOffset, m, k, n, false, false, self.device().index);
    }
  } else {
    gemm(retPtr, selfPtr, otherPtr, m, k, n, false, false, self.device().index);
    if (prependA) {
      retTensor.reshape_({n});
    }
  }

  // reduce dimension if necessary
  if (appendB) {
    if (prependA) {
      retTensor.reshape_({});
    } else {
      retTensor.reshape_({m});
    }
  }

  return retTensor;
}

template <typename T>
Tensor matmulTransOpImpl(const Tensor &self, const Tensor &other, bool transA, bool transB) {
  // fast path
  if (self.dim() == 2 && other.dim() == 2) {
    // a[m, k], b[k, n] -> [m, n]
    int64_t m = self.shape(transA ? 1 : 0);
    int64_t k = self.shape(transA ? 0 : 1);
    int64_t n = other.shape(transB ? 0 : 1);
    if (k != other.shape(transB ? 1 : 0)) {
      ASSERT(false && "matmul error: shape not aligned");
      return {};
    }
    Tensor retTensor = Tensor::empty({m, n}, self.options().noGrad());

    const T *selfPtr = self.dataPtr<T>();
    const T *otherPtr = other.dataPtr<T>();
    T *retPtr = retTensor.dataPtr<T>();

    auto gemm = getGemmFunc<T>(self.device().type);
    gemm(retPtr, selfPtr, otherPtr, m, k, n, transA, transB, self.device().index);
    return retTensor;
  }

  // slow path
  return matmul(transA ? self.permute() : self, transB ? other.permute() : other);
}

void registerLinalgCommon() {
  REGISTER_OP_IMPL_ALL_DEVICES(matmul, Float32, &(matmulOpImpl<DTypeToType_t<DType::Float32>>));
  REGISTER_OP_IMPL_ALL_DEVICES(matmulTrans, Float32, &(matmulTransOpImpl<DTypeToType_t<DType::Float32>>));
}

}  // namespace tinytorch::op
