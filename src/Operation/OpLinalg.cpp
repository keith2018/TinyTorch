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
Tensor matmulOpImplDetail(const Tensor &a, const Tensor &b, bool transA = false, bool transB = false) {
  if (a.dim() == 0 || b.dim() == 0) {
    ASSERT(false && "matmul error: invalid shape");
    return {};
  }

  SizeVector shapeA(a.shape());
  SizeVector shapeB(b.shape());
  bool prependA = false;
  bool appendB = false;

  if (shapeA.size() == 1) {
    shapeA.insert(shapeA.begin(), 1);
    prependA = true;
    transA = false;
  }
  if (shapeB.size() == 1) {
    shapeB.insert(shapeB.end(), 1);
    appendB = true;
    transB = false;
  }

  // check matrix multiplication compatible
  int64_t effectiveADim1 = transA ? shapeA[shapeA.size() - 2] : shapeA.back();
  int64_t effectiveBDim0 = transB ? shapeB.back() : shapeB[shapeB.size() - 2];

  if (effectiveADim1 != effectiveBDim0) {
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
  auto m = transA ? shapeA.back() : shapeA[shapeA.size() - 2];
  auto k = transA ? shapeA[shapeA.size() - 2] : shapeA.back();
  auto n = transB ? shapeB[shapeB.size() - 2] : shapeB.back();

  retShape[retDimCnt - 2] = m;
  retShape[retDimCnt - 1] = n;
  Tensor retTensor = Tensor::empty(retShape, a.options().noGrad());

  const T *selfPtr = a.dataPtr<T>();
  const T *otherPtr = b.dataPtr<T>();
  T *retPtr = retTensor.dataPtr<T>();

  auto gemm = getGemmFunc<T>(a.device().type);

  if (retDimCnt > 2) {
    // batched matrix multiply with broadcasting
    int64_t batchSize = retTensor.numel() / (m * n);

    SizeVector aStrides(a.strides());
    SizeVector bStrides(b.strides());
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
        if (static_cast<int64_t>(a.shape().size()) > i && a.shape()[i] != 1) {
          aOffset += index * aStrides[i];
        }
        if (static_cast<int64_t>(b.shape().size()) > i && b.shape()[i] != 1) {
          bOffset += index * bStrides[i];
        }
      }

      gemm(retPtr + batch * m * n, selfPtr + aOffset, otherPtr + bOffset, m, k, n, transA, transB, a.device().index);
    }
  } else {
    gemm(retPtr, selfPtr, otherPtr, m, k, n, transA, transB, a.device().index);
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
Tensor matmulOpImpl(const Tensor &a, const Tensor &b, bool transA, bool transB) {
  // fast path
  if (a.dim() == 2 && b.dim() == 2) {
    // a[m, k], b[k, n] -> [m, n]
    int64_t m = a.shape(transA ? 1 : 0);
    int64_t k = a.shape(transA ? 0 : 1);
    int64_t n = b.shape(transB ? 0 : 1);
    if (k != b.shape(transB ? 1 : 0)) {
      ASSERT(false && "matmul error: shape not aligned");
      return {};
    }
    Tensor retTensor = Tensor::empty({m, n}, a.options().noGrad());

    const T *selfPtr = a.dataPtr<T>();
    const T *otherPtr = b.dataPtr<T>();
    T *retPtr = retTensor.dataPtr<T>();

    auto gemm = getGemmFunc<T>(a.device().type);
    gemm(retPtr, selfPtr, otherPtr, m, k, n, transA, transB, a.device().index);
    return retTensor;
  }

  // slow path
  return matmulOpImplDetail<T>(a, b, transA, transB);
}

void registerLinalgCommon() {
  REGISTER_OP_IMPL_ALL_DEVICES(matmul, Float32, &(matmulOpImpl<DTypeToType_t<DType::Float32>>));
  REGISTER_OP_IMPL_ALL_DEVICES(matmul, Float16, &(matmulOpImpl<DTypeToType_t<DType::Float16>>));
  REGISTER_OP_IMPL_ALL_DEVICES(matmul, BFloat16, &(matmulOpImpl<DTypeToType_t<DType::BFloat16>>));
}

}  // namespace tinytorch::op
