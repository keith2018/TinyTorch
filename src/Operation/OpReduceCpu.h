/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "OpElemWise.h"
#include "OpReduce.h"

namespace tinytorch::op {

struct OpCpuReduceMin {
  template <typename T>
  static T apply(const T &a, const T &b) {
    return std::min(a, b);
  }

  template <typename T>
  static bool compare(const T &a, const T &b) {
    return a < b;
  }

  template <typename T>
  static T defaultVal() {
    return std::numeric_limits<T>::max();
  }
};

struct OpCpuReduceMax {
  template <typename T>
  static T apply(const T &a, const T &b) {
    return std::max(a, b);
  }

  template <typename T>
  static bool compare(const T &a, const T &b) {
    return a > b;
  }

  template <typename T>
  static T defaultVal() {
    return -std::numeric_limits<T>::max();
  }
};

struct OpCpuReduceSum {
  template <typename T>
  static T apply(const T &a, const T &b) {
    return a + b;
  }

  template <typename T>
  static T defaultVal() {
    return 0;
  }
};

class ReducerCpu {
 public:
  static int64_t getReduceSrcIndex(const Tensor &ret, const Tensor &t, int64_t idx, int64_t dim, bool keepDims);
  static int64_t getReduceDstIndex(const Tensor &t, int64_t idx, int64_t dim);
  static int64_t getReduceDstIndex(const Tensor &t, int64_t idx, const DimArray<int64_t> &inAxis);

  template <typename T, typename OP>
  static void reduceAll(T *output, const T *input, int64_t n);

  template <typename T, typename OP>
  static void reduceIdxAll(int64_t *output, const T *input, int64_t n);

  template <typename T, typename OP, bool IsLastDim>
  static void reduceIdxDimImpl(Tensor &values, Tensor &indices, const Tensor &t, int64_t dim, bool keepDims);
  template <typename T, typename OP>
  static std::pair<Tensor, Tensor> reduceIdxDim(const Tensor &t, int64_t dim, bool keepDims);

  template <typename T, typename Func>
  static Tensor reduceMultiDim(const Tensor &t, IntArrayView dims, bool keepDims, Func func);
};

template <typename T, typename OP>
void ReducerCpu::reduceAll(T *output, const T *input, int64_t n) {
  T val = OP::template defaultVal<T>();
  for (int64_t i = 0; i < n; i++) {
    val = OP::template apply<T>(input[i], val);
  }
  *output = val;
}

template <typename T, typename OP>
void ReducerCpu::reduceIdxAll(int64_t *output, const T *input, int64_t n) {
  T val = OP::template defaultVal<T>();
  int64_t valIdx = 0;
  for (int64_t i = 0; i < n; i++) {
    if (OP::template apply<T>(val, input[i]) != val) {
      val = input[i];
      valIdx = i;
    }
  }
  *output = valIdx;
}

template <typename T, typename OP, bool IsLastDim>
void ReducerCpu::reduceIdxDimImpl(Tensor &values, Tensor &indices, const Tensor &t, int64_t dim, bool keepDims) {
  ASSERT(values.shape() == indices.shape());
  ASSERT(values.dtype() == t.dtype());
  ASSERT(indices.dtype() == DType::Int64);

  const auto dimSize = t.shape(dim);
  const auto stride = IsLastDim ? 1 : t.stride(dim);

  const T *dataPtr = t.dataPtr<T>();
  T *valuesPtr = values.dataPtr<T>();
  auto *indicesPtr = indices.dataPtr<int64_t>();

  for (int64_t i = 0; i < values.numel(); i++) {
    auto targetVal = OP::template defaultVal<T>();
    int64_t targetIdx = 0;
    int64_t srcIdx = IsLastDim ? i * dimSize : getReduceSrcIndex(values, t, i, dim, keepDims);
    for (int64_t j = 0; j < dimSize; j++) {
      auto val = dataPtr[srcIdx];
      srcIdx += stride;
      if (OP::template compare(val, targetVal)) {
        targetVal = val;
        targetIdx = j;
      }
    }
    valuesPtr[i] = targetVal;
    indicesPtr[i] = targetIdx;
  }
}

template <typename T, typename OP>
std::pair<Tensor, Tensor> ReducerCpu::reduceIdxDim(const Tensor &t, int64_t dim, bool keepDims) {
  if (dim < 0) {
    dim += t.dim();
  }
  if (dim < 0 || dim >= t.dim()) {
    LOGE("Invalid reduce dim: %lld", dim);
    ASSERT(false);
    return {};
  }

  const auto retShape = getReduceShape(t, dim, keepDims);
  auto values = Tensor::empty(retShape.view(), t.options().noGrad());
  auto indices = Tensor::empty(retShape.view(), getIndicesOptions(t));

  if (dim == t.dim() - 1) {
    reduceIdxDimImpl<T, OP, true>(values, indices, t, dim, keepDims);
  } else {
    reduceIdxDimImpl<T, OP, false>(values, indices, t, dim, keepDims);
  }
  return {values, indices};
}

template <typename T, typename Func>
Tensor ReducerCpu::reduceMultiDim(const Tensor &t, const IntArrayView dims, bool keepDims, Func func) {
  DimArray<int64_t> inAxis{};
  for (int64_t d : dims) {
    if (d < 0) {
      d += t.dim();
    }
    if (d < 0 || d >= t.dim()) {
      LOGE("Invalid reduce dim: %lld", d);
      ASSERT(false);
      return {};
    }
    inAxis.data[d] = 1;
  }

  const auto retShape = getReduceShape(t, inAxis, keepDims);
  auto ret = Tensor::zeros(retShape.view(), t.options().noGrad());

  const T *tPtr = t.dataPtr<T>();
  T *retPtr = ret.dataPtr<T>();
  for (int64_t srcIdx = 0; srcIdx < t.numel(); srcIdx++) {
    int64_t retIdx = getReduceDstIndex(t, srcIdx, inAxis);
    func(retPtr, tPtr, retIdx, srcIdx);
  }
  return ret;
}

template <typename T, typename OP>
Tensor reduceOpAllCpuImpl(const Tensor &t) {
  if (t.isScalar()) {
    return t;
  }
  auto ret = Tensor::empty({}, t.options().noGrad());
  const T *tPtr = t.dataPtr<T>();
  T *retPtr = ret.dataPtr<T>();
  ReducerCpu::reduceAll<T, OP>(retPtr, tPtr, t.numel());
  return ret;
}

template <typename T, typename OP>
Tensor reduceOpArgMinMaxCpuImpl(const Tensor &t) {
  if (t.isScalar()) {
    return Tensor::scalar(0, getIndicesOptions(t));
  }
  auto ret = Tensor::empty({}, getIndicesOptions(t));
  const T *tPtr = t.dataPtr<T>();
  auto *retPtr = ret.dataPtr<int64_t>();
  ReducerCpu::reduceIdxAll<T, OP>(retPtr, tPtr, t.numel());
  return ret;
}

template <typename T, typename OP>
TensorPair reduceOpMinMaxDimCpuImpl(const Tensor &t, int64_t dim, bool keepDims = false) {
  if (t.isScalar()) {
    return {t, Tensor::scalar(0, getIndicesOptions(t))};
  }
  return ReducerCpu::reduceIdxDim<T, OP>(t, dim, keepDims);
}

template <typename T>
Tensor reduceOpSumDimsCpuImpl(const Tensor &t, const IntArrayView dims, bool keepDims = false) {
  if (t.isScalar()) {
    return t;
  }
  ASSERT(!dims.empty());
  return ReducerCpu::reduceMultiDim<T>(t, dims, keepDims, [](T *retPtr, const T *tPtr, int64_t retIdx, int64_t srcIdx) {
    retPtr[retIdx] += tPtr[srcIdx];
  });
}

template <typename T>
Tensor reduceOpSumDimCpuImpl(const Tensor &t, const int64_t dim, bool keepDims = false) {
  return reduceOpSumDimsCpuImpl<T>(t, {dim}, keepDims);
}

template <typename T>
Tensor reduceOpSumCpuImpl(const Tensor &t) {
  return reduceOpAllCpuImpl<T, OpCpuReduceSum>(t);
}

template <typename T>
Tensor reduceOpMeanCpuImpl(const Tensor &t) {
  if (t.isScalar()) {
    return t;
  }
  auto ret = reduceOpAllCpuImpl<T, OpCpuReduceSum>(t);
  const auto r = 1.f / static_cast<float>(t.numel());
  op::mulInplace(ret, Tensor::scalar(r, ret.options().noGrad()));
  return ret;
}

template <typename T>
Tensor reduceOpMeanDimsCpuImpl(const Tensor &t, const IntArrayView dims, bool keepDims = false) {
  if (t.isScalar()) {
    return t;
  }
  ASSERT(!dims.empty());
  auto ret = reduceOpSumDimsCpuImpl<T>(t, dims, keepDims);
  if (ret.defined()) {
    auto r = static_cast<float>(ret.numel()) / static_cast<float>(t.numel());
    op::mulInplace(ret, Tensor::scalar(r, ret.options().noGrad()));
  }
  return ret;
}

template <typename T>
Tensor reduceOpMeanDimCpuImpl(const Tensor &t, const int64_t dim, bool keepDims = false) {
  return reduceOpMeanDimsCpuImpl<T>(t, {dim}, keepDims);
}

template <typename T>
TensorPair reduceOpVarMeanCpuImpl(const Tensor &t, bool unbiased = true) {
  if (t.isScalar()) {
    return {Tensor::scalar(0, t.options().noGrad()), t};
  }
  const auto meanVal = op::mean(t);

  const T *tPtr = t.dataPtr<T>();
  const T *meanPtr = meanVal.dataPtr<T>();

  T squaredDiff = 0;
  for (int64_t i = 0; i < t.numel(); i++) {
    const auto diff = tPtr[i] - meanPtr[0];
    squaredDiff += diff * diff;
  }
  auto varVal = Tensor::scalar(squaredDiff, t.options().noGrad());
  const auto n = static_cast<float>(t.numel());
  auto r = 1.f / n;
  if (unbiased) {
    r *= n / (n - 1.f);
  }
  op::mulInplace(varVal, Tensor::scalar(r, varVal.options().noGrad()));
  return {varVal, meanVal};
}

template <typename T>
TensorPair reduceOpVarMeanDimsCpuImpl(const Tensor &t, const IntArrayView dims, bool unbiased = true,
                                      bool keepDims = false) {
  if (t.isScalar()) {
    return {Tensor::scalar(0, t.options().noGrad()), t};
  }
  ASSERT(!dims.empty());
  auto meanVal = op::meanOnDims(t, dims, true);
  const T *meanPtr = meanVal.dataPtr<T>();

  auto varVal = ReducerCpu::reduceMultiDim<T>(t, dims, keepDims,
                                              [meanPtr](T *retPtr, const T *tPtr, int64_t retIdx, int64_t srcIdx) {
                                                T diff = tPtr[srcIdx] - meanPtr[retIdx];
                                                retPtr[retIdx] += diff * diff;
                                              });
  if (varVal.defined()) {
    auto reduceSize = static_cast<float>(t.numel()) / static_cast<float>(varVal.numel());
    auto r = 1.f / reduceSize;
    if (unbiased) {
      r *= reduceSize / (reduceSize - 1.f);
    }
    op::mulInplace(varVal, Tensor::scalar(r, varVal.options().noGrad()));
  }
  return {varVal, meanVal};
}

template <typename T>
TensorPair reduceOpVarMeanDimCpuImpl(const Tensor &t, const int64_t dim, bool unbiased = true, bool keepDims = false) {
  return reduceOpVarMeanDimsCpuImpl<T>(t, {dim}, unbiased, keepDims);
}

}  // namespace tinytorch::op
