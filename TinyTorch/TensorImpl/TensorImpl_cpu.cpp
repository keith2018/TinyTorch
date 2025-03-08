/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "TensorImpl_cpu.h"

#include <algorithm>
#include <cassert>

#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include "cblas.h"
#endif
#endif

#include "TensorImpl_cpu.inc"

namespace TinyTorch {

std::optional<unsigned long> RandomGeneratorCPU::seed_;
static std::random_device _r;
std::default_random_engine RandomGeneratorCPU::randomEngine_(_r());

void AllocatorCPU::allocate(void** ptr, size_t size) {
  *ptr = std::malloc(size);
}

void AllocatorCPU::deallocate(void* ptr) {
  if (ptr) {
    std::free(ptr);
  }
}

TensorOpsCPU::TensorOpsCPU() {
  allocator_.setBaseAllocator(std::make_shared<AllocatorCPU>());
}

TensorOpsCPU::~TensorOpsCPU() { allocator_.clear(); }

template <typename OP>
void TensorOpsCPU::opSingle_(TensorImpl& t) {
  OP opFunc;
  for (int32_t i = 0; i < t.elemCount_; i++) {
    opFunc(t.data_[i]);
  }
}

template <typename OP>
TensorImpl TensorOpsCPU::opSingle(const TensorImpl& t) {
  OP opFunc;
  auto result = TensorImpl::shape(t.shape(), t.device_);
  for (int32_t i = 0; i < t.elemCount_; i++) {
    result.data_[i] = opFunc(t.data_[i]);
  }
  return result;
}

template <typename OP>
TensorImpl TensorOpsCPU::opPair(const TensorImpl& a, const TensorImpl& b) {
  OP opFunc;
  auto result = TensorImpl::shape(a.shape(), a.device_);
  for (int32_t i = 0; i < a.elemCount_; i++) {
    result.data_[i] = opFunc(a.data_[i], b.data_[i]);
  }
  return result;
}

template <typename OP>
TensorImpl TensorOpsCPU::opPair(const TensorImpl& a, float b) {
  OP opFunc;
  auto result = TensorImpl::shape(a.shape(), a.device_);
  for (int32_t i = 0; i < a.elemCount_; i++) {
    result.data_[i] = opFunc(a.data_[i], b);
  }
  return result;
}

template <typename OP>
TensorImpl TensorOpsCPU::opPair(float a, const TensorImpl& b) {
  OP opFunc;
  auto result = TensorImpl::shape(b.shape(), b.device_);
  for (int32_t i = 0; i < b.elemCount_; i++) {
    result.data_[i] = opFunc(a, b.data_[i]);
  }
  return result;
}

template <typename OP>
void TensorOpsCPU::opPair_(TensorImpl& t, float b) {
  OP opFunc;
  for (int32_t i = 0; i < t.elemCount_; i++) {
    t.data_[i] = opFunc(t.data_[i], b);
  }
}

template <typename OP>
void TensorOpsCPU::opPair_(TensorImpl& t, const TensorImpl& b) {
  OP opFunc;
  for (int32_t i = 0; i < t.elemCount_; i++) {
    t.data_[i] = opFunc(t.data_[i], b.data_[i]);
  }
}

template <typename OP>
void TensorOpsCPU::broadcastFastPass(TensorImpl& result,
                                     const TensorImpl& larger,
                                     const TensorImpl& smaller, bool reverse) {
  OP opFunc;
  int32_t n = result.elemCount_ / smaller.elemCount_;
  for (int32_t idx = 0; idx < n; idx++) {
    for (int32_t i = 0; i < smaller.elemCount_; i++) {
      auto& dataA = larger.data_[i + idx * smaller.elemCount_];
      auto& dataB = smaller.data_[i];
      auto& dataRet = result.data_[i + idx * smaller.elemCount_];
      dataRet = reverse ? opFunc(dataB, dataA) : opFunc(dataA, dataB);
    }
  }
}

template <typename OP>
void TensorOpsCPU::broadcastImpl(TensorImpl& result, const TensorImpl& a,
                                 const TensorImpl& b) {
  OP opFunc;

  // fast pass 1
  if (a.elemCount_ == result.elemCount_ && isLeadingOnes(b.shape())) {
    broadcastFastPass<OP>(result, a, b, false);
    return;
  }

  // fast pass 2
  if (b.elemCount_ == result.elemCount_ && isLeadingOnes(a.shape())) {
    broadcastFastPass<OP>(result, b, a, true);
    return;
  }

  // slow pass
  static int32_t cIndices[TENSOR_MAX_DIMS];
  static int32_t aIndices[TENSOR_MAX_DIMS];
  static int32_t bIndices[TENSOR_MAX_DIMS];
  memset(aIndices, 0, TENSOR_MAX_DIMS * sizeof(int32_t));
  memset(bIndices, 0, TENSOR_MAX_DIMS * sizeof(int32_t));
  for (int32_t i = 0; i < result.elemCount_; i++) {
    offsetToIndices(cIndices, result.shape_, i);
    for (int32_t j = 0; j < result.dimCount_; j++) {
      if (j >= result.dimCount_ - a.dimCount_) {
        int32_t aIndex = j - (result.dimCount_ - a.dimCount_);
        if (a.shape()[aIndex] != 1) {
          aIndices[aIndex] = cIndices[j];
        }
      }
      if (j >= result.dimCount_ - b.dimCount_) {
        int32_t bIndex = j - (result.dimCount_ - b.dimCount_);
        if (b.shape()[bIndex] != 1) {
          bIndices[bIndex] = cIndices[j];
        }
      }
    }
    auto aIdx = indicesToOffset(a.strides_, aIndices);
    auto bIdx = indicesToOffset(b.strides_, bIndices);
    result.data_[i] = opFunc(a.data_[aIdx], b.data_[bIdx]);
  }
}

template <typename OP>
TensorImpl TensorOpsCPU::opPairBroadcast(const TensorImpl& a,
                                         const TensorImpl& b) {
  Shape retShape;
  auto comp = checkShapeCompatible(a.shape(), b.shape(), retShape);
  if (comp == ShapeCompatible_Error) {
    error(__FUNCTION__, TensorError_ShapeNotAligned);
    return {};
  }
  if (comp == ShapeCompatible_SameShape) {
    return opPair<OP>(a, b);
  }

  auto result = TensorImpl::shape(retShape, a.device_);
  broadcastImpl<OP>(result, a, b);
  return result;
}

template <typename OP>
void TensorOpsCPU::opPairBroadcast_(TensorImpl& a, const TensorImpl& b) {
  Shape retShape;
  auto comp = checkShapeCompatible(a.shape(), b.shape(), retShape);
  if (comp == ShapeCompatible_Error) {
    error(__FUNCTION__, TensorError_ShapeNotAligned);
    return;
  }
  if (comp == ShapeCompatible_SameShape) {
    opPair_<OP>(a, b);
    return;
  }

  auto result = TensorImpl::shape(retShape, a.device_);
  broadcastImpl<OP>(result, a, b);
  a = std::move(result);
}

int32_t TensorOpsCPU::getReduceSrcIndex(const TensorImpl& ret,
                                        const TensorImpl& t, int32_t idx,
                                        int32_t dim, bool keepDims) {
  int32_t outIndex = idx;
  int32_t inIndex = 0;
  for (int32_t d = ret.dimCount_ - 1; d >= 0; d--) {
    int32_t coord = outIndex % ret.shape_[d];
    outIndex /= ret.shape_[d];
    if (keepDims || d < dim) {
      inIndex += coord * t.strides_[d];
    } else {
      inIndex += coord * t.strides_[d + 1];
    }
  }
  return inIndex;
}

int32_t TensorOpsCPU::getReduceDstIndex(const TensorImpl& t, int32_t idx,
                                        int32_t dim) {
  int32_t retIdx = 0;
  int32_t stride = 1;
  for (int32_t d = t.dimCount_ - 1; d >= 0; d--) {
    if (d != dim) {
      retIdx += (idx / t.strides_[d] % t.shape_[d]) * stride;
      stride *= t.shape_[d];
    }
  }
  return retIdx;
}

int32_t TensorOpsCPU::getReduceDstIndex(const TensorImpl& t, int32_t idx,
                                        const FixedVector<uint8_t>& inAxis) {
  int32_t retIdx = 0;
  int32_t stride = 1;
  for (int32_t d = t.dimCount_ - 1; d >= 0; d--) {
    if (0 == inAxis.data[d]) {
      retIdx += (idx / t.strides_[d] % t.shape_[d]) * stride;
      stride *= t.shape_[d];
    }
  }
  return retIdx;
}

template <typename Compare, bool IsLastDim>
void TensorOpsCPU::reduceImpl(TensorImpl& ret, const TensorImpl& t, int32_t dim,
                              bool keepDims, float initVal, Compare comp,
                              bool returnIndex) {
  auto dimSize = t.shape_[dim];
  auto stride = IsLastDim ? 1 : t.strides_[dim];

  for (int32_t i = 0; i < ret.elemCount_; i++) {
    auto targetVal = initVal;
    int32_t targetIdx = 0;
    int32_t srcIdx =
        IsLastDim ? i * dimSize : getReduceSrcIndex(ret, t, i, dim, keepDims);
    for (int32_t j = 0; j < dimSize; j++) {
      auto val = t.data_[srcIdx];
      srcIdx += stride;
      if (comp(val, targetVal)) {
        targetVal = val;
        targetIdx = j;
      }
    }
    ret.data_[i] = returnIndex ? static_cast<float>(targetIdx) : targetVal;
  }
}

template <typename Compare>
TensorImpl TensorOpsCPU::reduce(const TensorImpl& t, int32_t dim, bool keepDims,
                                float initVal, Compare comp, bool returnIndex) {
  if (dim < 0) {
    dim += t.dimCount_;
  }
  if (dim < 0 || dim >= t.dimCount_) {
    error(__FUNCTION__, TensorError_InvalidAxis);
    return {};
  }

  auto retShape = getReduceShape(t, dim, keepDims);
  auto ret = TensorImpl::shape(retShape, t.device_);
  if (dim == t.dimCount_ - 1) {
    reduceImpl<Compare, true>(ret, t, dim, keepDims, initVal, comp,
                              returnIndex);
  } else {
    reduceImpl<Compare, false>(ret, t, dim, keepDims, initVal, comp,
                               returnIndex);
  }
  return ret;
}

void TensorOpsCPU::getSubIndices(
    int32_t* subIndices, const TensorImpl& t,
    const std::vector<std::reference_wrapper<TensorImpl>>& indices,
    int32_t idx) {
  for (int32_t i = 0; i < indices.size(); i++) {
    auto ind = (int32_t)indices[i].get().data_[idx];
    subIndices[i] = ind >= 0 ? ind : ind + t.shape_[i];
  }
}

void TensorOpsCPU::allocate(void** ptr, size_t size) {
  allocator_.allocate(ptr, size);
}

void TensorOpsCPU::deallocate(void* ptr) { allocator_.deallocate(ptr); }

void TensorOpsCPU::copyHostToDevice(void* dst, const void* src, size_t count) {
  std::memcpy(dst, src, count);
}

void TensorOpsCPU::copyOnDevice(void* dst, const void* src, size_t count) {
  std::memcpy(dst, src, count);
}

void TensorOpsCPU::copyDeviceToHost(void* dst, const void* src, size_t count) {
  std::memcpy(dst, src, count);
}

void TensorOpsCPU::fillConstant_(float* dst, float val, size_t count) {
  std::fill(dst, dst + count, val);
}

void TensorOpsCPU::fillConstant_(TensorImpl& t, float val) {
  std::fill(t.data_, t.data_ + t.elemCount_, val);
}

void TensorOpsCPU::fillLinSpace_(float* dst, float start, float step,
                                 size_t count) {
  for (size_t i = 0; i < count; i++) {
    dst[i] = start + (float)i * step;
  }
}

void TensorOpsCPU::fillRandUniform_(TensorImpl& t, float min, float max) {
  auto generator = RandomGeneratorCPU::getGenerator();
  std::uniform_real_distribution distribution(min, max);
  for (int32_t i = 0; i < t.elemCount_; i++) {
    t.data_[i] = distribution(generator);
  }
}

void TensorOpsCPU::fillRandNormal_(TensorImpl& t) {
  auto generator = RandomGeneratorCPU::getGenerator();
  std::normal_distribution distribution(0.0f, 1.0f);
  for (int32_t i = 0; i < t.elemCount_; i++) {
    t.data_[i] = distribution(generator);
  }
}

void TensorOpsCPU::fillRandBernoulli_(TensorImpl& t, float p) {
  auto generator = RandomGeneratorCPU::getGenerator();
  std::bernoulli_distribution distribution(p);
  for (int32_t i = 0; i < t.elemCount_; i++) {
    t.data_[i] = distribution(generator);
  }
}

TensorImpl TensorOpsCPU::add(const TensorImpl& a, const TensorImpl& b) {
  if (a.dimCount_ == 0) {
    return add(a.data_[0], b);
  }
  if (b.dimCount_ == 0) {
    return add(a, b.data_[0]);
  }
  return opPairBroadcast<OpCpuAdd>(a, b);
}

TensorImpl TensorOpsCPU::sub(const TensorImpl& a, const TensorImpl& b) {
  if (a.dimCount_ == 0) {
    return sub(a.data_[0], b);
  }
  if (b.dimCount_ == 0) {
    return sub(a, b.data_[0]);
  }
  return opPairBroadcast<OpCpuSub>(a, b);
}

TensorImpl TensorOpsCPU::mul(const TensorImpl& a, const TensorImpl& b) {
  if (a.dimCount_ == 0) {
    return mul(a.data_[0], b);
  }
  if (b.dimCount_ == 0) {
    return mul(a, b.data_[0]);
  }
  return opPairBroadcast<OpCpuMul>(a, b);
}

TensorImpl TensorOpsCPU::div(const TensorImpl& a, const TensorImpl& b) {
  if (a.dimCount_ == 0) {
    return div(a.data_[0], b);
  }
  if (b.dimCount_ == 0) {
    return div(a, b.data_[0]);
  }
  return opPairBroadcast<OpCpuDiv>(a, b);
}

TensorImpl TensorOpsCPU::pow(const TensorImpl& a, const TensorImpl& b) {
  if (b.dimCount_ == 0) {
    return pow(a, b.data_[0]);
  }
  return opPairBroadcast<OpCpuPow>(a, b);
}

TensorImpl TensorOpsCPU::add(const TensorImpl& a, const float& b) {
  return opPair<OpCpuAdd>(a, b);
}

TensorImpl TensorOpsCPU::sub(const TensorImpl& a, const float& b) {
  return opPair<OpCpuSub>(a, b);
}

TensorImpl TensorOpsCPU::mul(const TensorImpl& a, const float& b) {
  return opPair<OpCpuMul>(a, b);
}

TensorImpl TensorOpsCPU::div(const TensorImpl& a, const float& b) {
  return opPair<OpCpuDiv>(a, b);
}

TensorImpl TensorOpsCPU::pow(const TensorImpl& a, const float& b) {
  return opPair<OpCpuPow>(a, b);
}

TensorImpl TensorOpsCPU::add(const float& a, const TensorImpl& b) {
  return opPair<OpCpuAdd>(a, b);
}

TensorImpl TensorOpsCPU::sub(const float& a, const TensorImpl& b) {
  return opPair<OpCpuSub>(a, b);
}

TensorImpl TensorOpsCPU::mul(const float& a, const TensorImpl& b) {
  return opPair<OpCpuMul>(a, b);
}

TensorImpl TensorOpsCPU::div(const float& a, const TensorImpl& b) {
  return opPair<OpCpuDiv>(a, b);
}

void TensorOpsCPU::add_(TensorImpl& a, const TensorImpl& b) {
  if (a.dimCount_ == 0) {
    a = add(a.data_[0], b);
    return;
  }
  if (b.dimCount_ == 0) {
    add_(a, b.data_[0]);
    return;
  }
  opPairBroadcast_<OpCpuAdd>(a, b);
}

void TensorOpsCPU::sub_(TensorImpl& a, const TensorImpl& b) {
  if (a.dimCount_ == 0) {
    a = sub(a.data_[0], b);
    return;
  }
  if (b.dimCount_ == 0) {
    sub_(a, b.data_[0]);
    return;
  }
  opPairBroadcast_<OpCpuSub>(a, b);
}

void TensorOpsCPU::mul_(TensorImpl& a, const TensorImpl& b) {
  if (a.dimCount_ == 0) {
    a = mul(a.data_[0], b);
    return;
  }
  if (b.dimCount_ == 0) {
    mul_(a, b.data_[0]);
    return;
  }
  opPairBroadcast_<OpCpuMul>(a, b);
}

void TensorOpsCPU::div_(TensorImpl& a, const TensorImpl& b) {
  if (a.dimCount_ == 0) {
    a = div(a.data_[0], b);
    return;
  }
  if (b.dimCount_ == 0) {
    div_(a, b.data_[0]);
    return;
  }
  opPairBroadcast_<OpCpuDiv>(a, b);
}

void TensorOpsCPU::add_(TensorImpl& a, const float& b) {
  opPair_<OpCpuAdd>(a, b);
}

void TensorOpsCPU::sub_(TensorImpl& a, const float& b) {
  opPair_<OpCpuSub>(a, b);
}

void TensorOpsCPU::mul_(TensorImpl& a, const float& b) {
  opPair_<OpCpuMul>(a, b);
}

void TensorOpsCPU::div_(TensorImpl& a, const float& b) {
  opPair_<OpCpuDiv>(a, b);
}

TensorImpl TensorOpsCPU::eq(const TensorImpl& a, const TensorImpl& b) {
  if (b.dimCount_ == 0) {
    return eq(a, b.data_[0]);
  }
  return opPairBroadcast<OpCpuEq>(a, b);
}

TensorImpl TensorOpsCPU::ne(const TensorImpl& a, const TensorImpl& b) {
  if (b.dimCount_ == 0) {
    return ne(a, b.data_[0]);
  }
  return opPairBroadcast<OpCpuNe>(a, b);
}

TensorImpl TensorOpsCPU::ge(const TensorImpl& a, const TensorImpl& b) {
  if (b.dimCount_ == 0) {
    return ge(a, b.data_[0]);
  }
  return opPairBroadcast<OpCpuGe>(a, b);
}

TensorImpl TensorOpsCPU::gt(const TensorImpl& a, const TensorImpl& b) {
  if (b.dimCount_ == 0) {
    return gt(a, b.data_[0]);
  }
  return opPairBroadcast<OpCpuGt>(a, b);
}

TensorImpl TensorOpsCPU::le(const TensorImpl& a, const TensorImpl& b) {
  if (b.dimCount_ == 0) {
    return le(a, b.data_[0]);
  }
  return opPairBroadcast<OpCpuLe>(a, b);
}

TensorImpl TensorOpsCPU::lt(const TensorImpl& a, const TensorImpl& b) {
  if (b.dimCount_ == 0) {
    return lt(a, b.data_[0]);
  }
  return opPairBroadcast<OpCpuLt>(a, b);
}

TensorImpl TensorOpsCPU::maximum(const TensorImpl& a, const TensorImpl& b) {
  if (b.dimCount_ == 0) {
    return maximum(a, b.data_[0]);
  }
  return opPairBroadcast<OpCpuMax>(a, b);
}

TensorImpl TensorOpsCPU::minimum(const TensorImpl& a, const TensorImpl& b) {
  if (b.dimCount_ == 0) {
    return minimum(a, b.data_[0]);
  }
  return opPairBroadcast<OpCpuMin>(a, b);
}

TensorImpl TensorOpsCPU::eq(const TensorImpl& a, const float& b) {
  return opPair<OpCpuEq>(a, b);
}

TensorImpl TensorOpsCPU::ne(const TensorImpl& a, const float& b) {
  return opPair<OpCpuNe>(a, b);
}

TensorImpl TensorOpsCPU::ge(const TensorImpl& a, const float& b) {
  return opPair<OpCpuGe>(a, b);
}

TensorImpl TensorOpsCPU::gt(const TensorImpl& a, const float& b) {
  return opPair<OpCpuGt>(a, b);
}

TensorImpl TensorOpsCPU::le(const TensorImpl& a, const float& b) {
  return opPair<OpCpuLe>(a, b);
}

TensorImpl TensorOpsCPU::lt(const TensorImpl& a, const float& b) {
  return opPair<OpCpuLt>(a, b);
}

TensorImpl TensorOpsCPU::maximum(const TensorImpl& a, const float& b) {
  return opPair<OpCpuMax>(a, b);
}

TensorImpl TensorOpsCPU::minimum(const TensorImpl& a, const float& b) {
  return opPair<OpCpuMin>(a, b);
}

void TensorOpsCPU::sin_(TensorImpl& t) { opSingle_<OpCpuSin_>(t); }

void TensorOpsCPU::cos_(TensorImpl& t) { opSingle_<OpCpuCos_>(t); }

void TensorOpsCPU::sqrt_(TensorImpl& t) { opSingle_<OpCpuSqrt_>(t); }

void TensorOpsCPU::tanh_(TensorImpl& t) { opSingle_<OpCpuTanh_>(t); }

void TensorOpsCPU::exp_(TensorImpl& t) { opSingle_<OpCpuExp_>(t); }

void TensorOpsCPU::log_(TensorImpl& t) { opSingle_<OpCpuLog_>(t); }

TensorImpl TensorOpsCPU::sin(const TensorImpl& t) {
  return opSingle<OpCpuSin>(t);
}

TensorImpl TensorOpsCPU::cos(const TensorImpl& t) {
  return opSingle<OpCpuCos>(t);
}

TensorImpl TensorOpsCPU::sqrt(const TensorImpl& t) {
  return opSingle<OpCpuSqrt>(t);
}

TensorImpl TensorOpsCPU::tanh(const TensorImpl& t) {
  return opSingle<OpCpuTanh>(t);
}

TensorImpl TensorOpsCPU::exp(const TensorImpl& t) {
  return opSingle<OpCpuExp>(t);
}

TensorImpl TensorOpsCPU::log(const TensorImpl& t) {
  return opSingle<OpCpuLog>(t);
}

void TensorOpsCPU::clampMin_(TensorImpl& t, float min) {
  opPair_<OpCpuMax>(t, min);
}

void TensorOpsCPU::clampMax_(TensorImpl& t, float max) {
  opPair_<OpCpuMin>(t, max);
}

void TensorOpsCPU::clamp_(TensorImpl& t, float min, float max) {
  for (int32_t i = 0; i < t.elemCount_; i++) {
    t.data_[i] = std::max(min, std::min(t.data_[i], max));
  }
}

TensorImpl TensorOpsCPU::clampMin(const TensorImpl& t, float min) {
  return opPair<OpCpuMax>(t, min);
}

TensorImpl TensorOpsCPU::clampMax(const TensorImpl& t, float max) {
  return opPair<OpCpuMin>(t, max);
}

TensorImpl TensorOpsCPU::clamp(const TensorImpl& t, float min, float max) {
  auto result = TensorImpl::shape(t.shape(), t.device_);
  for (int32_t i = 0; i < t.elemCount_; i++) {
    result.data_[i] = std::max(min, std::min(t.data_[i], max));
  }
  return result;
}

TensorImpl TensorOpsCPU::min(const TensorImpl& t) {
  if (t.dimCount_ == 0) {
    return t;
  }
  float min = std::numeric_limits<float>::max();
  for (int32_t i = 0; i < t.elemCount_; i++) {
    min = std::min(t.data_[i], min);
  }
  return TensorImpl::scalar(min, t.device_);
}

TensorImpl TensorOpsCPU::max(const TensorImpl& t) {
  if (t.dimCount_ == 0) {
    return t;
  }
  float max = -std::numeric_limits<float>::max();
  for (int32_t i = 0; i < t.elemCount_; i++) {
    max = std::max(t.data_[i], max);
  }
  return TensorImpl::scalar(max, t.device_);
}

TensorImpl TensorOpsCPU::sum(const TensorImpl& t) {
  if (t.dimCount_ == 0) {
    return t;
  }
  float sum = 0;
  for (int32_t i = 0; i < t.elemCount_; i++) {
    sum += t.data_[i];
  }
  return TensorImpl::scalar(sum, t.device_);
}

TensorImpl TensorOpsCPU::mean(const TensorImpl& t) {
  if (t.dimCount_ == 0) {
    return t;
  }
  float sum = 0;
  for (int32_t i = 0; i < t.elemCount_; i++) {
    sum += t.data_[i];
  }
  auto mean = sum / (float)t.elemCount_;
  return TensorImpl::scalar(mean, t.device_);
}

TensorImpl TensorOpsCPU::var(const TensorImpl& t, bool unbiased) {
  if (t.dimCount_ == 0) {
    return TensorImpl::scalar(0, t.device_);
  }
  float sum = 0;
  for (int32_t i = 0; i < t.elemCount_; i++) {
    sum += t.data_[i];
  }
  auto mean = sum / (float)t.elemCount_;
  auto var = 0.f;
  for (int32_t i = 0; i < t.elemCount_; i++) {
    var += std::pow(t.data_[i] - mean, 2.f);
  }
  var /= (float)t.elemCount_;
  if (unbiased) {
    var *= (float)t.elemCount_ / ((float)t.elemCount_ - 1.f);
  }
  return TensorImpl::scalar(var, t.device_);
}

TensorImpl TensorOpsCPU::argmin(const TensorImpl& t) {
  if (t.dimCount_ == 0) {
    return TensorImpl::scalar(0, t.device_);
  }
  float min = std::numeric_limits<float>::max();
  int32_t minIdx = 0;
  for (int32_t i = 0; i < t.elemCount_; i++) {
    if (t.data_[i] < min) {
      min = t.data_[i];
      minIdx = i;
    }
  }
  return TensorImpl::scalar((float)minIdx, t.device_);
}

TensorImpl TensorOpsCPU::argmax(const TensorImpl& t) {
  if (t.dimCount_ == 0) {
    return TensorImpl::scalar(0, t.device_);
  }
  float max = -std::numeric_limits<float>::max();
  int32_t maxIdx = 0;
  for (int32_t i = 0; i < t.elemCount_; i++) {
    if (t.data_[i] > max) {
      max = t.data_[i];
      maxIdx = i;
    }
  }
  return TensorImpl::scalar((float)maxIdx, t.device_);
}

TensorImpl TensorOpsCPU::min(const TensorImpl& t, int32_t dim, bool keepDims) {
  if (t.dimCount_ == 0) {
    return t;
  }
  return reduce(t, dim, keepDims, std::numeric_limits<float>::max(),
                std::less<>());
}

TensorImpl TensorOpsCPU::max(const TensorImpl& t, int32_t dim, bool keepDims) {
  if (t.dimCount_ == 0) {
    return t;
  }
  return reduce(t, dim, keepDims, -std::numeric_limits<float>::max(),
                std::greater<>());
}

TensorImpl TensorOpsCPU::argmin(const TensorImpl& t, int32_t dim,
                                bool keepDims) {
  if (t.dimCount_ == 0) {
    return TensorImpl::scalar(0, t.device_);
  }
  return reduce(t, dim, keepDims, std::numeric_limits<float>::max(),
                std::less<>(), true);
}

TensorImpl TensorOpsCPU::argmax(const TensorImpl& t, int32_t dim,
                                bool keepDims) {
  if (t.dimCount_ == 0) {
    return TensorImpl::scalar(0, t.device_);
  }
  return reduce(t, dim, keepDims, -std::numeric_limits<float>::max(),
                std::greater<>(), true);
}

TensorImpl TensorOpsCPU::sum(const TensorImpl& t,
                             const std::vector<int32_t>& dims, bool keepDims) {
  if (t.dimCount_ == 0) {
    return t;
  }
  FixedVector<uint8_t> inAxis{};
  int32_t reduceSize = 1;
  for (int32_t d : dims) {
    if (d < 0) {
      d += t.dimCount_;
    }
    if (d < 0 || d >= t.dimCount_) {
      error(__FUNCTION__, TensorError_InvalidAxis);
      return {};
    }
    reduceSize *= t.shape_[d];
    inAxis.data[d] = 1;
  }

  auto retShape = getReduceShape(t, inAxis, keepDims);
  auto ret = TensorImpl::shape(retShape, t.device_);

  fillConstant_(ret, 0);

  for (int32_t i = 0; i < t.elemCount_; i++) {
    int32_t retIdx = getReduceDstIndex(t, i, inAxis);
    ret.data_[retIdx] += t.data_[i];
  }

  return ret;
}

TensorImpl TensorOpsCPU::mean(const TensorImpl& t,
                              const std::vector<int32_t>& dims, bool keepDims) {
  if (t.dimCount_ == 0) {
    return t;
  }
  FixedVector<uint8_t> inAxis{};
  int32_t reduceSize = 1;
  for (int32_t d : dims) {
    if (d < 0) {
      d += t.dimCount_;
    }
    if (d < 0 || d >= t.dimCount_) {
      error(__FUNCTION__, TensorError_InvalidAxis);
      return {};
    }
    reduceSize *= t.shape_[d];
    inAxis.data[d] = 1;
  }

  auto retShape = getReduceShape(t, inAxis, keepDims);
  auto ret = TensorImpl::shape(retShape, t.device_);

  fillConstant_(ret, 0);

  for (int32_t i = 0; i < t.elemCount_; i++) {
    int32_t retIdx = getReduceDstIndex(t, i, inAxis);
    ret.data_[retIdx] += t.data_[i];
  }

  auto r = 1.f / (float)reduceSize;
  mul_(ret, r);
  return ret;
}

TensorImpl TensorOpsCPU::var(const TensorImpl& t,
                             const std::vector<int32_t>& dims, bool unbiased,
                             bool keepDims) {
  if (t.dimCount_ == 0) {
    return TensorImpl::scalar(0, t.device_);
  }
  FixedVector<uint8_t> inAxis{};
  int32_t reduceSize = 1;
  for (int32_t d : dims) {
    if (d < 0) {
      d += t.dimCount_;
    }
    if (d < 0 || d >= t.dimCount_) {
      error(__FUNCTION__, TensorError_InvalidAxis);
      return {};
    }
    reduceSize *= t.shape_[d];
    inAxis.data[d] = 1;
  }

  auto retShape = getReduceShape(t, inAxis, keepDims);
  auto ret = TensorImpl::shape(retShape, t.device_);

  auto meanTensor = mean(t, dims, true);
  fillConstant_(ret, 0);

  for (int32_t i = 0; i < t.elemCount_; i++) {
    int32_t retIdx = getReduceDstIndex(t, i, inAxis);
    float diff = t.data_[i] - meanTensor.data_[retIdx];
    ret.data_[retIdx] += diff * diff;
  }

  auto r = 1.f / (float)reduceSize;
  if (unbiased) {
    r *= (float)reduceSize / ((float)reduceSize - 1.f);
  }
  mul_(ret, r);
  return ret;
}

TensorImpl TensorOpsCPU::permute(const TensorImpl& t,
                                 const std::vector<int32_t>& dims) {
  auto retShape = t.shape_;
  reorderIndices(retShape.data(), dims);
  auto ret = TensorImpl::shape(retShape, t.device_);

  for (int32_t i = 0; i < t.elemCount_; i++) {
    int32_t originIndex = 0;
    int32_t offset = i;
    for (int32_t d = 0; d < t.dimCount_; d++) {
      originIndex += (offset / ret.strides_[d]) * t.strides_[dims[d]];
      offset %= ret.strides_[d];
    }
    ret.data_[i] = t.data_[originIndex];
  }
  return ret;
}

TensorImpl TensorOpsCPU::index(
    const TensorImpl& t,
    const std::vector<std::reference_wrapper<TensorImpl>>& indices) {
  auto len = (int32_t)indices.size();
  auto fistDim = (int32_t)indices[0].get().elemCount_;
  auto dimStride = t.strides_[len - 1];
  Shape retShape = {fistDim};
  for (auto i = len; i < t.dimCount_; i++) {
    retShape.push_back(t.shape_[i]);
  }
  auto retTensor = TensorImpl::shape(retShape, t.device_);

  static int32_t subIndices[TENSOR_MAX_DIMS];
  for (int32_t i = 0; i < fistDim; i++) {
    getSubIndices(subIndices, t, indices, i);
    int32_t dataIdx = indicesToOffset(t.strides_, subIndices);
    copyOnDevice(&retTensor.data_[dimStride * i], &t.data_[dataIdx],
                 dimStride * sizeof(float));
  }

  return retTensor;
}

void TensorOpsCPU::indexPut_(
    TensorImpl& t,
    const std::vector<std::reference_wrapper<TensorImpl>>& indices, float val) {
  auto len = (int32_t)indices.size();
  auto fistDim = (int32_t)indices[0].get().elemCount_;
  auto dimStride = t.strides_[len - 1];

  static int32_t subIndices[TENSOR_MAX_DIMS];
  for (int32_t i = 0; i < fistDim; i++) {
    getSubIndices(subIndices, t, indices, i);
    int32_t dataIdx = indicesToOffset(t.strides_, subIndices);
    fillConstant_(&t.data_[dataIdx], val, dimStride);
  }
}

void TensorOpsCPU::indexPut_(
    TensorImpl& t,
    const std::vector<std::reference_wrapper<TensorImpl>>& indices,
    const TensorImpl& val) {
  auto len = (int32_t)indices.size();
  auto fistDim = (int32_t)indices[0].get().elemCount_;
  auto dimStride = t.strides_[len - 1];
  assert(val.elemCount_ == dimStride * fistDim);

  static int32_t subIndices[TENSOR_MAX_DIMS];
  for (int32_t i = 0; i < fistDim; i++) {
    getSubIndices(subIndices, t, indices, i);
    int32_t dataIdx = indicesToOffset(t.strides_, subIndices);
    copyOnDevice(&t.data_[dataIdx], &val.data_[dimStride * i],
                 dimStride * sizeof(float));
  }
}

TensorImpl TensorOpsCPU::im2col(const TensorImpl& t, Size2D kernel,
                                Size2D stride, Size2D padding) {
  // this: [C, H, W], [N, C, H, W]
  assert(t.dimCount_ == 3 || t.dimCount_ == 4);
  int32_t batch = (t.dimCount_ == 4) ? t.shape_[0] : 1;
  int32_t channels = (t.dimCount_ == 4) ? t.shape_[1] : t.shape_[0];
  int32_t height = (t.dimCount_ == 4) ? t.shape_[2] : t.shape_[1];
  int32_t width = (t.dimCount_ == 4) ? t.shape_[3] : t.shape_[2];
  int32_t outH = (height - kernel.h + 2 * padding.h) / stride.h + 1;
  int32_t outW = (width - kernel.w + 2 * padding.w) / stride.w + 1;

  int32_t colH = outH * outW;
  int32_t colW = channels * kernel.h * kernel.w;
  auto retTensor = TensorImpl::shape({batch * colH, colW}, t.device_);

  int32_t imStride = t.strides_[0];
  for (int32_t n = 0; n < batch; n++) {
    for (int32_t c = 0; c < channels; c++) {
      for (int32_t kh = 0; kh < kernel.h; kh++) {
        for (int32_t kw = 0; kw < kernel.w; kw++) {
          for (int32_t h = 0; h < outH; h++) {
            for (int32_t w = 0; w < outW; w++) {
              int32_t imRow = h * stride.h + kh - padding.h;
              int32_t imCol = w * stride.w + kw - padding.w;
              int32_t colIdx = (n * outH + h) * outW + w;
              int32_t colWIdx = c * kernel.h * kernel.w + kh * kernel.w + kw;
              if (imRow < 0 || imRow >= height || imCol < 0 || imCol >= width) {
                retTensor.data_[colIdx * colW + colWIdx] = 0;  // zero padding
              } else {
                int32_t imgIdx = imCol + width * (imRow + height * c);
                retTensor.data_[colIdx * colW + colWIdx] =
                    t.data_[n * imStride + imgIdx];
              }
            }
          }
        }
      }
    }
  }
  return retTensor;
}

TensorImpl TensorOpsCPU::col2im(const TensorImpl& t, const Shape& shape,
                                Size2D kernel, Size2D stride, Size2D padding) {
  // shape: [C, H, W], [N, C, H, W]
  assert(shape.size() == 3 || shape.size() == 4);
  int32_t batch = (shape.size() == 4) ? shape[0] : 1;
  int32_t channels = (shape.size() == 4) ? shape[1] : shape[0];
  int32_t height = (shape.size() == 4) ? shape[2] : shape[1];
  int32_t width = (shape.size() == 4) ? shape[3] : shape[2];

  auto outH = (height - kernel.h + 2 * padding.h) / stride.h + 1;
  auto outW = (width - kernel.w + 2 * padding.w) / stride.w + 1;

  // int32_t colH = outH * outW;
  int32_t colW = channels * kernel.h * kernel.w;

  auto retTensor = TensorImpl::zeros(shape, t.device_);

  auto imStride = retTensor.strides_[0];
  for (int32_t n = 0; n < batch; n++) {
    for (int32_t c = 0; c < channels; c++) {
      for (int32_t kh = 0; kh < kernel.h; kh++) {
        for (int32_t kw = 0; kw < kernel.w; kw++) {
          for (int32_t h = 0; h < outH; h++) {
            for (int32_t w = 0; w < outW; w++) {
              int32_t imRow = h * stride.h + kh - padding.h;
              int32_t imCol = w * stride.w + kw - padding.w;
              int32_t colIdx = (n * outH + h) * outW + w;
              int32_t colWIdx = c * kernel.h * kernel.w + kh * kernel.w + kw;
              if (imRow >= 0 && imRow < height && imCol >= 0 && imCol < width) {
                int32_t imgIdx = imCol + width * (imRow + height * c);
                retTensor.data_[n * imStride + imgIdx] +=
                    t.data_[colIdx * colW + colWIdx];
              }
            }
          }
        }
      }
    }
  }
  return retTensor;
}

TensorImpl TensorOpsCPU::dot(const TensorImpl& a, const TensorImpl& b) {
  float ret = 0.f;
  for (int32_t i = 0; i < a.elemCount_; i++) {
    ret += a.data_[i] * b.data_[i];
  }
  return TensorImpl::scalar(ret, a.device_);
}

void TensorOpsCPU::gemm(float* c, const float* a, const float* b, int32_t m,
                        int32_t k, int32_t n, bool transA, bool transB) {
#ifdef USE_BLAS
  cblas_sgemm(CblasRowMajor, transA ? CblasTrans : CblasNoTrans,
              transB ? CblasTrans : CblasNoTrans, m, n, k, 1.f, a,
              transA ? m : k, b, transB ? k : n, 0.f, c, n);
  return;
#endif
  for (int i = 0; i < m * n; i++) {
    c[i] = 0.0f;
  }

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int p = 0; p < k; p++) {
        float aVal = transA ? a[p * m + i] : a[i * k + p];
        float bVal = transB ? b[j * k + p] : b[p * n + j];
        c[i * n + j] += aVal * bVal;
      }
    }
  }
}

}  // namespace TinyTorch
