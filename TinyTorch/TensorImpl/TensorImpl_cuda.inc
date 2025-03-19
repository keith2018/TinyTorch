/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

namespace TinyTorch {

#define WARP_SIZE 32
#define TRANSPOSE_TILE_DIM 32

#define FETCH_FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])
#define FETCH_FLOAT3(pointer) (reinterpret_cast<float3*>(&(pointer))[0])
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

struct OpCudaAdd {
  __device__ float operator()(const float a, const float b) const {
    return a + b;
  }
};

struct OpCudaSub {
  __device__ float operator()(const float a, const float b) const {
    return a - b;
  }
};

struct OpCudaMul {
  __device__ float operator()(const float a, const float b) const {
    return a * b;
  }
};

struct OpCudaDiv {
  __device__ float operator()(const float a, const float b) const {
    return a / b;
  }
};

struct OpCudaPow {
  __device__ float operator()(const float a, const float b) const {
    return pow(a, b);
  }
};

struct OpCudaEq {
  __device__ float operator()(const float a, const float b) const {
    return a == b ? 1.f : 0.f;
  }
};

struct OpCudaNe {
  __device__ float operator()(const float a, const float b) const {
    return a != b ? 1.f : 0.f;
  }
};

struct OpCudaLt {
  __device__ float operator()(const float a, const float b) const {
    return a < b ? 1.f : 0.f;
  }
};

struct OpCudaLe {
  __device__ float operator()(const float a, const float b) const {
    return a <= b ? 1.f : 0.f;
  }
};

struct OpCudaGt {
  __device__ float operator()(const float a, const float b) const {
    return a > b ? 1.f : 0.f;
  }
};

struct OpCudaGe {
  __device__ float operator()(const float a, const float b) const {
    return a >= b ? 1.f : 0.f;
  }
};

struct OpCudaMax {
  __device__ float operator()(const float a, const float b) const {
    return max(a, b);
  }
};

struct OpCudaMin {
  __device__ float operator()(const float a, const float b) const {
    return min(a, b);
  }
};

struct OpCudaSin_ {
  __device__ void operator()(float& a) const { a = sin(a); }
};

struct OpCudaCos_ {
  __device__ void operator()(float& a) const { a = cos(a); }
};

struct OpCudaSqrt_ {
  __device__ void operator()(float& a) const { a = sqrt(a); }
};

struct OpCudaTanh_ {
  __device__ void operator()(float& a) const { a = tanh(a); }
};

struct OpCudaExp_ {
  __device__ void operator()(float& a) const { a = exp(a); }
};

struct OpCudaLog_ {
  __device__ void operator()(float& a) const { a = log(a); }
};

struct OpCudaSin {
  __device__ float operator()(const float a) const { return sin(a); }
};

struct OpCudaCos {
  __device__ float operator()(const float a) const { return cos(a); }
};

struct OpCudaSqrt {
  __device__ float operator()(const float a) const { return sqrt(a); }
};

struct OpCudaTanh {
  __device__ float operator()(const float a) const { return tanh(a); }
};

struct OpCudaExp {
  __device__ float operator()(const float a) const { return exp(a); }
};

struct OpCudaLog {
  __device__ float operator()(const float a) const { return log(a); }
};

__device__ int32_t cuGetReduceSrcIndex(const int32_t* retShape,
                                       const int32_t* srcStrides,
                                       const int32_t idx, const int32_t dim,
                                       const int32_t retDimCount) {
  int32_t outIndex = idx;
  int32_t inIndex = 0;
#pragma unroll
  for (int32_t d = retDimCount - 1; d >= 0; d--) {
    const int32_t coord = outIndex % retShape[d];
    outIndex /= retShape[d];
    inIndex += coord * srcStrides[d < dim ? d : d + 1];
  }
  return inIndex;
}

__device__ int32_t cuGetReduceDstIndex(const int32_t* shape,
                                       const int32_t* strides,
                                       const int32_t idx, const int32_t dim,
                                       const int32_t dimCount) {
  int32_t retIdx = 0;
  int32_t stride = 1;
#pragma unroll
  for (int32_t d = dimCount - 1; d >= 0; d--) {
    if (d != dim) {
      retIdx += (idx / strides[d] % shape[d]) * stride;
      stride *= shape[d];
    }
  }
  return retIdx;
}

__device__ int32_t cuGetReduceDstIndex(const int32_t* shape,
                                       const int32_t* strides,
                                       const uint8_t* inAxis, const int32_t idx,
                                       const int32_t dimCount) {
  int32_t retIdx = 0;
  int32_t stride = 1;
#pragma unroll
  for (int32_t d = dimCount - 1; d >= 0; d--) {
    if (0 == inAxis[d]) {
      retIdx += (idx / strides[d] % shape[d]) * stride;
      stride *= shape[d];
    }
  }
  return retIdx;
}

__device__ __forceinline__ void cuGetSubIndices(int32_t* subIndices,
                                                const int32_t* shape,
                                                const float* const* indices,
                                                const int32_t idx,
                                                const int32_t len) {
#pragma unroll
  for (int32_t i = 0; i < len; i++) {
    const auto ind = static_cast<int32_t>(indices[i][idx]);
    subIndices[i] = ind >= 0 ? ind : ind + shape[i];
  }
}

__global__ void kFillConstant(float* t, const float val, const int32_t n) {
  const auto index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (index + 3 < n) {
    FETCH_FLOAT4(t[index]) = make_float4(val, val, val, val);
  } else {
    if (index < n) t[index] = val;
    if (index + 1 < n) t[index + 1] = val;
    if (index + 2 < n) t[index + 2] = val;
  }
}

__global__ void kFillLinSpace(float* dst, const float start, const float step,
                              const int32_t n) {
  const auto index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  const auto base = start + static_cast<float>(index) * step;
  if (index + 3 < n) {
    FETCH_FLOAT4(dst[index]) =
        make_float4(base, base + step, base + 2 * step, base + 3 * step);
  } else {
    if (index < n) dst[index] = base;
    if (index + 1 < n) dst[index + 1] = base + step;
    if (index + 2 < n) dst[index + 2] = base + 2 * step;
  }
}

__global__ void kFillRandUniform(float* t, const float minVal,
                                 const float maxVal, const unsigned long seed,
                                 const unsigned long seq, const int32_t n) {
  const auto index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (index < n) {
    curandStatePhilox4_32_10_t state;
    curand_init(seed, seq, index, &state);
    const auto rand = curand_uniform4(&state);
    const auto range = maxVal - minVal;

    if (index + 3 < n) {
      FETCH_FLOAT4(t[index]) =
          make_float4(rand.x * range + minVal, rand.y * range + minVal,
                      rand.z * range + minVal, rand.w * range + minVal);
    } else {
      if (index < n) t[index] = rand.x * range + minVal;
      if (index + 1 < n) t[index + 1] = rand.y * range + minVal;
      if (index + 2 < n) t[index + 2] = rand.z * range + minVal;
    }
  }
}

__global__ void kFillRandNormal(float* t, const float mean, const float stddev,
                                const unsigned long seed,
                                const unsigned long seq, const int32_t n) {
  const auto index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (index < n) {
    curandStatePhilox4_32_10_t state;
    curand_init(seed, seq, index, &state);
    const auto rand = curand_normal4(&state);

    if (index + 3 < n) {
      FETCH_FLOAT4(t[index]) =
          make_float4(rand.x * stddev + mean, rand.y * stddev + mean,
                      rand.z * stddev + mean, rand.w * stddev + mean);
    } else {
      if (index < n) t[index] = rand.x * stddev + mean;
      if (index + 1 < n) t[index + 1] = rand.y * stddev + mean;
      if (index + 2 < n) t[index + 2] = rand.z * stddev + mean;
    }
  }
}

__global__ void kFillRandBernoulli(float* t, const float p,
                                   const unsigned long seed,
                                   const unsigned long seq, const int32_t n) {
  const auto index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (index < n) {
    curandStatePhilox4_32_10_t state;
    curand_init(seed, seq, index, &state);
    const auto rand = curand_uniform4(&state);

    if (index + 3 < n) {
      FETCH_FLOAT4(t[index]) =
          make_float4(rand.x < p ? 1.f : 0.f, rand.y < p ? 1.f : 0.f,
                      rand.z < p ? 1.f : 0.f, rand.w < p ? 1.f : 0.f);
    } else {
      if (index < n) t[index] = rand.x < p ? 1.f : 0.f;
      if (index + 1 < n) t[index + 1] = rand.y < p ? 1.f : 0.f;
      if (index + 2 < n) t[index + 2] = rand.z < p ? 1.f : 0.f;
    }
  }
}

template <typename OP>
__global__ void kSingleOp_(float* t, const int32_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP opFunc;
  if (index < n) {
    opFunc(t[index]);
  }
}

template <typename OP>
__global__ void kSingleOp(float* ret, const float* t, const int32_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP opFunc;
  if (index < n) {
    ret[index] = opFunc(t[index]);
  }
}

template <typename OP>
__global__ void kPairOp(float* c, const float* a, const float* b,
                        const int32_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP opFunc;
  if (index < n) {
    c[index] = opFunc(a[index], b[index]);
  }
}

template <typename OP>
__global__ void kPairScalarFirstOp(float* c, const float a, const float* b,
                                   const int32_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP opFunc;
  if (index < n) {
    c[index] = opFunc(a, b[index]);
  }
}

template <typename OP>
__global__ void kPairScalarFirstOp(float* c, const float* a, const float* b,
                                   const int32_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP opFunc;

  if (index < n) {
    c[index] = opFunc(a[0], b[index]);
  }
}

template <typename OP>
__global__ void kPairScalarSecondOp(float* c, const float* a, const float b,
                                    const int32_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP opFunc;
  if (index < n) {
    c[index] = opFunc(a[index], b);
  }
}

template <typename OP>
__global__ void kPairScalarSecondOp(float* c, const float* a, const float* b,
                                    const int32_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP opFunc;

  if (index < n) {
    c[index] = opFunc(a[index], b[0]);
  }
}

template <typename OP>
__global__ void kPairOp_(float* a, const float* b, const int32_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP opFunc;
  if (index < n) {
    a[index] = opFunc(a[index], b[index]);
  }
}

template <typename OP>
__global__ void kPairScalarSecondOp_(float* a, const float b, const int32_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP opFunc;
  if (index < n) {
    a[index] = opFunc(a[index], b);
  }
}

template <typename OP>
__global__ void kPairScalarSecondOp_(float* a, const float* b,
                                     const int32_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP opFunc;

  if (index < n) {
    a[index] = opFunc(a[index], b[0]);
  }
}

__global__ void kClamp(float* ret, const float* t, const float minVal,
                       const float maxVal, const int32_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    ret[index] = max(minVal, min(t[index], maxVal));
  }
}

__global__ void kClamp_(float* t, const float minVal, const float maxVal,
                        const int32_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    t[index] = max(minVal, min(t[index], maxVal));
  }
}

template <typename OP, bool Leading, bool First>
__global__ void kBroadcastOpFast(float* ret, const float* a, const float* b,
                                 const int32_t stride, const int32_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP opFunc;
  if (index < n) {
    if (First) {
      ret[index] =
          opFunc(a[Leading ? index % stride : index / stride], b[index]);
    } else {
      ret[index] =
          opFunc(a[index], b[Leading ? index % stride : index / stride]);
    }
  }
}

__device__ int32_t cuIndicesToOffset(const int32_t* strides,
                                     const int32_t* indices,
                                     const int32_t dimCount) {
  int32_t offset = 0;
#pragma unroll
  for (int32_t i = 0; i < dimCount; i++) {
    offset += indices[i] * strides[i];
  }
  return offset;
}

__device__ void cuOffsetToIndices(int32_t* indices, const int32_t* shape,
                                  const int32_t index, const int32_t dimCount) {
  int32_t offset = index;
#pragma unroll
  for (int32_t i = dimCount - 1; i >= 0; i--) {
    indices[i] = offset % shape[i];
    offset /= shape[i];
  }
}

// TODO optimize
template <typename OP>
__global__ void kBroadcastOpCommon(TensorCudaCtx c, const TensorCudaCtx a,
                                   const TensorCudaCtx b, const int32_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP opFunc;

  if (index < n) {
    int32_t cIndices[TENSOR_MAX_DIMS];
    int32_t aIndices[TENSOR_MAX_DIMS] = {};
    int32_t bIndices[TENSOR_MAX_DIMS] = {};

    cuOffsetToIndices(cIndices, c.shape_, static_cast<int32_t>(index),
                      c.dimCount_);

#pragma unroll
    for (auto j = 0; j < c.dimCount_; j++) {
      if (j >= c.dimCount_ - a.dimCount_) {
        const int32_t aIndex = j - (c.dimCount_ - a.dimCount_);
        aIndices[aIndex] = (a.shape_[aIndex] != 1) ? cIndices[j] : 0;
      }
      if (j >= c.dimCount_ - b.dimCount_) {
        const int32_t bIndex = j - (c.dimCount_ - b.dimCount_);
        bIndices[bIndex] = (b.shape_[bIndex] != 1) ? cIndices[j] : 0;
      }
    }
    const auto aIdx = cuIndicesToOffset(a.strides_, aIndices, a.dimCount_);
    const auto bIdx = cuIndicesToOffset(b.strides_, bIndices, b.dimCount_);
    c.data_[index] = opFunc(a.data_[aIdx], b.data_[bIdx]);
  }
}

struct OpCudaReduceMax {
  __device__ float operator()(const float a, const float b) const {
    return max(a, b);
  }

  static __device__ float defaultVal() { return -FLT_MAX; }
};

struct OpCudaReduceMin {
  __device__ float operator()(const float a, const float b) const {
    return min(a, b);
  }

  static __device__ float defaultVal() { return FLT_MAX; }
};

struct OpCudaReduceSum {
  __device__ float operator()(const float a, const float b) const {
    return a + b;
  }

  static __device__ float defaultVal() { return 0.f; }
};

struct ReduceIdx {
  int32_t inIdx;
  int32_t outIdx;
};

struct ReduceIndexAll {
  __device__ ReduceIdx operator()(const int32_t index, const int32_t segLength,
                                  const int32_t segCount) const {
    const int32_t idx = (index < segLength) ? index : -1;
    return {idx, idx};
  }
};

struct ReduceIndexFirstDim {
  __device__ ReduceIdx operator()(const int32_t index, const int32_t segLength,
                                  const int32_t segCount) const {
    const auto segDim = static_cast<int32_t>(gridDim.x * blockDim.x / segCount);
    const int32_t segIdx = index / segDim;
    const int32_t segTid = index % segDim;
    const int32_t idx = (segTid < segLength) ? segIdx + segTid * segCount : -1;
    return {idx, segTid};
  }
};

struct ReduceIndexLastDim {
  __device__ ReduceIdx operator()(const int32_t index, const int32_t segLength,
                                  const int32_t segCount) const {
    const auto segDim = static_cast<int32_t>(gridDim.x * blockDim.x / segCount);
    const int32_t segIdx = index / segDim;
    const int32_t segTid = index % segDim;
    const int32_t idx = (segTid < segLength) ? segTid + segIdx * segLength : -1;
    return {idx, segTid};
  }
};

template <typename OP>
__device__ __forceinline__ void cuWarpReduce(float& val) {
  const OP op;
  val = op(val, __shfl_down_sync(0xFFFFFFFF, val, 16));
  val = op(val, __shfl_down_sync(0xFFFFFFFF, val, 8));
  val = op(val, __shfl_down_sync(0xFFFFFFFF, val, 4));
  val = op(val, __shfl_down_sync(0xFFFFFFFF, val, 2));
  val = op(val, __shfl_down_sync(0xFFFFFFFF, val, 1));
}

template <typename OP>
__device__ __forceinline__ void cuWarpReduceIdx(float& val, int& idx) {
  const OP op;

  float otherVal = __shfl_down_sync(0xFFFFFFFF, val, 16);
  int32_t otherIdx = __shfl_down_sync(0xFFFFFFFF, idx, 16);
  if (op(otherVal, val) == otherVal) {
    val = otherVal;
    idx = otherIdx;
  }

  otherVal = __shfl_down_sync(0xFFFFFFFF, val, 8);
  otherIdx = __shfl_down_sync(0xFFFFFFFF, idx, 8);
  if (op(otherVal, val) == otherVal) {
    val = otherVal;
    idx = otherIdx;
  }

  otherVal = __shfl_down_sync(0xFFFFFFFF, val, 4);
  otherIdx = __shfl_down_sync(0xFFFFFFFF, idx, 4);
  if (op(otherVal, val) == otherVal) {
    val = otherVal;
    idx = otherIdx;
  }

  otherVal = __shfl_down_sync(0xFFFFFFFF, val, 2);
  otherIdx = __shfl_down_sync(0xFFFFFFFF, idx, 2);
  if (op(otherVal, val) == otherVal) {
    val = otherVal;
    idx = otherIdx;
  }

  otherVal = __shfl_down_sync(0xFFFFFFFF, val, 1);
  otherIdx = __shfl_down_sync(0xFFFFFFFF, idx, 1);
  if (op(otherVal, val) == otherVal) {
    val = otherVal;
    idx = otherIdx;
  }
}

template <typename OP, typename IndexFunc>
__global__ void kReduceMerge(float* outValues, const float* inValues,
                             const int32_t segLength, const int32_t segCount) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float shared[WARP_SIZE];
  float val = OP::defaultVal();

  const IndexFunc indexer;
  const auto [inIdx, outIdx] = indexer(index, segLength, segCount);
  if (inIdx >= 0) {
    val = inValues[inIdx];
  }
  cuWarpReduce<OP>(val);

  if (threadIdx.x % warpSize == 0) {
    shared[threadIdx.x / warpSize] = val;
  }
  __syncthreads();

  if (threadIdx.x < warpSize) {
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[threadIdx.x]
                                                : OP::defaultVal();
    cuWarpReduce<OP>(val);
  }

  if (threadIdx.x == 0) {
    outValues[blockIdx.x] = val;
  }
}

template <typename OP, typename IndexFunc>
__global__ void kReduceIdxMerge(float* outValues, float* outIndices,
                                const float* inValues, const float* inIndices,
                                const int32_t segLength,
                                const int32_t segCount) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float sharedVal[WARP_SIZE];
  __shared__ int32_t sharedIdx[WARP_SIZE];

  float val = OP::defaultVal();
  int32_t idx = -1;

  const IndexFunc indexer;
  const auto [inIdx, outIdx] = indexer(index, segLength, segCount);
  if (inIdx >= 0) {
    val = inValues[inIdx];
    idx = inIndices ? static_cast<int>(inIndices[inIdx]) : outIdx;
  }
  cuWarpReduceIdx<OP>(val, idx);

  if (threadIdx.x % warpSize == 0) {
    sharedVal[threadIdx.x / warpSize] = val;
    sharedIdx[threadIdx.x / warpSize] = idx;
  }
  __syncthreads();

  if (threadIdx.x < warpSize) {
    val = (threadIdx.x < blockDim.x / warpSize) ? sharedVal[threadIdx.x]
                                                : OP::defaultVal();
    idx = (threadIdx.x < blockDim.x / warpSize) ? sharedIdx[threadIdx.x] : -1;

    cuWarpReduceIdx<OP>(val, idx);
  }

  if (threadIdx.x == 0) {
    if (outIndices) outIndices[blockIdx.x] = static_cast<float>(idx);
    if (outValues) outValues[blockIdx.x] = val;
  }
}

template <typename OP, bool isFirstDim>
__global__ void kReduceDimFirstOrLast(float* outValues, const float* inValues,
                                      const int32_t segLength,
                                      const int32_t segCount) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP op;
  if (index < segCount) {
    auto targetVal = OP::defaultVal();
    auto srcIdx = isFirstDim ? index : index * segLength;
#pragma unroll
    for (int32_t j = 0; j < segLength; j++) {
      const auto val = inValues[srcIdx];
      srcIdx += isFirstDim ? segCount : 1;
      targetVal = op(val, targetVal);
    }
    outValues[index] = targetVal;
  }
}

template <typename OP>
__global__ void kReduceDim(TensorCudaCtx values, const TensorCudaCtx t,
                           const int32_t dim, const int32_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP op;

  const auto dimSize = t.shape_[dim];
  const auto stride = t.strides_[dim];

  if (index < n) {
    auto targetVal = OP::defaultVal();
    int32_t srcIdx =
        cuGetReduceSrcIndex(values.shape_, t.strides_,
                            static_cast<int32_t>(index), dim, values.dimCount_);
#pragma unroll
    for (int32_t j = 0; j < dimSize; j++) {
      const auto val = t.data_[srcIdx];
      srcIdx += stride;
      targetVal = op(val, targetVal);
    }
    values.data_[index] = targetVal;
  }
}

template <typename OP, bool isFirstDim>
__global__ void kReduceIdxDimFirstOrLast(float* outValues, float* outIndices,
                                         const float* inValues,
                                         const int32_t segLength,
                                         const int32_t segCount) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP op;
  if (index < segCount) {
    auto targetVal = OP::defaultVal();
    auto targetIdx = 0;
    auto srcIdx = isFirstDim ? index : index * segLength;
#pragma unroll
    for (int32_t j = 0; j < segLength; j++) {
      const auto val = inValues[srcIdx];
      srcIdx += isFirstDim ? segCount : 1;
      if (op(val, targetVal) == val) {
        targetVal = val;
        targetIdx = j;
      }
    }
    outValues[index] = targetVal;
    outIndices[index] = static_cast<float>(targetIdx);
  }
}

template <typename OP>
__global__ void kReduceIdxDim(TensorCudaCtx values, float* indices,
                              const TensorCudaCtx t, const int32_t dim,
                              const int32_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP op;

  const auto dimSize = t.shape_[dim];
  const auto stride = t.strides_[dim];

  if (index < n) {
    auto targetVal = OP::defaultVal();
    int32_t targetIdx = 0;
    int32_t srcIdx =
        cuGetReduceSrcIndex(values.shape_, t.strides_,
                            static_cast<int32_t>(index), dim, values.dimCount_);
#pragma unroll
    for (int32_t j = 0; j < dimSize; j++) {
      const auto val = t.data_[srcIdx];
      srcIdx += stride;
      if (op(val, targetVal) == val) {
        targetVal = val;
        targetIdx = j;
      }
    }
    values.data_[index] = targetVal;
    indices[index] = static_cast<float>(targetIdx);
  }
}

__global__ void kSquaredDiff(float* output, const float* input,
                             const float* mean, const int32_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    const float diff = input[index] - *mean;
    output[index] = diff * diff;
  }
}

// TODO optimize
__global__ void kReduceMultiDimSum(float* retPtr, const TensorCudaCtx t,
                                   const FixedVector<uint8_t> inAxis,
                                   const int32_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    const int32_t retIdx =
        cuGetReduceDstIndex(t.shape_, t.strides_, inAxis.data,
                            static_cast<int32_t>(index), t.dimCount_);
    atomicAdd(&retPtr[retIdx], t.data_[index]);
  }
}

// TODO optimize
__global__ void kReduceMultiDimVar(float* retPtr, const TensorCudaCtx t,
                                   const float* meanValues,
                                   const FixedVector<uint8_t> inAxis,
                                   const int32_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    const int32_t retIdx =
        cuGetReduceDstIndex(t.shape_, t.strides_, inAxis.data,
                            static_cast<int32_t>(index), t.dimCount_);
    const float diff = t.data_[index] - meanValues[retIdx];
    atomicAdd(&retPtr[retIdx], diff * diff);
  }
}

__global__ void kPermute(const TensorCudaCtx ret, const TensorCudaCtx t,
                         const FixedVector<int32_t> dims, const int32_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    int32_t srcIndex = 0;
    auto offset = static_cast<int32_t>(index);
#pragma unroll
    for (int32_t d = 0; d < t.dimCount_; d++) {
      srcIndex += (offset / ret.strides_[d]) * t.strides_[dims.data[d]];
      offset %= ret.strides_[d];
    }
    ret.data_[index] = t.data_[srcIndex];
  }
}

__global__ void kTranspose(float* out, const float* in, const int32_t width,
                           const int32_t height) {
  __shared__ float tile[TRANSPOSE_TILE_DIM]
                       [TRANSPOSE_TILE_DIM + 1];  // +1 to avoid bank conflicts

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

__global__ void kIndex(float* ret, const TensorCudaCtx t,
                       const FixedVector<float*> indices,
                       const int32_t dimStride, const int32_t len,
                       const int32_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    int32_t subIndices[TENSOR_MAX_DIMS];
    cuGetSubIndices(subIndices, t.shape_, indices.data,
                    static_cast<int32_t>(index), len);
    const int32_t dataIdx =
        cuIndicesToOffset(t.strides_, subIndices, t.dimCount_);
    memcpy(&ret[dimStride * index], &t.data_[dataIdx],
           dimStride * sizeof(float));
  }
}

__global__ void kIndex2D(float* ret, const float* t, const float* indices0,
                         const float* indices1, const int32_t dim0,
                         const int32_t dim1, const int32_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    auto idx0 = static_cast<int32_t>(indices0[index]);
    auto idx1 = static_cast<int32_t>(indices1[index]);
    if (idx0 < 0) idx0 += dim0;
    if (idx1 < 0) idx1 += dim1;
    const auto dataIdx = idx0 * dim1 + idx1;
    ret[index] = t[dataIdx];
  }
}

__global__ void kIndexPut(TensorCudaCtx t, FixedVector<float*> indices,
                          const int32_t dimStride, const int32_t len,
                          const float val, const int32_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    int32_t subIndices[TENSOR_MAX_DIMS];
    cuGetSubIndices(subIndices, t.shape_, indices.data,
                    static_cast<int32_t>(index), len);
    const int32_t dataIdx =
        cuIndicesToOffset(t.strides_, subIndices, t.dimCount_);
#pragma unroll
    for (int32_t i = 0; i < dimStride; i++) {
      t.data_[dataIdx + i] = val;
    }
  }
}

__global__ void kIndexPut2D(float* t, const float* indices0,
                            const float* indices1, const int32_t dim0,
                            const int32_t dim1, const float val,
                            const int32_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    auto idx0 = static_cast<int32_t>(indices0[index]);
    auto idx1 = static_cast<int32_t>(indices1[index]);
    if (idx0 < 0) idx0 += dim0;
    if (idx1 < 0) idx1 += dim1;
    const auto dataIdx = idx0 * dim1 + idx1;
    t[dataIdx] = val;
  }
}

__global__ void kIndexPut(TensorCudaCtx t, FixedVector<float*> indices,
                          const int32_t dimStride, const int32_t len,
                          const float* val, const int32_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    int32_t subIndices[TENSOR_MAX_DIMS];
    cuGetSubIndices(subIndices, t.shape_, indices.data,
                    static_cast<int32_t>(index), len);
    const int32_t dataIdx =
        cuIndicesToOffset(t.strides_, subIndices, t.dimCount_);
    memcpy(&t.data_[dataIdx], &val[dimStride * index],
           dimStride * sizeof(float));
  }
}

__global__ void kIndexPut2D(float* t, const float* indices0,
                            const float* indices1, const int32_t dim0,
                            const int32_t dim1, const float* val,
                            const int32_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    auto idx0 = static_cast<int32_t>(indices0[index]);
    auto idx1 = static_cast<int32_t>(indices1[index]);
    if (idx0 < 0) idx0 += dim0;
    if (idx1 < 0) idx1 += dim1;
    const auto dataIdx = idx0 * dim1 + idx1;
    t[dataIdx] = val[index];
  }
}

__global__ void kIm2Col(float* ret, const float* t, const int32_t n,
                        const int32_t channels, const int32_t height,
                        const int32_t width, const int32_t outH,
                        const int32_t outW, const int32_t kernelH,
                        const int32_t kernelW, const int32_t strideH,
                        const int32_t strideW, const int32_t paddingH,
                        const int32_t paddingW) {
  const auto index =
      static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < n) {
    const int32_t colH = outH * outW;
    const int32_t kernelSize = kernelH * kernelW;

    const int32_t kernelIdx = index % kernelSize;
    const int32_t colIdx = index / kernelSize;

    const int32_t b = colIdx / (channels * colH);
    const int32_t c = (colIdx % (channels * colH)) / colH;
    const int32_t h = (colIdx % colH) / outW;
    const int32_t w = colIdx % outW;

    const int32_t kh = kernelIdx / kernelW;
    const int32_t kw = kernelIdx % kernelW;

    const float* imPtr = t + (b * channels + c) * height * width;
    float* colPtr =
        ret + ((b * colH + h * outW + w) * channels + c) * kernelSize;

    const int32_t ih = h * strideH - paddingH + kh;
    const int32_t iw = w * strideW - paddingW + kw;
    float val = 0.f;
    if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
      val = imPtr[ih * width + iw];
    }
    colPtr[kh * kernelW + kw] = val;
  }
}

__global__ void kCol2Im(float* ret, const float* t, const int32_t n,
                        const int32_t channels, const int32_t height,
                        const int32_t width, const int32_t outH,
                        const int32_t outW, const int32_t kernelH,
                        const int32_t kernelW, const int32_t strideH,
                        const int32_t strideW, const int32_t paddingH,
                        const int32_t paddingW) {
  const auto index =
      static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < n) {
    const int32_t colH = outH * outW;
    const int32_t kernelSize = kernelH * kernelW;

    const int32_t b = index / (channels * colH);
    const int32_t c = (index % (channels * colH)) / colH;
    const int32_t h = (index % colH) / outW;
    const int32_t w = index % outW;

    const int32_t hStride = h * strideH - paddingH;
    const int32_t wStride = w * strideW - paddingW;

    float* imPtr = ret + (b * channels + c) * height * width;
    const float* colPtr =
        t + ((b * colH + h * outW + w) * channels + c) * kernelSize;

    for (int32_t i = 0; i < kernelH; i++) {
      for (int32_t j = 0; j < kernelW; j++) {
        const int32_t ih = hStride + i;
        const int32_t iw = wStride + j;
        if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
          atomicAdd(&imPtr[ih * width + iw], colPtr[i * kernelW + j]);
        }
      }
    }
  }
}

__global__ void kDot(float* ret, const float* a, const float* b,
                     const int32_t n) {
  extern __shared__ float sharedData[];

  auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const auto tid = threadIdx.x;

  float temp = 0.f;
  while (index < n) {
    temp += a[index] * b[index];
    index += blockDim.x * gridDim.x;
  }

  sharedData[tid] = temp;
  __syncthreads();

  for (auto stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sharedData[tid] += sharedData[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(ret, sharedData[0]);
  }
}

}  // namespace TinyTorch
