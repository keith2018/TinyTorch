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

struct OpCudaLess {
  __device__ bool operator()(const float& a, const float& b) { return a < b; }
};

struct OpCudaGreater {
  __device__ bool operator()(const float& a, const float& b) { return a > b; }
};

__device__ int32_t cuGetReduceSrcIndex(const TensorCudaCtx ret,
                                       const TensorCudaCtx t, int32_t idx,
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

__device__ int32_t cuGetReduceDstIndex(const TensorCudaCtx t, int32_t idx,
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

__device__ int32_t cuGetReduceDstIndex(const TensorCudaCtx t, int32_t idx,
                                       const FixedVector<uint8_t> inAxis) {
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

__device__ float cuAtomicMinFloat(float* address, float val) {
  int* address_as_i = (int*)address;
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_i, assumed,
                    __float_as_int(fminf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

__device__ float cuAtomicMaxFloat(float* address, float val) {
  int* address_as_i = (int*)address;
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_i, assumed,
                    __float_as_int(fmaxf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

__device__ void cuReorderIndices(int32_t* indices, const int32_t* order,
                                 int32_t dimCount) {
  int32_t temp[TENSOR_MAX_DIMS];
  for (int i = 0; i < dimCount; i++) {
    temp[i] = indices[order[i]];
  }
  memcpy(indices, temp, sizeof(int32_t) * dimCount);
}

__device__ void cuGetSubIndices(int32_t* subIndices, const TensorCudaCtx t,
                                const FixedVector<float*> indices, int32_t idx,
                                int len) {
  for (int32_t i = 0; i < len; i++) {
    auto ind = (int32_t)((float*)indices.data[i])[idx];
    subIndices[i] = ind >= 0 ? ind : ind + t.shape_[i];
  }
}

__global__ void kFillConstant(float* __restrict__ t, const float val,
                              const size_t n) {
  const auto index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (index + 3 < n) {
    FETCH_FLOAT4(t[index]) = make_float4(val, val, val, val);
  } else {
    if (index < n) t[index] = val;
    if (index + 1 < n) t[index + 1] = val;
    if (index + 2 < n) t[index + 2] = val;
  }
}

__global__ void kFillLinSpace(float* __restrict__ dst, const float start,
                              const float step, const size_t n) {
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

__global__ void kFillRandUniform(float* __restrict__ t, const float minVal,
                                 const float maxVal, const unsigned long seed,
                                 const unsigned long seq, const int n) {
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

__global__ void kFillRandNormal(float* __restrict__ t, const float mean,
                                const float stddev, const unsigned long seed,
                                const unsigned long seq, const int n) {
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

__global__ void kFillRandBernoulli(float* __restrict__ t, const float p,
                                   const unsigned long seed,
                                   const unsigned long seq, const int n) {
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
__global__ void kSingleOp_(float* __restrict__ t, const int n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP opFunc;
  if (index < n) {
    opFunc(t[index]);
  }
}

template <typename OP>
__global__ void kSingleOp(float* __restrict__ ret, const float* __restrict__ t,
                          const int n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP opFunc;
  if (index < n) {
    ret[index] = opFunc(t[index]);
  }
}

template <typename OP>
__global__ void kPairOp(float* __restrict__ c, const float* __restrict__ a,
                        const float* __restrict__ b, const int n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP opFunc;
  if (index < n) {
    c[index] = opFunc(a[index], b[index]);
  }
}

template <typename OP>
__global__ void kPairScalarFirstOp(float* __restrict__ c, const float a,
                                   const float* __restrict__ b, const int n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP opFunc;
  if (index < n) {
    c[index] = opFunc(a, b[index]);
  }
}

template <typename OP>
__global__ void kPairScalarFirstOp(float* __restrict__ c,
                                   const float* __restrict__ a,
                                   const float* __restrict__ b, const int n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP opFunc;

  if (index < n) {
    c[index] = opFunc(a[0], b[index]);
  }
}

template <typename OP>
__global__ void kPairScalarSecondOp(float* __restrict__ c,
                                    const float* __restrict__ a, const float b,
                                    const int n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP opFunc;
  if (index < n) {
    c[index] = opFunc(a[index], b);
  }
}

template <typename OP>
__global__ void kPairScalarSecondOp(float* __restrict__ c,
                                    const float* __restrict__ a,
                                    const float* __restrict__ b, const int n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP opFunc;

  if (index < n) {
    c[index] = opFunc(a[index], b[0]);
  }
}

template <typename OP>
__global__ void kPairOp_(float* __restrict__ a, const float* __restrict__ b,
                         const int n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP opFunc;
  if (index < n) {
    a[index] = opFunc(a[index], b[index]);
  }
}

template <typename OP>
__global__ void kPairScalarSecondOp_(float* __restrict__ a, const float b,
                                     const int n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP opFunc;
  if (index < n) {
    a[index] = opFunc(a[index], b);
  }
}

template <typename OP>
__global__ void kPairScalarSecondOp_(float* __restrict__ a,
                                     const float* __restrict__ b, const int n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP opFunc;

  if (index < n) {
    a[index] = opFunc(a[index], b[0]);
  }
}

__global__ void kClamp(float* __restrict__ ret, const float* __restrict__ t,
                       const float minVal, const float maxVal, const int n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    ret[index] = max(minVal, min(t[index], maxVal));
  }
}

__global__ void kClamp_(float* __restrict__ t, const float minVal,
                        const float maxVal, const int n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    t[index] = max(minVal, min(t[index], maxVal));
  }
}

template <typename OP, bool Leading, bool First>
__global__ void kBroadcastOpFast(float* __restrict__ ret,
                                 const float* __restrict__ a,
                                 const float* __restrict__ b,
                                 const int32_t stride, const int n) {
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

__device__ int32_t cuIndicesToOffset(const int32_t* __restrict__ strides,
                                     const int32_t* __restrict__ indices,
                                     const int32_t dimCount) {
  int32_t offset = 0;
  for (int32_t i = 0; i < dimCount; i++) {
    offset += indices[i] * strides[i];
  }
  return offset;
}

__device__ void cuOffsetToIndices(int32_t* __restrict__ indices,
                                  const int32_t* __restrict__ shape,
                                  const int32_t index, const int32_t dimCount) {
  int32_t offset = index;
  for (int32_t i = dimCount - 1; i >= 0; i--) {
    indices[i] = offset % shape[i];
    offset /= shape[i];
  }
}

template <typename OP>
__global__ void kBroadcastOpCommon(TensorCudaCtx c, const TensorCudaCtx a,
                                   const TensorCudaCtx b, const int n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP opFunc;

  if (index < n) {
    int32_t cIndices[TENSOR_MAX_DIMS];
    int32_t aIndices[TENSOR_MAX_DIMS] = {};
    int32_t bIndices[TENSOR_MAX_DIMS] = {};

    cuOffsetToIndices(cIndices, c.shape_, static_cast<int32_t>(index),
                      c.dimCount_);
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

  static __device__ __forceinline__ float defaultVal() { return -FLT_MAX; }
};

struct OpCudaReduceMin {
  __device__ float operator()(const float a, const float b) const {
    return min(a, b);
  }

  static __device__ __forceinline__ float defaultVal() { return FLT_MAX; }
};

struct OpCudaReduceSum {
  __device__ __forceinline__ float operator()(const float a,
                                              const float b) const {
    return a + b;
  }

  static __device__ __forceinline__ float defaultVal() { return 0.f; }
};

template <typename OP>
__device__ __forceinline__ float cuWarpReduce(float val) {
  const OP op;
  val = op(val, __shfl_down_sync(0xFFFFFFFF, val, 16));
  val = op(val, __shfl_down_sync(0xFFFFFFFF, val, 8));
  val = op(val, __shfl_down_sync(0xFFFFFFFF, val, 4));
  val = op(val, __shfl_down_sync(0xFFFFFFFF, val, 2));
  val = op(val, __shfl_down_sync(0xFFFFFFFF, val, 1));
  return val;
}

template <typename OP>
__device__ __forceinline__ int cuWarpReduceIdx(float val, int idx) {
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

  return idx;
}

template <typename OP>
__global__ void kReduceAll(float* __restrict__ output,
                           const float* __restrict__ input, const int n,
                           const int m) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float shared[WARP_SIZE];
  float val = OP::defaultVal();

  if (index < n) {
    val = input[index];
  }
  val = cuWarpReduce<OP>(val);

  if (threadIdx.x % warpSize == 0) {
    shared[threadIdx.x / warpSize] = val;
  }
  __syncthreads();

  if (threadIdx.x < warpSize) {
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[threadIdx.x]
                                                : OP::defaultVal();
    val = cuWarpReduce<OP>(val);
  }

  if (threadIdx.x == 0) {
    output[blockIdx.x] = val;
  }
}

template <typename OP>
__global__ void kReduceAllFirstDim(float* __restrict__ output,
                                   const float* __restrict__ input, const int n,
                                   const int m) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float shared[WARP_SIZE];
  float val = OP::defaultVal();

  const auto segDim = gridDim.x * blockDim.x / m;
  const auto segIdx = index / segDim;
  const auto segTid = index % segDim;
  if (segTid < n) {
    val = input[segIdx + segTid * m];
  }
  val = cuWarpReduce<OP>(val);

  if (threadIdx.x % warpSize == 0) {
    shared[threadIdx.x / warpSize] = val;
  }
  __syncthreads();

  if (threadIdx.x < warpSize) {
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[threadIdx.x]
                                                : OP::defaultVal();
    val = cuWarpReduce<OP>(val);
  }

  if (threadIdx.x == 0) {
    output[blockIdx.x] = val;
  }
}

template <typename OP>
__global__ void kReduceAllLastDim(float* __restrict__ output,
                                  const float* __restrict__ input, const int n,
                                  const int m) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float shared[WARP_SIZE];
  float val = OP::defaultVal();

  const auto segDim = gridDim.x * blockDim.x / m;
  const auto segIdx = index / segDim;
  const auto segTid = index % segDim;
  if (segTid < n) {
    val = input[segTid + segIdx * n];
  }
  val = cuWarpReduce<OP>(val);

  if (threadIdx.x % warpSize == 0) {
    shared[threadIdx.x / warpSize] = val;
  }
  __syncthreads();

  if (threadIdx.x < warpSize) {
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[threadIdx.x]
                                                : OP::defaultVal();
    val = cuWarpReduce<OP>(val);
  }

  if (threadIdx.x == 0) {
    output[blockIdx.x] = val;
  }
}

template <typename OP>
__global__ void kReduceAllIdx(float* output, const float* __restrict__ input,
                              const int n, const int m) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float sharedVal[WARP_SIZE];
  __shared__ int sharedIdx[WARP_SIZE];

  float val = OP::defaultVal();
  int idx = -1;

  if (index < n) {
    val = input[index];
    idx = static_cast<int>(index);
  }
  idx = cuWarpReduceIdx<OP>(val, idx);

  if (threadIdx.x % warpSize == 0) {
    sharedVal[threadIdx.x / warpSize] = val;
    sharedIdx[threadIdx.x / warpSize] = idx;
  }
  __syncthreads();

  if (threadIdx.x < warpSize) {
    val = (threadIdx.x < blockDim.x / warpSize) ? sharedVal[threadIdx.x]
                                                : OP::defaultVal();
    idx = (threadIdx.x < blockDim.x / warpSize) ? sharedIdx[threadIdx.x] : -1;

    idx = cuWarpReduceIdx<OP>(val, idx);
  }

  if (threadIdx.x == 0) {
    output[blockIdx.x] = static_cast<float>(idx);
  }
}

__global__ void kSquaredDiff(float* __restrict__ output,
                             const float* __restrict__ input,
                             const float* __restrict__ mean, const int n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    const float diff = input[index] - *mean;
    output[index] = diff * diff;
  }
}

template <typename Compare>
__global__ void kReduceLastDim(float* values, float* indices, const float* t,
                               float initVal, int32_t dimSize, Compare comp,
                               int n) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    auto targetVal = initVal;
    int32_t targetIdx = 0;
    int32_t srcIdx = index * dimSize;
    for (int32_t j = 0; j < dimSize; j++) {
      auto val = t[srcIdx++];
      if (comp(val, targetVal)) {
        targetVal = val;
        targetIdx = j;
      }
    }
    values[index] = targetVal;
    indices[index] = static_cast<float>(targetIdx);
  }
}

template <typename Compare>
__global__ void kReduceDim(TensorCudaCtx values, float* indices,
                           const TensorCudaCtx t, int32_t dim, bool keepDims,
                           float initVal, Compare comp, int n) {
  const auto dimSize = t.shape_[dim];
  const auto stride = t.strides_[dim];

  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    auto targetVal = initVal;
    int32_t targetIdx = 0;
    int32_t srcIdx = cuGetReduceSrcIndex(values, t, index, dim, keepDims);
    for (int32_t j = 0; j < dimSize; j++) {
      auto val = t.data_[srcIdx];
      srcIdx += stride;
      if (comp(val, targetVal)) {
        targetVal = val;
        targetIdx = j;
      }
    }
    values.data_[index] = targetVal;
    indices[index] = static_cast<float>(targetIdx);
  }
}

__global__ void kReduceSumFirstDim(float* retPtr, const float* t,
                                   int32_t dimSize, int32_t stride) {
  extern __shared__ float sharedData[];

  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int tid = threadIdx.x;

  for (auto i = tid; i < stride; i += blockDim.x) {
    sharedData[i] = 0.f;
  }
  __syncthreads();

  if (index < dimSize) {
    auto srcIdx = index * stride;
    for (auto i = 0; i < stride; i++) {
      atomicAdd(&sharedData[i], t[srcIdx++]);
    }
  }
  __syncthreads();

  for (auto i = tid; i < stride; i += blockDim.x) {
    retPtr[i] = sharedData[i];
  }
}

__global__ void kReduceSum(float* retPtr, const TensorCudaCtx t,
                           const FixedVector<uint8_t> inAxis, int n) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    int32_t retIdx = cuGetReduceDstIndex(t, index, inAxis);
    atomicAdd(&retPtr[retIdx], t.data_[index]);
  }
}

__global__ void kReduceVar(float* retPtr, const TensorCudaCtx t,
                           const float* meanValues,
                           const FixedVector<uint8_t> inAxis, int n) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    int32_t retIdx = cuGetReduceDstIndex(t, index, inAxis);
    float diff = t.data_[index] - meanValues[retIdx];
    atomicAdd(&retPtr[retIdx], diff * diff);
  }
}

__global__ void kPermute(const TensorCudaCtx ret, const TensorCudaCtx t,
                         const FixedVector<int32_t> dims, int n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    unsigned int srcIndex = 0;
    auto offset = index;
    for (int32_t d = 0; d < t.dimCount_; d++) {
      srcIndex += (offset / ret.strides_[d]) * t.strides_[dims.data[d]];
      offset %= ret.strides_[d];
    }
    ret.data_[index] = t.data_[srcIndex];
  }
}

__global__ void kTranspose(float* __restrict__ out,
                           const float* __restrict__ in, const int width,
                           const int height) {
  __shared__ float tile[TRANSPOSE_TILE_DIM]
                       [TRANSPOSE_TILE_DIM + 1];  // +1 to avoid bank conflicts

  int x = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.y;

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

__global__ void kIndex(float* retData, const TensorCudaCtx t,
                       const FixedVector<float*> indices, int dimStride,
                       int len, int n) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    int32_t subIndices[TENSOR_MAX_DIMS];
    cuGetSubIndices(subIndices, t, indices, index, len);
    int32_t dataIdx = cuIndicesToOffset(t.strides_, subIndices, t.dimCount_);
    memcpy(&retData[dimStride * index], &t.data_[dataIdx],
           dimStride * sizeof(float));
  }
}

__global__ void kIndexPut(TensorCudaCtx t, FixedVector<float*> indices,
                          int dimStride, int len, float val, int n) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    int32_t subIndices[TENSOR_MAX_DIMS];
    cuGetSubIndices(subIndices, t, indices, index, len);
    int32_t dataIdx = cuIndicesToOffset(t.strides_, subIndices, t.dimCount_);
    for (int32_t i = 0; i < dimStride; i++) {
      t.data_[dataIdx + i] = val;
    }
  }
}

__global__ void kIndexPut(TensorCudaCtx t, FixedVector<float*> indices,
                          int dimStride, int len, float* val, int n) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    int32_t subIndices[TENSOR_MAX_DIMS];
    cuGetSubIndices(subIndices, t, indices, index, len);
    int32_t dataIdx = cuIndicesToOffset(t.strides_, subIndices, t.dimCount_);
    memcpy(&t.data_[dataIdx], &val[dimStride * index],
           dimStride * sizeof(float));
  }
}

__global__ void kIm2Col(float* ret, const float* t, int32_t batch,
                        int32_t channels, int32_t height, int32_t width,
                        int32_t outH, int32_t outW, int32_t kernelH,
                        int32_t kernelW, int32_t strideH, int32_t strideW,
                        int32_t paddingH, int32_t paddingW, int32_t imStride,
                        int32_t colH, int32_t colW) {
  auto index = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  int32_t totalElements = batch * outH * outW * channels * kernelH * kernelW;

  if (index < totalElements) {
    int32_t kw = index % kernelW;
    int32_t kh = (index / kernelW) % kernelH;
    int32_t c = (index / (kernelW * kernelH)) % channels;
    int32_t w = (index / (kernelW * kernelH * channels)) % outW;
    int32_t h = (index / (kernelW * kernelH * channels * outW)) % outH;
    int32_t n = (index / (kernelW * kernelH * channels * outW * outH)) % batch;

    int32_t colIdx = (n * outH + h) * outW + w;
    int32_t imRow = h * strideH + kh - paddingH;
    int32_t imCol = w * strideW + kw - paddingW;
    int32_t colWIdx = c * kernelH * kernelW + kh * kernelW + kw;

    float value = 0.0f;
    if (imRow >= 0 && imRow < height && imCol >= 0 && imCol < width) {
      int32_t imgIdx = imCol + width * (imRow + height * c);
      value = t[n * imStride + imgIdx];
    }
    ret[colIdx * colW + colWIdx] = value;
  }
}

__global__ void kCol2Im(float* ret, const float* t, int32_t batch,
                        int32_t channels, int32_t height, int32_t width,
                        int32_t outH, int32_t outW, int32_t kernelH,
                        int32_t kernelW, int32_t strideH, int32_t strideW,
                        int32_t paddingH, int32_t paddingW, int32_t imStride,
                        int32_t colW) {
  auto index = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  int32_t totalElements = batch * channels * height * width;

  if (index < totalElements) {
    int32_t w = index % width;
    int32_t h = (index / width) % height;
    int32_t c = (index / (width * height)) % channels;
    int32_t n = index / (width * height * channels);

    float value = 0.0f;
    for (int32_t kh = 0; kh < kernelH; ++kh) {
      for (int32_t kw = 0; kw < kernelW; ++kw) {
        int32_t imRow = h + paddingH - kh;
        int32_t imCol = w + paddingW - kw;

        if (imRow % strideH == 0 && imCol % strideW == 0) {
          int32_t outRow = imRow / strideH;
          int32_t outCol = imCol / strideW;

          if (outRow >= 0 && outRow < outH && outCol >= 0 && outCol < outW) {
            int32_t colIdx = (n * outH + outRow) * outW + outCol;
            int32_t colWIdx = c * kernelH * kernelW + kh * kernelW + kw;
            value += t[colIdx * colW + colWIdx];
          }
        }
      }
    }

    ret[index] = value;
  }
}

__global__ void kDot(float* ret, const float* a, const float* b, int n) {
  extern __shared__ float sharedData[];

  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto threadId = threadIdx.x;

  float temp = 0.f;
  while (tid < n) {
    temp += a[tid] * b[tid];
    tid += blockDim.x * gridDim.x;
  }

  sharedData[threadId] = temp;
  __syncthreads();

  for (int32_t stride = (int32_t)blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadId < stride) {
      sharedData[threadId] += sharedData[threadId + stride];
    }
    __syncthreads();
  }

  if (threadId == 0) {
    atomicAdd(ret, sharedData[0]);
  }
}

}  // namespace TinyTorch
