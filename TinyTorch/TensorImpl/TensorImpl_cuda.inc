/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

namespace TinyTorch {
const char* curandGetErrorString(curandStatus_t status);
const char* cublasGetErrorString(cublasStatus_t status);

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
      std::cerr << "CUDA error in file '" << __FILE__ << "' in line "         \
                << __LINE__ << ": " << cudaGetErrorString(err) << " (" << err \
                << ")" << std::endl;                                          \
      abort();                                                                \
    }                                                                         \
  } while (0)

#define CURAND_CHECK(call)                                               \
  do {                                                                   \
    curandStatus_t err = call;                                           \
    if (err != CURAND_STATUS_SUCCESS) {                                  \
      std::cerr << "CURAND error in file '" << __FILE__ << "' in line "  \
                << __LINE__ << ": " << curandGetErrorString(err) << " (" \
                << err << ")" << std::endl;                              \
      abort();                                                           \
    }                                                                    \
  } while (0)

#define CUBLAS_CHECK(call)                                               \
  do {                                                                   \
    cublasStatus_t err = call;                                           \
    if (err != CUBLAS_STATUS_SUCCESS) {                                  \
      std::cerr << "CUBLAS error in file '" << __FILE__ << "' in line "  \
                << __LINE__ << ": " << cublasGetErrorString(err) << " (" \
                << err << ")" << std::endl;                              \
      abort();                                                           \
    }                                                                    \
  } while (0)

#define CUDA_KERNEL_CHECK()                                                   \
  do {                                                                        \
    cudaError_t err = cudaGetLastError();                                     \
    if (err != cudaSuccess) {                                                 \
      std::cerr << "CUDA kernel error in file '" << __FILE__ << "' in line "  \
                << __LINE__ << ": " << cudaGetErrorString(err) << " (" << err \
                << ")" << std::endl;                                          \
      abort();                                                                \
    }                                                                         \
  } while (0)

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

template <typename T>
__device__ void cuSwap(T& a, T& b) {
  T c(a);
  a = b;
  b = c;
}

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

__device__ int32_t cuIndicesToOffset(const int32_t* strides,
                                     const int32_t* indices, int32_t dimCount) {
  int32_t offset = 0;
  for (int32_t i = 0; i < dimCount; i++) {
    offset += indices[i] * strides[i];
  }
  return offset;
}

__device__ void cuOffsetToIndices(int32_t* indices, const int32_t* shape,
                                  int32_t offset, int32_t dimCount) {
  for (int32_t i = dimCount - 1; i >= 0; i--) {
    indices[i] = offset % shape[i];
    offset /= shape[i];
  }
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

__global__ void kFillConstant(float* t, float val, size_t n) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    t[index] = val;
  }
}

__global__ void kFillLinSpace(float* dst, float start, float step,
                              size_t count) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    dst[index] = start + (float)index * step;
  }
}

__global__ void kFillRandUniform(float* t, float minVal, float maxVal,
                                 unsigned long seed, unsigned long seq, int n) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto index = tid * 4;
  if (index < n) {
    curandStatePhilox4_32_10_t state;
    curand_init(seed, seq, index, &state);
    float4 rand = curand_uniform4(&state);

    float range = maxVal - minVal;
    if (index < n) t[index] = rand.x * range + minVal;
    if (index + 1 < n) t[index + 1] = rand.y * range + minVal;
    if (index + 2 < n) t[index + 2] = rand.z * range + minVal;
    if (index + 3 < n) t[index + 3] = rand.w * range + minVal;
  }
}

__global__ void kFillRandNormal(float* t, float mean, float stddev,
                                unsigned long seed, unsigned long seq, int n) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto index = tid * 4;
  if (index < n) {
    curandStatePhilox4_32_10_t state;
    curand_init(seed, seq, index, &state);
    float4 rand = curand_normal4(&state);

    if (index < n) t[index] = rand.x * stddev + mean;
    if (index + 1 < n) t[index + 1] = rand.y * stddev + mean;
    if (index + 2 < n) t[index + 2] = rand.z * stddev + mean;
    if (index + 3 < n) t[index + 3] = rand.w * stddev + mean;
  }
}

__global__ void kFillRandBernoulli(float* t, float p, unsigned long seed,
                                   unsigned long seq, int n) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto index = tid * 4;
  if (index < n) {
    curandStatePhilox4_32_10_t state;
    curand_init(seed, seq, index, &state);
    float4 rand = curand_uniform4(&state);

    if (index < n) t[index] = rand.x < p ? 1.f : 0.f;
    if (index + 1 < n) t[index + 1] = rand.y < p ? 1.f : 0.f;
    if (index + 2 < n) t[index + 2] = rand.z < p ? 1.f : 0.f;
    if (index + 3 < n) t[index + 3] = rand.w < p ? 1.f : 0.f;
  }
}

template <typename OP>
__global__ void kSingleOp_(float* t, int n) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  OP opFunc;
  if (index < n) {
    opFunc(t[index]);
  }
}

template <typename OP>
__global__ void kSingleOp(float* ret, const float* t, int n) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  OP opFunc;
  if (index < n) {
    ret[index] = opFunc(t[index]);
  }
}

template <typename OP>
__global__ void kPairOp(const float* a, const float* b, float* c, int n) {
  OP opFunc;
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    c[index] = opFunc(a[index], b[index]);
  }
}

template <typename OP>
__global__ void kPairScalarFirstOp(const float a, const float* b, float* c,
                                   int n) {
  OP opFunc;
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    c[index] = opFunc(a, b[index]);
  }
}

template <typename OP>
__global__ void kPairScalarFirstOp(const float* a, const float* b, float* c,
                                   int n) {
  OP opFunc;
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    c[index] = opFunc(a[0], b[index]);
  }
}

template <typename OP>
__global__ void kPairScalarSecondOp(const float* a, const float b, float* c,
                                    int n) {
  OP opFunc;
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    c[index] = opFunc(a[index], b);
  }
}

template <typename OP>
__global__ void kPairScalarSecondOp(const float* a, const float* b, float* c,
                                    int n) {
  OP opFunc;
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    c[index] = opFunc(a[index], b[0]);
  }
}

template <typename OP>
__global__ void kPairOp_(float* a, const float* b, int n) {
  OP opFunc;
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    a[index] = opFunc(a[index], b[index]);
  }
}

template <typename OP>
__global__ void kPairScalarSecondOp_(float* a, const float b, int n) {
  OP opFunc;
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    a[index] = opFunc(a[index], b);
  }
}

template <typename OP>
__global__ void kPairScalarSecondOp_(float* a, const float* b, int n) {
  OP opFunc;
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    a[index] = opFunc(a[index], b[0]);
  }
}

template <typename OP>
__global__ void kBroadcastFastPassOp(TensorCudaCtx result,
                                     const TensorCudaCtx larger,
                                     const TensorCudaCtx smaller, bool reverse,
                                     int n) {
  OP opFunc;
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    auto& dataA = larger.data_[index];
    auto& dataB = smaller.data_[index % smaller.elemCount_];
    auto& dataRet = result.data_[index];
    dataRet = reverse ? opFunc(dataB, dataA) : opFunc(dataA, dataB);
  }
}

template <typename OP>
__global__ void kBroadcastOp(TensorCudaCtx c, const TensorCudaCtx a,
                             const TensorCudaCtx b, int n) {
  OP opFunc;
  int32_t cIndices[TENSOR_MAX_DIMS];
  int32_t aIndices[TENSOR_MAX_DIMS];
  int32_t bIndices[TENSOR_MAX_DIMS];

  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    memset(aIndices, 0, TENSOR_MAX_DIMS * sizeof(int32_t));
    memset(bIndices, 0, TENSOR_MAX_DIMS * sizeof(int32_t));

    int offset = index;
    for (int32_t i = c.dimCount_ - 1; i >= 0; i--) {
      cIndices[i] = offset % c.shape_[i];
      offset /= c.shape_[i];
    }

    for (int32_t j = 0; j < c.dimCount_; j++) {
      if (j >= c.dimCount_ - a.dimCount_) {
        int32_t aIndex = j - (c.dimCount_ - a.dimCount_);
        if (a.shape_[aIndex] != 1) {
          aIndices[aIndex] = cIndices[j];
        }
      }
      if (j >= c.dimCount_ - b.dimCount_) {
        int32_t bIndex = j - (c.dimCount_ - b.dimCount_);
        if (b.shape_[bIndex] != 1) {
          bIndices[bIndex] = cIndices[j];
        }
      }
    }

    int32_t aIdx = 0;
    for (int32_t i = 0; i < a.dimCount_; i++) {
      aIdx += aIndices[i] * a.strides_[i];
    }

    int32_t bIdx = 0;
    for (int32_t i = 0; i < b.dimCount_; i++) {
      bIdx += bIndices[i] * b.strides_[i];
    }
    c.data_[index] = opFunc(a.data_[aIdx], b.data_[bIdx]);
  }
}

__global__ void kClamp(float* ret, const float* t, float minVal, float maxVal,
                       int n) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    ret[index] = max(minVal, min(t[index], maxVal));
  }
}

__global__ void kClamp_(float* t, float minVal, float maxVal, int n) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    t[index] = max(minVal, min(t[index], maxVal));
  }
}

// TODO opt
__global__ void kReduceAllMin(float* retPtr, const float* data, int n) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    cuAtomicMinFloat(retPtr, data[index]);
  }
}

// TODO opt
__global__ void kReduceAllMax(float* retPtr, const float* data, int n) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    cuAtomicMaxFloat(retPtr, data[index]);
  }
}

// TODO opt
__global__ void kReduceAllSum(float* retPtr, const float* data, int n) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    atomicAdd(retPtr, data[index]);
  }
}

// TODO opt
__global__ void kReduceAllVar(float* retPtr, const float* data,
                              const float* meanVal, int n) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    float diff = data[index] - *meanVal;
    atomicAdd(retPtr, diff * diff);
  }
}

// TODO opt
template <bool FindMax>
__global__ void kReduceAllArg(float* retPtr, const float* input, int n) {
  extern __shared__ float sharedData[];
  auto* sharedIndex = (int32_t*)&sharedData[blockDim.x];

  auto tid = threadIdx.x;
  auto globalIndex = (int32_t)(blockIdx.x * blockDim.x + tid);

  if (globalIndex < n) {
    sharedData[tid] = input[globalIndex];
    sharedIndex[tid] = globalIndex;
  } else {
    sharedData[tid] = FindMax ? -FLT_MAX : FLT_MAX;
    sharedIndex[tid] = -1;
  }
  __syncthreads();

  for (auto s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (FindMax) {
        if (sharedData[tid + s] > sharedData[tid]) {
          sharedData[tid] = sharedData[tid + s];
          sharedIndex[tid] = sharedIndex[tid + s];
        }
      } else {
        if (sharedData[tid + s] < sharedData[tid]) {
          sharedData[tid] = sharedData[tid + s];
          sharedIndex[tid] = sharedIndex[tid + s];
        }
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicExch(retPtr, (float)sharedIndex[0]);
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
#pragma unroll
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

__global__ void kReduceSumLastDim(float* retPtr, const float* t,
                                  int32_t dimSize, int n) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    auto srcIdx = index * dimSize;
    float val = 0.f;
    for (int32_t j = 0; j < dimSize; j++) {
      val += t[srcIdx++];
    }
    retPtr[index] = val;
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
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    int32_t originIndex = 0;
    auto offset = (int32_t)index;
    for (int32_t d = 0; d < t.dimCount_; d++) {
      originIndex += (offset / ret.strides_[d]) * t.strides_[dims.data[d]];
      offset %= ret.strides_[d];
    }
    ret.data_[index] = t.data_[originIndex];
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

const char* curandGetErrorString(curandStatus_t status) {
  switch (status) {
    case CURAND_STATUS_SUCCESS:
      return "CURAND_STATUS_SUCCESS";
    case CURAND_STATUS_VERSION_MISMATCH:
      return "CURAND_STATUS_VERSION_MISMATCH";
    case CURAND_STATUS_NOT_INITIALIZED:
      return "CURAND_STATUS_NOT_INITIALIZED";
    case CURAND_STATUS_ALLOCATION_FAILED:
      return "CURAND_STATUS_ALLOCATION_FAILED";
    case CURAND_STATUS_TYPE_ERROR:
      return "CURAND_STATUS_TYPE_ERROR";
    case CURAND_STATUS_OUT_OF_RANGE:
      return "CURAND_STATUS_OUT_OF_RANGE";
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
      return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
      return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
    case CURAND_STATUS_LAUNCH_FAILURE:
      return "CURAND_STATUS_LAUNCH_FAILURE";
    case CURAND_STATUS_PREEXISTING_FAILURE:
      return "CURAND_STATUS_PREEXISTING_FAILURE";
    case CURAND_STATUS_INITIALIZATION_FAILED:
      return "CURAND_STATUS_INITIALIZATION_FAILED";
    case CURAND_STATUS_ARCH_MISMATCH:
      return "CURAND_STATUS_ARCH_MISMATCH";
    case CURAND_STATUS_INTERNAL_ERROR:
      return "CURAND_STATUS_INTERNAL_ERROR";
  }
  return "Unknown cuRAND error";
}

const char* cublasGetErrorString(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "Unknown cuBLAS error";
}

}  // namespace TinyTorch
