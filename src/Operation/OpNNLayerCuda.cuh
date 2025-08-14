/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "OpNNLayer.h"
#include "OpReduceCuda.cuh"
#include "Tensor/TensorIterator.cuh"
#include "Utils/CUDAUtils.h"
#include "Utils/MathUtils.h"
#include "Utils/RandomGenerator.h"

namespace tinytorch::op {

template <typename T, typename OP>
__device__ T cudaBlockReduce(T val, T init) {
  cudaWarpReduce<T, OP>(val);
  __shared__ T shared[32];
  auto lane = threadIdx.x & 0x1f;
  auto wid = threadIdx.x >> 5;
  if (lane == 0) shared[wid] = val;
  __syncthreads();
  val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : init;
  if (wid == 0) {
    cudaWarpReduce<T, OP>(val);
    if (threadIdx.x == 0) shared[0] = val;
  }
  __syncthreads();
  return shared[0];
}

template <typename T, SoftmaxType type>
__global__ void kSoftmaxForward(T* out, const T* self, int64_t dimSize, int64_t innerSize) {
  auto outer = blockIdx.x;
  auto inner = blockIdx.y;
  auto tid = threadIdx.x;
  auto base = (outer * dimSize * innerSize) + inner;

  // max
  T maxVal = -cuda::Inf<T>();
  for (auto i = tid; i < dimSize; i += blockDim.x) {
    maxVal = ::max(maxVal, self[base + i * innerSize]);
  }
  T blockMax = cudaBlockReduce<T, OpCudaReduceMax>(maxVal, -cuda::Inf<T>());

  // sum of exp
  T sum = 0;
  for (auto i = tid; i < dimSize; i += blockDim.x) {
    sum += ::expf(self[base + i * innerSize] - blockMax);
  }
  T blockSum = cudaBlockReduce<T, OpCudaReduceSum>(sum, T(0));

  // output
  if constexpr (type == SoftmaxType::Softmax) {
    for (auto i = tid; i < dimSize; i += blockDim.x) {
      out[base + i * innerSize] = ::expf(self[base + i * innerSize] - blockMax) / blockSum;
    }
  } else {
    T logSum = ::logf(blockSum);
    for (auto i = tid; i < dimSize; i += blockDim.x) {
      out[base + i * innerSize] = self[base + i * innerSize] - blockMax - logSum;
    }
  }
}

template <typename T, SoftmaxType type>
__global__ void kSoftmaxForwardLargeDim(T* out, const T* self, int64_t dimSize, int64_t innerSize) {
  // shared[0:blockPerRow]: max
  // shared[blockPerRow:2*blockPerRow]: sum
  extern __shared__ T shared[];

  auto blockPerRow = gridDim.x;
  auto blockId = blockIdx.x;
  auto outer = blockIdx.y;
  auto inner = blockIdx.z;
  auto tid = threadIdx.x;

  auto tileSize = (dimSize + blockPerRow - 1) / blockPerRow;
  auto tileStart = blockId * tileSize;
  auto tileEnd = ::min(tileStart + tileSize, dimSize);

  auto base = (outer * dimSize * innerSize) + inner;

  // max
  T localMax = -cuda::Inf<T>();
  for (auto i = tileStart + tid; i < tileEnd; i += blockDim.x) {
    auto idx = base + i * innerSize;
    localMax = ::max(localMax, self[idx]);
  }
  T tileMax = cudaBlockReduce<T, OpCudaReduceMax>(localMax, -cuda::Inf<T>());
  if (tid == 0) shared[blockId] = tileMax;
  __syncthreads();

  T globalMax = -cuda::Inf<T>();
  if (blockId == 0) {
    for (auto i = tid; i < blockPerRow; i += blockDim.x) {
      globalMax = ::max(globalMax, shared[i]);
    }
    globalMax = cudaBlockReduce<T, OpCudaReduceMax>(globalMax, -cuda::Inf<T>());
    if (tid == 0) shared[blockPerRow] = globalMax;
  }
  __syncthreads();
  globalMax = shared[blockPerRow];

  // sum of exp
  T localSum = 0;
  for (auto i = tileStart + tid; i < tileEnd; i += blockDim.x) {
    auto idx = base + i * innerSize;
    localSum += ::expf(self[idx] - globalMax);
  }
  T tileSum = cudaBlockReduce<T, OpCudaReduceSum>(localSum, T(0));
  if (tid == 0) shared[blockId] = tileSum;
  __syncthreads();

  T globalSum = 0;
  if (blockId == 0) {
    for (auto i = tid; i < blockPerRow; i += blockDim.x) {
      globalSum += shared[i];
    }
    globalSum = cudaBlockReduce<T, OpCudaReduceSum>(globalSum, T(0));
    if (tid == 0) shared[blockPerRow] = globalSum;
  }
  __syncthreads();
  globalSum = shared[blockPerRow];

  // output
  if constexpr (type == SoftmaxType::Softmax) {
    for (auto i = tileStart + tid; i < tileEnd; i += blockDim.x) {
      auto idx = base + i * innerSize;
      out[idx] = ::expf(self[idx] - globalMax) / globalSum;
    }
  } else {
    T logSum = ::logf(globalSum);
    for (auto i = tileStart + tid; i < tileEnd; i += blockDim.x) {
      auto idx = base + i * innerSize;
      out[idx] = self[idx] - globalMax - logSum;
    }
  }
}

template <typename T>
struct OpCudaSoftmaxBackward {
  __device__ static T computePartialSum(const T* output, const T* grad, int64_t idx) { return output[idx] * grad[idx]; }

  __device__ static T computeResult(const T* output, const T* grad, T totalSum, int64_t idx) {
    return output[idx] * (grad[idx] - totalSum);
  }
};

template <typename T>
struct OpCudaLogSoftmaxBackward {
  __device__ static T computePartialSum(const T* output, const T* grad, int64_t idx) { return grad[idx]; }

  __device__ static T computeResult(const T* output, const T* grad, T totalSum, int64_t idx) {
    return grad[idx] - ::expf(output[idx]) * totalSum;
  }
};

template <typename T, typename Op>
__global__ void kSoftmaxBackward(T* out, const T* output, const T* grad, int64_t dimSize, int64_t innerSize) {
  auto outer = blockIdx.x;
  auto inner = blockIdx.y;
  auto tid = threadIdx.x;
  auto base = (outer * dimSize * innerSize) + inner;

  T sum = 0;
  for (auto i = tid; i < dimSize; i += blockDim.x) {
    auto idx = base + i * innerSize;
    sum += Op::computePartialSum(output, grad, idx);
  }
  T blockSum = cudaBlockReduce<T, OpCudaReduceSum>(sum, T(0));

  for (auto i = tid; i < dimSize; i += blockDim.x) {
    auto idx = base + i * innerSize;
    out[idx] = Op::computeResult(output, grad, blockSum, idx);
  }
}

template <typename T, typename Op, bool isPhase1>
__global__ void kSoftmaxBackwardLarge(T* out, const T* output, const T* grad, T* partialSums, int64_t dimSize,
                                      int64_t innerSize, int64_t numBlocks) {
  auto outer = blockIdx.z;
  auto inner = blockIdx.y;
  auto blockId = blockIdx.x;
  auto tid = threadIdx.x;
  auto base = (outer * dimSize * innerSize) + inner;

  auto elemsPerBlock = (dimSize + numBlocks - 1) / numBlocks;
  auto start = blockId * elemsPerBlock;
  auto end = ::min(start + elemsPerBlock, dimSize);

  if constexpr (isPhase1) {
    T sum = 0;
    for (auto i = start + tid; i < end; i += blockDim.x) {
      auto idx = base + i * innerSize;
      sum += Op::computePartialSum(output, grad, idx);
    }
    T blockSum = cudaBlockReduce<T, OpCudaReduceSum>(sum, T(0));

    if (tid == 0) {
      auto partialIdx = (outer * innerSize + inner) * numBlocks + blockId;
      partialSums[partialIdx] = blockSum;
    }
  } else {
    T totalSum = 0;
    auto partialBase = (outer * innerSize + inner) * numBlocks;
    for (auto i = tid; i < numBlocks; i += blockDim.x) {
      totalSum += partialSums[partialBase + i];
    }
    T blockSum = cudaBlockReduce<T, OpCudaReduceSum>(totalSum, T(0));

    for (auto i = start + tid; i < end; i += blockDim.x) {
      auto idx = base + i * innerSize;
      out[idx] = Op::computeResult(output, grad, blockSum, idx);
    }
  }
}

template <typename T>
__global__ void kDropout(T* out, const T* self, const float p, const unsigned long seed, const unsigned long seq,
                         const int64_t n) {
  const auto index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (index < n) {
    curandStatePhilox4_32_10_t state;
    curand_init(seed, seq, index, &state);
    const auto rand = curand_uniform4(&state);

    if (index + 3 < n) {
      out[index] = rand.x < p ? (self[index] / p) : 0.f;
      out[index + 1] = rand.y < p ? (self[index + 1] / p) : 0.f;
      out[index + 2] = rand.z < p ? (self[index + 2] / p) : 0.f;
      out[index + 3] = rand.w < p ? (self[index + 3] / p) : 0.f;
    } else {
      if (index < n) out[index] = rand.x < p ? (self[index] / p) : 0.f;
      if (index + 1 < n) out[index + 1] = rand.y < p ? (self[index + 1] / p) : 0.f;
      if (index + 2 < n) out[index + 2] = rand.z < p ? (self[index + 2] / p) : 0.f;
    }
  }
}

template <typename T, NormType normType>
__global__ void kNormSmall(T* out, const T* input, const T* weight, const T* bias, int64_t dim, float eps) {
  auto row = blockIdx.x;
  auto tid = threadIdx.x;
  auto base = row * dim;

  T sum = 0;
  for (auto i = tid; i < dim; i += blockDim.x) {
    if constexpr (normType == NormType::LayerNorm) {
      sum += input[base + i];
    } else {  // RMSNorm
      T val = input[base + i];
      sum += val * val;
    }
  }

  T stat = cudaBlockReduce<T, OpCudaReduceSum>(sum, T(0)) / static_cast<T>(dim);

  T invStd;
  if constexpr (normType == NormType::LayerNorm) {
    T mean = stat;
    // var
    T varSum = 0;
    for (auto i = tid; i < dim; i += blockDim.x) {
      T diff = input[base + i] - mean;
      varSum += diff * diff;
    }
    T var = cudaBlockReduce<T, OpCudaReduceSum>(varSum, T(0)) / static_cast<T>(dim);
    invStd = ::rsqrtf(var + eps);

    // norm + affine
    for (auto i = tid; i < dim; i += blockDim.x) {
      T normed = (input[base + i] - mean) * invStd;
      if (weight) normed *= weight[i];
      if (bias) normed += bias[i];
      out[base + i] = normed;
    }
  } else {
    invStd = ::rsqrtf(stat + eps);

    // norm + affine
    for (auto i = tid; i < dim; i += blockDim.x) {
      T normed = input[base + i] * invStd;
      if (weight) normed *= weight[i];
      out[base + i] = normed;
    }
  }
}

template <typename T, NormType normType>
__global__ void kNormLarge(T* out, const T* input, const T* weight, const T* bias, int64_t dim, float eps) {
  extern __shared__ T sharedMem[];
  T* sharedStats = sharedMem;

  auto row = blockIdx.x;
  auto tid = threadIdx.x;
  auto base = row * dim;

  T sum = 0;
  for (auto i = tid; i < dim; i += blockDim.x) {
    if constexpr (normType == NormType::LayerNorm) {
      sum += input[base + i];
    } else {  // RMSNorm
      T val = input[base + i];
      sum += val * val;
    }
  }

  T stat = cudaBlockReduce<T, OpCudaReduceSum>(sum, T(0));
  if (tid == 0) sharedStats[0] = stat / static_cast<T>(dim);
  __syncthreads();
  stat = sharedStats[0];

  T invStd;
  if constexpr (normType == NormType::LayerNorm) {
    T mean = stat;
    // var
    T varSum = 0;
    for (auto i = tid; i < dim; i += blockDim.x) {
      T diff = input[base + i] - mean;
      varSum += diff * diff;
    }
    T var = cudaBlockReduce<T, OpCudaReduceSum>(varSum, T(0));
    if (tid == 0) sharedStats[1] = var / static_cast<T>(dim);
    __syncthreads();
    var = sharedStats[1];
    invStd = ::rsqrtf(var + eps);

    // norm + affine
    for (auto i = tid; i < dim; i += blockDim.x) {
      T normed = (input[base + i] - mean) * invStd;
      if (weight) normed *= weight[i];
      if (bias) normed += bias[i];
      out[base + i] = normed;
    }
  } else {
    invStd = ::rsqrtf(stat + eps);

    // norm + affine
    for (auto i = tid; i < dim; i += blockDim.x) {
      T normed = input[base + i] * invStd;
      if (weight) normed *= weight[i];
      out[base + i] = normed;
    }
  }
}

template <typename T>
__global__ void kRopeComputeInvFreq(T* invFreqPtr, int64_t halfDim, float thetaBase) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < halfDim) {
    invFreqPtr[idx] = 1.f / powf(thetaBase, static_cast<float>(idx << 1) / static_cast<float>(halfDim << 1));
  }
}

template <typename T>
__global__ void kRopeApplyScaling(T* invFreqPtr, int64_t halfDim, float originalContextLength, float lowFreqFactor,
                                  float highFreqFactor, float scalingFactor) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < halfDim) {
    auto waveLen = 2.f * static_cast<float>(M_PI) / invFreqPtr[idx];
    auto lowWaveLen = originalContextLength / lowFreqFactor;
    auto highWaveLen = originalContextLength / highFreqFactor;

    if (waveLen > lowWaveLen) {
      invFreqPtr[idx] /= scalingFactor;
    } else if (waveLen < highWaveLen) {
      // do nothing
    } else {
      auto smoothFactor = (originalContextLength / waveLen - lowFreqFactor) / (highFreqFactor - lowFreqFactor);
      auto scaled = invFreqPtr[idx] / scalingFactor;
      invFreqPtr[idx] = (1.f - smoothFactor) * scaled + smoothFactor * invFreqPtr[idx];
    }
  }
}

template <typename T>
__global__ void kRopePrecomputeCosSin(const T* invFreqPtr, T* cosPtr, T* sinPtr, int64_t contextLength,
                                      int64_t headDim) {
  const int64_t pos = blockIdx.x;
  if (pos < contextLength) {
    auto halfDim = headDim >> 1;
    for (auto i = threadIdx.x; i < halfDim; i += blockDim.x) {
      float angle = static_cast<float>(pos) * invFreqPtr[i];
      int64_t offset1 = pos * headDim + i;
      int64_t offset2 = pos * headDim + halfDim + i;

      float cosVal = cosf(angle);
      float sinVal = sinf(angle);

      cosPtr[offset1] = cosVal;
      sinPtr[offset1] = sinVal;
      cosPtr[offset2] = cosVal;
      sinPtr[offset2] = sinVal;
    }
  }
}

template <typename T>
__global__ void kRopeApply(T* out, const T* input, const T* cos, const T* sin, int numHead, int seqLen, int headDim) {
  auto t = blockIdx.x;   // sequence pos
  auto h = blockIdx.y;   // head id
  auto b = blockIdx.z;   // batch id
  auto i = threadIdx.x;  // dim idx

  auto halfDim = headDim >> 1;
  if (i >= halfDim) {
    return;
  }
  auto base = ((b * numHead + h) * seqLen + t) * headDim;
  const T* xPtr = input + base;
  T* yPtr = out + base;
  const T* cosRow = cos + t * headDim;
  const T* sinRow = sin + t * headDim;
  T x1 = xPtr[i];
  T x2 = xPtr[halfDim + i];
  T c = cosRow[i];
  T s = sinRow[i];
  yPtr[i] = x1 * c - x2 * s;
  yPtr[halfDim + i] = x2 * c + x1 * s;
}

template <typename T, SoftmaxType type>
void softmaxForwardCudaImpl(Tensor& out, const Tensor& self, int64_t dim) {
  ASSERT(out.shape() == self.shape());
  auto info = getSoftmaxDimInfo(self, dim);

  const T* selfPtr = self.dataPtr<T>();
  T* outPtr = out.dataPtr<T>();

  if (info.dimSize <= 1024) {
    dim3 block(std::clamp(nextPow2(info.dimSize), 32u, 1024u));
    dim3 grid(info.outerSize, info.innerSize);
    const auto stream = cuda::getCurrentCUDAStream(self.device().index).stream;
    kSoftmaxForward<T, type><<<grid, block, 0, stream>>>(outPtr, selfPtr, info.dimSize, info.innerSize);
  } else {
    auto blockSize = cuda::getKernelBlockSize(self.device().index);
    dim3 grid(1, info.outerSize, info.innerSize);
    size_t sharedMem = 2 * sizeof(T);  // max, sum
    const auto stream = cuda::getCurrentCUDAStream(self.device().index).stream;
    kSoftmaxForwardLargeDim<T, type>
        <<<grid, blockSize, sharedMem, stream>>>(outPtr, selfPtr, info.dimSize, info.innerSize);
  }
  CUDA_KERNEL_CHECK();
}

template <typename T>
void softmaxOpOutCudaImpl(Tensor& out, const Tensor& self, int64_t dim) {
  softmaxForwardCudaImpl<T, SoftmaxType::Softmax>(out, self, dim);
}

template <typename T>
Tensor softmaxOpCudaImpl(const Tensor& self, int64_t dim) {
  Tensor out(self.shape(), self.options().noGrad());
  softmaxOpOutCudaImpl<T>(out, self, dim);
  return out;
}

template <typename T, typename Op>
Tensor softmaxBackwardCudaImplDetail(const Tensor& grad, const Tensor& output, int64_t dim) {
  ASSERT(output.shape() == grad.shape());
  auto info = getSoftmaxDimInfo(output, dim);
  Tensor out(output.shape(), output.options().noGrad());

  const T* outputPtr = output.dataPtr<T>();
  const T* gradPtr = grad.dataPtr<T>();
  T* outPtr = out.dataPtr<T>();
  const auto stream = cuda::getCurrentCUDAStream(output.device().index).stream;

  if (info.dimSize <= 1024) {
    dim3 block(std::clamp(nextPow2(info.dimSize), 32u, 1024u));
    dim3 grid(info.outerSize, info.innerSize);
    kSoftmaxBackward<T, Op><<<grid, block, 0, stream>>>(outPtr, outputPtr, gradPtr, info.dimSize, info.innerSize);
  } else {
    constexpr int64_t maxBlocksPerDim = 16;
    const int64_t numBlocks = std::min(maxBlocksPerDim, (info.dimSize + 1023) / 1024);
    const int64_t partialSumsSize = info.outerSize * info.innerSize * numBlocks;

    Storage partialSums(static_cast<int64_t>(partialSumsSize * sizeof(T)), output.device());
    T* partialSumsPtr = partialSums.dataPtr<T>();

    dim3 block(cuda::getKernelBlockSize(output.device().index));
    dim3 grid(numBlocks, info.innerSize, info.outerSize);

    // phase 1
    kSoftmaxBackwardLarge<T, Op, true><<<grid, block, 0, stream>>>(outPtr, outputPtr, gradPtr, partialSumsPtr,
                                                                   info.dimSize, info.innerSize, numBlocks);

    // phase 2
    kSoftmaxBackwardLarge<T, Op, false><<<grid, block, 0, stream>>>(outPtr, outputPtr, gradPtr, partialSumsPtr,
                                                                    info.dimSize, info.innerSize, numBlocks);
  }
  CUDA_KERNEL_CHECK();
  return out;
}

template <typename T>
Tensor softmaxOpBackwardCudaImpl(const Tensor& grad, const Tensor& output, int64_t dim) {
  return softmaxBackwardCudaImplDetail<T, OpCudaSoftmaxBackward<T>>(grad, output, dim);
}

template <typename T>
void logSoftmaxOpOutCudaImpl(Tensor& out, const Tensor& self, int64_t dim) {
  softmaxForwardCudaImpl<T, SoftmaxType::LogSoftmax>(out, self, dim);
}

template <typename T>
Tensor logSoftmaxOpCudaImpl(const Tensor& self, int64_t dim) {
  Tensor out(self.shape(), self.options().noGrad());
  logSoftmaxOpOutCudaImpl<T>(out, self, dim);
  return out;
}

template <typename T>
Tensor logSoftmaxOpBackwardCudaImpl(const Tensor& grad, const Tensor& output, int64_t dim) {
  return softmaxBackwardCudaImplDetail<T, OpCudaLogSoftmaxBackward<T>>(grad, output, dim);
}

template <typename T>
Tensor dropoutOpCudaImpl(const Tensor& self, float p) {
  Tensor out(self.shape(), self.options().noGrad());
  const auto* selfPtr = self.dataPtr<T>();
  auto* outPtr = out.dataPtr<T>();

  auto seed = RandomGeneratorCUDA::getSeed();
  auto seq = RandomGeneratorCUDA::nextSequence();
  int64_t n = self.numel();

  auto params = cuda::getKernelLaunchParams(self.device().index, n, 4);
  CUDA_LAUNCH_KERNEL(kDropout<T>, params, outPtr, selfPtr, p, seed, seq, n);
  return out;
}

template <typename T>
Tensor dropoutMaskedOpCudaImpl(const Tensor& self, const Tensor& mask, float p) {
  TensorIteratorCuda iterator(self, mask);
  auto outShape = iterator.setupBroadcast();
  ASSERT(iterator.isBroadcastOk());
  Tensor out(outShape, self.options().noGrad());
  iterator.template forEach<T>(out, [p] __device__(const T& a, const T& b) -> T { return a * b / p; });
  return out;
}

template <typename T, NormType normType>
Tensor normOpCudaImplDetail(const Tensor& self, IntArrayView normalizedShape, const Tensor& weight, const Tensor& bias,
                            float eps) {
  int64_t dim = self.shape().back();
  int64_t numRows = self.numel() / dim;
  ASSERT(normalizedShape.size() == 1);
  ASSERT(normalizedShape.front() == dim);

  Tensor out(self.shape(), self.options().noGrad());

  const auto* inputPtr = self.dataPtr<T>();
  const auto* weightPtr = weight.defined() ? weight.dataPtr<T>() : nullptr;
  const auto* biasPtr = bias.defined() ? bias.dataPtr<T>() : nullptr;
  auto* outPtr = out.dataPtr<T>();

  auto stream = cuda::getCurrentCUDAStream(self.device().index).stream;
  dim3 blockSize(std::clamp(nextPow2(dim), 32u, 1024u));
  dim3 gridSize(numRows);

  if (dim <= 1024) {
    if constexpr (normType == NormType::LayerNorm) {
      kNormSmall<T, NormType::LayerNorm>
          <<<gridSize, blockSize, 0, stream>>>(outPtr, inputPtr, weightPtr, biasPtr, dim, eps);
    } else {
      kNormSmall<T, NormType::RMSNorm>
          <<<gridSize, blockSize, 0, stream>>>(outPtr, inputPtr, weightPtr, biasPtr, dim, eps);
    }
  } else {
    size_t sharedMemSize = sizeof(T) * 2;  // mean + var
    if constexpr (normType == NormType::LayerNorm) {
      kNormLarge<T, NormType::LayerNorm>
          <<<gridSize, blockSize, sharedMemSize, stream>>>(outPtr, inputPtr, weightPtr, biasPtr, dim, eps);
    } else {
      kNormLarge<T, NormType::RMSNorm>
          <<<gridSize, blockSize, sharedMemSize, stream>>>(outPtr, inputPtr, weightPtr, biasPtr, dim, eps);
    }
  }
  CUDA_KERNEL_CHECK();
  return out;
}

template <typename T>
Tensor layerNormOpCudaImpl(const Tensor& self, IntArrayView normalizedShape, const Tensor& weight, const Tensor& bias,
                           float eps) {
  return normOpCudaImplDetail<T, NormType::LayerNorm>(self, normalizedShape, weight, bias, eps);
}

template <typename T>
Tensor rmsNormOpCudaImpl(const Tensor& self, IntArrayView normalizedShape, const Tensor& weight, float eps) {
  return normOpCudaImplDetail<T, NormType::RMSNorm>(self, normalizedShape, weight, {}, eps);
}

template <typename T>
TensorPair ropeInitOpCudaImpl(int64_t headDim, int64_t contextLength, float thetaBase,
                              std::optional<RopeScalingConfig> scaling, Options options) {
  ASSERT(!options.requiresGrad_);
  ASSERT(options.device_.type == DeviceType::CUDA);
  ASSERT(options.dtype_ == DType::Float32);

  ASSERT(headDim % 2 == 0);
  int64_t halfDim = headDim >> 1;

  // inverse frequency
  Tensor invFreq({halfDim}, options);
  auto* invFreqPtr = invFreq.dataPtr<T>();

  auto params = cuda::getKernelLaunchParams(options.device_.index, halfDim);
  CUDA_LAUNCH_KERNEL(kRopeComputeInvFreq<T>, params, invFreqPtr, halfDim, thetaBase);

  // apply scaling if needed
  if (scaling.has_value()) {
    CUDA_LAUNCH_KERNEL(kRopeApplyScaling<T>, params, invFreqPtr, halfDim,
                       static_cast<float>(scaling->originalContextLength), scaling->lowFreqFactor,
                       scaling->highFreqFactor, scaling->factor);
  }

  // precompute cos/sin
  Tensor cos({contextLength, headDim}, options);
  Tensor sin({contextLength, headDim}, options);
  auto* cosPtr = cos.dataPtr<T>();
  auto* sinPtr = sin.dataPtr<T>();

  auto blockSize = cuda::getKernelBlockSize(options.device_.index);
  auto stream = cuda::getCurrentCUDAStream(options.device_.index).stream;
  kRopePrecomputeCosSin<T><<<contextLength, blockSize, 0, stream>>>(invFreqPtr, cosPtr, sinPtr, contextLength, headDim);
  CUDA_KERNEL_CHECK();
  return {cos, sin};
}

template <typename T>
Tensor ropeApplyOpCudaImpl(const Tensor& input, const TensorPair& rope) {
  const auto& shape = input.shape();  // [batch, numHead, seqLen, headDim]
  ASSERT(shape.size() == 4);

  int64_t batch = shape[0];
  int64_t numHead = shape[1];
  int64_t seqLen = shape[2];
  int64_t headDim = shape[3];

  ASSERT(headDim % 2 == 0);
  int64_t halfDim = headDim >> 1;

  const auto* inputPtr = input.dataPtr<T>();
  const auto* cosPtr = rope.first.dataPtr<T>();
  const auto* sinPtr = rope.second.dataPtr<T>();

  Tensor out(shape, input.options().noGrad());
  auto* outPtr = out.dataPtr<T>();

  dim3 gridSize(seqLen, numHead, batch);
  dim3 blockSize = halfDim;
  auto stream = cuda::getCurrentCUDAStream(input.device().index).stream;
  kRopeApply<T><<<gridSize, blockSize, 0, stream>>>(outPtr, inputPtr, cosPtr, sinPtr, numHead, seqLen, headDim);
  CUDA_KERNEL_CHECK();
  return out;
}

}  // namespace tinytorch::op
