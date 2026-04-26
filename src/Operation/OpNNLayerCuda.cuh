/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "OpNNLayer.h"
#include "OpReduceCuda.cuh"
#include "Tensor/TensorIterator.cuh"
#include "Utils/CUDAMath.h"
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
  float maxVal = std::numeric_limits<float>::lowest();
  for (auto i = tid; i < dimSize; i += blockDim.x) {
    auto idx = base + i * innerSize;
    maxVal = cuda::max(maxVal, static_cast<float>(self[idx]));
  }
  auto blockMax = cudaBlockReduce<float, OpCudaReduceMax>(maxVal, std::numeric_limits<float>::lowest());

  // sum of exp
  float sum = 0;
  for (auto i = tid; i < dimSize; i += blockDim.x) {
    auto idx = base + i * innerSize;
    sum += cuda::exp(static_cast<float>(self[idx]) - blockMax);
  }
  auto blockSum = cudaBlockReduce<float, OpCudaReduceSum>(sum, 0.f);

  // output
  if constexpr (type == SoftmaxType::Softmax) {
    for (auto i = tid; i < dimSize; i += blockDim.x) {
      auto idx = base + i * innerSize;
      out[idx] = static_cast<T>(cuda::exp(static_cast<float>(self[idx]) - blockMax) / blockSum);
    }
  } else {
    float logSum = cuda::log(blockSum);
    for (auto i = tid; i < dimSize; i += blockDim.x) {
      auto idx = base + i * innerSize;
      out[idx] = static_cast<T>(static_cast<float>(self[idx]) - blockMax - logSum);
    }
  }
}

template <typename T, SoftmaxType type>
__global__ void kSoftmaxForwardLargeDim(T* out, const T* self, int64_t dimSize, int64_t innerSize) {
  __shared__ float shared[2];  // max, sum

  auto blockPerRow = gridDim.x;
  auto blockId = blockIdx.x;
  auto outer = blockIdx.y;
  auto inner = blockIdx.z;
  auto tid = threadIdx.x;

  auto tileSize = (dimSize + blockPerRow - 1) / blockPerRow;
  auto tileStart = blockId * tileSize;
  auto tileEnd = cuda::min(tileStart + tileSize, dimSize);

  auto base = (outer * dimSize * innerSize) + inner;

  // max
  float localMax = std::numeric_limits<float>::lowest();
  for (auto i = tileStart + tid; i < tileEnd; i += blockDim.x) {
    auto idx = base + i * innerSize;
    localMax = cuda::max(localMax, static_cast<float>(self[idx]));
  }
  auto tileMax = cudaBlockReduce<float, OpCudaReduceMax>(localMax, std::numeric_limits<float>::lowest());
  if (tid == 0) shared[blockId] = tileMax;
  __syncthreads();

  float globalMax = std::numeric_limits<float>::lowest();
  if (blockId == 0) {
    for (auto i = tid; i < blockPerRow; i += blockDim.x) {
      globalMax = cuda::max(globalMax, shared[i]);
    }
    globalMax = cudaBlockReduce<float, OpCudaReduceMax>(globalMax, std::numeric_limits<float>::lowest());
    if (tid == 0) shared[blockPerRow] = globalMax;
  }
  __syncthreads();
  globalMax = shared[blockPerRow];

  // sum of exp
  float localSum = 0;
  for (auto i = tileStart + tid; i < tileEnd; i += blockDim.x) {
    auto idx = base + i * innerSize;
    localSum += cuda::exp(static_cast<float>(self[idx]) - globalMax);
  }
  auto tileSum = cudaBlockReduce<float, OpCudaReduceSum>(localSum, 0.f);
  if (tid == 0) shared[blockId] = tileSum;
  __syncthreads();

  float globalSum = 0;
  if (blockId == 0) {
    for (auto i = tid; i < blockPerRow; i += blockDim.x) {
      globalSum += shared[i];
    }
    globalSum = cudaBlockReduce<float, OpCudaReduceSum>(globalSum, 0.f);
    if (tid == 0) shared[blockPerRow] = globalSum;
  }
  __syncthreads();
  globalSum = shared[blockPerRow];

  // output
  if constexpr (type == SoftmaxType::Softmax) {
    for (auto i = tileStart + tid; i < tileEnd; i += blockDim.x) {
      auto idx = base + i * innerSize;
      out[idx] = static_cast<T>(cuda::exp(static_cast<float>(self[idx]) - globalMax) / globalSum);
    }
  } else {
    float logSum = cuda::log(globalSum);
    for (auto i = tileStart + tid; i < tileEnd; i += blockDim.x) {
      auto idx = base + i * innerSize;
      out[idx] = static_cast<T>(static_cast<float>(self[idx]) - globalMax - logSum);
    }
  }
}

template <typename T>
struct OpCudaSoftmaxBackward {
  __device__ static float computePartialSum(const T* output, const T* grad, int64_t idx) {
    return static_cast<float>(output[idx]) * static_cast<float>(grad[idx]);
  }

  __device__ static float computeResult(const T* output, const T* grad, float totalSum, int64_t idx) {
    return static_cast<float>(output[idx]) * (static_cast<float>(grad[idx]) - totalSum);
  }
};

template <typename T>
struct OpCudaLogSoftmaxBackward {
  __device__ static float computePartialSum(const T* output, const T* grad, int64_t idx) {
    return static_cast<float>(grad[idx]);
  }

  __device__ static float computeResult(const T* output, const T* grad, float totalSum, int64_t idx) {
    return static_cast<float>(grad[idx]) - cuda::exp(static_cast<float>(output[idx])) * totalSum;
  }
};

template <typename T, typename Op>
__global__ void kSoftmaxBackward(T* out, const T* output, const T* grad, int64_t dimSize, int64_t innerSize) {
  auto outer = blockIdx.x;
  auto inner = blockIdx.y;
  auto tid = threadIdx.x;
  auto base = (outer * dimSize * innerSize) + inner;

  float sum = 0;
  for (auto i = tid; i < dimSize; i += blockDim.x) {
    auto idx = base + i * innerSize;
    sum += Op::computePartialSum(output, grad, idx);
  }
  auto blockSum = cudaBlockReduce<float, OpCudaReduceSum>(sum, 0.f);

  for (auto i = tid; i < dimSize; i += blockDim.x) {
    auto idx = base + i * innerSize;
    out[idx] = static_cast<T>(Op::computeResult(output, grad, blockSum, idx));
  }
}

template <typename T, typename Op, bool isPhase1>
__global__ void kSoftmaxBackwardLarge(T* out, const T* output, const T* grad, float* partialSums, int64_t dimSize,
                                      int64_t innerSize, int64_t numBlocks) {
  auto outer = blockIdx.z;
  auto inner = blockIdx.y;
  auto blockId = blockIdx.x;
  auto tid = threadIdx.x;
  auto base = (outer * dimSize * innerSize) + inner;

  auto elemsPerBlock = (dimSize + numBlocks - 1) / numBlocks;
  auto start = blockId * elemsPerBlock;
  auto end = cuda::min(start + elemsPerBlock, dimSize);

  if constexpr (isPhase1) {
    float sum = 0;
    for (auto i = start + tid; i < end; i += blockDim.x) {
      auto idx = base + i * innerSize;
      sum += Op::computePartialSum(output, grad, idx);
    }
    auto blockSum = cudaBlockReduce<float, OpCudaReduceSum>(sum, 0.f);

    if (tid == 0) {
      auto partialIdx = (outer * innerSize + inner) * numBlocks + blockId;
      partialSums[partialIdx] = blockSum;
    }
  } else {
    float totalSum = 0;
    auto partialBase = (outer * innerSize + inner) * numBlocks;
    for (auto i = tid; i < numBlocks; i += blockDim.x) {
      totalSum += partialSums[partialBase + i];
    }
    auto blockSum = cudaBlockReduce<float, OpCudaReduceSum>(totalSum, 0.f);

    for (auto i = start + tid; i < end; i += blockDim.x) {
      auto idx = base + i * innerSize;
      out[idx] = static_cast<T>(Op::computeResult(output, grad, blockSum, idx));
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

    T tp = static_cast<T>(p);
    if (index + 3 < n) {
      out[index] = rand.x < p ? (self[index] / tp) : T(0.f);
      out[index + 1] = rand.y < p ? (self[index + 1] / tp) : T(0.f);
      out[index + 2] = rand.z < p ? (self[index + 2] / tp) : T(0.f);
      out[index + 3] = rand.w < p ? (self[index + 3] / tp) : T(0.f);
    } else {
      if (index < n) out[index] = rand.x < p ? (self[index] / tp) : T(0.f);
      if (index + 1 < n) out[index + 1] = rand.y < p ? (self[index + 1] / tp) : T(0.f);
      if (index + 2 < n) out[index + 2] = rand.z < p ? (self[index + 2] / tp) : T(0.f);
    }
  }
}

template <typename T, NormType normType>
__global__ void kNormSmall(T* out, const T* input, const T* weight, const T* bias, int64_t dim, float eps) {
  auto row = blockIdx.x;
  auto tid = threadIdx.x;
  auto base = row * dim;

  float sum = 0;
  for (auto i = tid; i < dim; i += blockDim.x) {
    if constexpr (normType == NormType::LayerNorm) {
      sum += static_cast<float>(input[base + i]);
    } else {  // RMSNorm
      auto val = static_cast<float>(input[base + i]);
      sum += val * val;
    }
  }

  float stat = cudaBlockReduce<float, OpCudaReduceSum>(sum, 0.f) / static_cast<float>(dim);

  float invStd;
  if constexpr (normType == NormType::LayerNorm) {
    float mean = stat;
    // var
    float varSum = 0;
    for (auto i = tid; i < dim; i += blockDim.x) {
      float diff = static_cast<float>(input[base + i]) - mean;
      varSum += diff * diff;
    }
    float var = cudaBlockReduce<float, OpCudaReduceSum>(varSum, 0.f) / static_cast<float>(dim);
    invStd = cuda::rsqrt(var + eps);

    // norm + affine
    for (auto i = tid; i < dim; i += blockDim.x) {
      float normed = (static_cast<float>(input[base + i]) - mean) * invStd;
      if (weight) normed *= static_cast<float>(weight[i]);
      if (bias) normed += static_cast<float>(bias[i]);
      out[base + i] = static_cast<T>(normed);
    }
  } else {
    invStd = cuda::rsqrt(stat + eps);

    // norm + affine
    for (auto i = tid; i < dim; i += blockDim.x) {
      float normed = static_cast<float>(input[base + i]) * invStd;
      if (weight) normed *= static_cast<float>(weight[i]);
      out[base + i] = static_cast<T>(normed);
    }
  }
}

template <typename T, NormType normType>
__global__ void kNormLarge(T* out, const T* input, const T* weight, const T* bias, int64_t dim, float eps) {
  __shared__ float sharedMem[2];  // mean, var
  float* sharedStats = sharedMem;

  auto row = blockIdx.x;
  auto tid = threadIdx.x;
  auto base = row * dim;

  float sum = 0;
  for (auto i = tid; i < dim; i += blockDim.x) {
    if constexpr (normType == NormType::LayerNorm) {
      sum += static_cast<float>(input[base + i]);
    } else {  // RMSNorm
      auto val = static_cast<float>(input[base + i]);
      sum += val * val;
    }
  }

  auto stat = cudaBlockReduce<float, OpCudaReduceSum>(sum, 0.f);
  if (tid == 0) sharedStats[0] = stat / static_cast<float>(dim);
  __syncthreads();
  stat = sharedStats[0];

  float invStd;
  if constexpr (normType == NormType::LayerNorm) {
    float mean = stat;
    // var
    float varSum = 0;
    for (auto i = tid; i < dim; i += blockDim.x) {
      float diff = static_cast<float>(input[base + i]) - mean;
      varSum += diff * diff;
    }
    auto var = cudaBlockReduce<float, OpCudaReduceSum>(varSum, 0.f);
    if (tid == 0) sharedStats[1] = var / static_cast<float>(dim);
    __syncthreads();
    var = sharedStats[1];
    invStd = cuda::rsqrt(var + eps);

    // norm + affine
    for (auto i = tid; i < dim; i += blockDim.x) {
      float normed = (static_cast<float>(input[base + i]) - mean) * invStd;
      if (weight) normed *= static_cast<float>(weight[i]);
      if (bias) normed += static_cast<float>(bias[i]);
      out[base + i] = static_cast<T>(normed);
    }
  } else {
    invStd = cuda::rsqrt(stat + eps);

    // norm + affine
    for (auto i = tid; i < dim; i += blockDim.x) {
      float normed = static_cast<float>(input[base + i]) * invStd;
      if (weight) normed *= static_cast<float>(weight[i]);
      out[base + i] = static_cast<T>(normed);
    }
  }
}

template <typename T, SoftmaxType type>
void softmaxForwardCudaImpl(Tensor& out, const Tensor& self, int64_t dim) {
  ASSERT(out.shape() == self.shape());
  auto info = getSoftmaxDimInfo(self, dim);

  using CudaT = typename cuda::CudaTypeCast<T>::type;
  const CudaT* selfPtr = self.dataPtr<CudaT>();
  CudaT* outPtr = out.dataPtr<CudaT>();

  if (info.dimSize <= 1024) {
    dim3 block(std::clamp(nextPow2(info.dimSize), 32u, 1024u));
    dim3 grid(info.outerSize, info.innerSize);
    const auto& stream = cuda::getCurrentCUDAStream(self.device().index).stream();
    kSoftmaxForward<CudaT, type><<<grid, block, 0, stream>>>(outPtr, selfPtr, info.dimSize, info.innerSize);
  } else {
    auto blockSize = cuda::getKernelBlockSize(self.device().index);
    dim3 grid(1, info.outerSize, info.innerSize);
    const auto& stream = cuda::getCurrentCUDAStream(self.device().index).stream();
    kSoftmaxForwardLargeDim<CudaT, type><<<grid, blockSize, 0, stream>>>(outPtr, selfPtr, info.dimSize, info.innerSize);
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

  using CudaT = typename cuda::CudaTypeCast<T>::type;
  const CudaT* outputPtr = output.dataPtr<CudaT>();
  const CudaT* gradPtr = grad.dataPtr<CudaT>();
  CudaT* outPtr = out.dataPtr<CudaT>();
  const auto& stream = cuda::getCurrentCUDAStream(output.device().index).stream();

  if (info.dimSize <= 1024) {
    dim3 block(std::clamp(nextPow2(info.dimSize), 32u, 1024u));
    dim3 grid(info.outerSize, info.innerSize);
    kSoftmaxBackward<CudaT, Op><<<grid, block, 0, stream>>>(outPtr, outputPtr, gradPtr, info.dimSize, info.innerSize);
  } else {
    constexpr int64_t maxBlocksPerDim = 16;
    const int64_t numBlocks = std::min(maxBlocksPerDim, (info.dimSize + 1023) / 1024);
    const int64_t partialSumsSize = info.outerSize * info.innerSize * numBlocks;

    Storage partialSums(static_cast<int64_t>(partialSumsSize * sizeof(float)), output.device());
    auto* partialSumsPtr = partialSums.dataPtr<float>();

    dim3 block(cuda::getKernelBlockSize(output.device().index));
    dim3 grid(numBlocks, info.innerSize, info.outerSize);

    // phase 1
    kSoftmaxBackwardLarge<CudaT, Op, true><<<grid, block, 0, stream>>>(outPtr, outputPtr, gradPtr, partialSumsPtr,
                                                                       info.dimSize, info.innerSize, numBlocks);

    // phase 2
    kSoftmaxBackwardLarge<CudaT, Op, false><<<grid, block, 0, stream>>>(outPtr, outputPtr, gradPtr, partialSumsPtr,
                                                                        info.dimSize, info.innerSize, numBlocks);
  }
  CUDA_KERNEL_CHECK();
  return out;
}

template <typename T>
Tensor softmaxOpBackwardCudaImpl(const Tensor& grad, const Tensor& output, int64_t dim) {
  using CudaT = typename cuda::CudaTypeCast<T>::type;
  return softmaxBackwardCudaImplDetail<CudaT, OpCudaSoftmaxBackward<CudaT>>(grad, output, dim);
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
  using CudaT = typename cuda::CudaTypeCast<T>::type;
  return softmaxBackwardCudaImplDetail<CudaT, OpCudaLogSoftmaxBackward<CudaT>>(grad, output, dim);
}

template <typename T>
Tensor dropoutOpCudaImpl(const Tensor& self, float p) {
  using CudaT = typename cuda::CudaTypeCast<T>::type;
  Tensor out(self.shape(), self.options().noGrad());
  const auto* selfPtr = self.dataPtr<CudaT>();
  auto* outPtr = out.dataPtr<CudaT>();

  auto seed = RandomGeneratorCUDA::getSeed();
  auto seq = RandomGeneratorCUDA::nextSequence();
  int64_t n = self.numel();

  auto params = cuda::getKernelLaunchParams(self.device().index, n, 4);
  CUDA_LAUNCH_KERNEL(kDropout<CudaT>, params, outPtr, selfPtr, p, seed, seq, n);
  return out;
}

template <typename T>
Tensor dropoutMaskedOpCudaImpl(const Tensor& self, const Tensor& mask, float p) {
  TensorIteratorCuda iterator(self, mask);
  auto outShape = iterator.setupBroadcast();
  ASSERT(iterator.isBroadcastOk());
  Tensor out(outShape, self.options().noGrad());
  using CudaT = typename cuda::CudaTypeCast<T>::type;
  auto tp = static_cast<CudaT>(p);
  iterator.template forEach<CudaT>(out,
                                   [tp] __device__(const CudaT& a, const CudaT& b) -> CudaT { return a * b / tp; });
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

  using CudaT = typename cuda::CudaTypeCast<T>::type;
  const CudaT* inputPtr = self.dataPtr<CudaT>();
  const CudaT* weightPtr = weight.defined() ? weight.dataPtr<CudaT>() : nullptr;
  const CudaT* biasPtr = bias.defined() ? bias.dataPtr<CudaT>() : nullptr;
  CudaT* outPtr = out.dataPtr<CudaT>();

  auto stream = cuda::getCurrentCUDAStream(self.device().index).stream();
  dim3 blockSize(std::clamp(nextPow2(dim), 32u, 1024u));
  dim3 gridSize(numRows);

  if (dim <= 1024) {
    if constexpr (normType == NormType::LayerNorm) {
      kNormSmall<CudaT, NormType::LayerNorm>
          <<<gridSize, blockSize, 0, stream>>>(outPtr, inputPtr, weightPtr, biasPtr, dim, eps);
    } else {
      kNormSmall<CudaT, NormType::RMSNorm>
          <<<gridSize, blockSize, 0, stream>>>(outPtr, inputPtr, weightPtr, biasPtr, dim, eps);
    }
  } else {
    if constexpr (normType == NormType::LayerNorm) {
      kNormLarge<CudaT, NormType::LayerNorm>
          <<<gridSize, blockSize, 0, stream>>>(outPtr, inputPtr, weightPtr, biasPtr, dim, eps);
    } else {
      kNormLarge<CudaT, NormType::RMSNorm>
          <<<gridSize, blockSize, 0, stream>>>(outPtr, inputPtr, weightPtr, biasPtr, dim, eps);
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

}  // namespace tinytorch::op
