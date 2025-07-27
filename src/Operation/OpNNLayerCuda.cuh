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

template <typename T, SoftmaxType type>
__global__ void kSoftmaxForward(T* out, const T* self, int64_t dimSize, int64_t innerSize) {
  __shared__ T warpShared[32];

  auto outer = blockIdx.x;
  auto inner = blockIdx.y;
  auto tid = threadIdx.x;

  auto base = (outer * dimSize * innerSize) + inner;

  // max
  T maxVal = -cuda::Inf<T>();
  for (auto i = tid; i < dimSize; i += blockDim.x) {
    auto idx = base + i * innerSize;
    maxVal = ::max(maxVal, self[idx]);
  }
  cudaWarpReduce<T, OpCudaReduceMax>(maxVal);

  auto lane = tid & 0x1f;
  auto wid = tid >> 5;
  if (lane == 0) {
    warpShared[wid] = maxVal;
  }
  __syncthreads();

  T blockMax = -cuda::Inf<T>();
  if (wid == 0) {
    blockMax = (tid < (blockDim.x >> 5)) ? warpShared[lane] : -cuda::Inf<T>();
    cudaWarpReduce<T, OpCudaReduceMax>(blockMax);
    if (tid == 0) {
      warpShared[0] = blockMax;
    }
  }
  __syncthreads();
  blockMax = warpShared[0];

  // sum of exp
  T sum = 0;
  for (auto i = tid; i < dimSize; i += blockDim.x) {
    auto idx = base + i * innerSize;
    sum += ::expf(self[idx] - blockMax);
  }
  cudaWarpReduce<T, OpCudaReduceSum>(sum);

  if (lane == 0) {
    warpShared[wid] = sum;
  }
  __syncthreads();

  T blockSum = 0;
  if (wid == 0) {
    blockSum = (tid < (blockDim.x >> 5)) ? warpShared[lane] : 0;
    cudaWarpReduce<T, OpCudaReduceSum>(blockSum);
    if (tid == 0) {
      warpShared[0] = blockSum;
    }
  }
  __syncthreads();
  blockSum = warpShared[0];

  // output
  if constexpr (type == SoftmaxType::Softmax) {
    for (auto i = tid; i < dimSize; i += blockDim.x) {
      auto idx = base + i * innerSize;
      out[idx] = ::expf(self[idx] - blockMax) / blockSum;
    }
  } else {  // LogSoftmax
    T logSum = ::logf(blockSum);
    for (auto i = tid; i < dimSize; i += blockDim.x) {
      auto idx = base + i * innerSize;
      out[idx] = self[idx] - blockMax - logSum;
    }
  }
}

template <typename T>
__global__ void kSoftmaxBackward(T* out, const T* output, const T* grad, int64_t dimSize, int64_t innerSize) {
  __shared__ T warpShared[32];

  auto outer = blockIdx.x;
  auto inner = blockIdx.y;
  auto tid = threadIdx.x;

  auto base = (outer * dimSize * innerSize) + inner;

  // sum_j(y_j * dL/dy_j)
  T sum = 0;
  for (auto i = tid; i < dimSize; i += blockDim.x) {
    auto idx = base + i * innerSize;
    sum += output[idx] * grad[idx];
  }
  cudaWarpReduce<T, OpCudaReduceSum>(sum);

  auto lane = tid & 0x1f;
  auto wid = tid >> 5;
  if (lane == 0) {
    warpShared[wid] = sum;
  }
  __syncthreads();

  T blockSum = 0;
  if (wid == 0) {
    blockSum = (tid < (blockDim.x >> 5)) ? warpShared[lane] : 0;
    cudaWarpReduce<T, OpCudaReduceSum>(blockSum);
    if (tid == 0) {
      warpShared[0] = blockSum;
    }
  }
  __syncthreads();
  blockSum = warpShared[0];

  // dL/dx_i = y_i * (dL/dy_i - sum)
  for (auto i = tid; i < dimSize; i += blockDim.x) {
    auto idx = base + i * innerSize;
    out[idx] = output[idx] * (grad[idx] - blockSum);
  }
}

template <typename T>
__global__ void kLogSoftmaxBackward(T* out, const T* output, const T* grad, int64_t dimSize, int64_t innerSize) {
  __shared__ T warpShared[32];

  auto outer = blockIdx.x;
  auto inner = blockIdx.y;
  auto tid = threadIdx.x;

  auto base = (outer * dimSize * innerSize) + inner;

  // sum_j(dL/dy_j)
  T sum = 0;
  for (auto i = tid; i < dimSize; i += blockDim.x) {
    auto idx = base + i * innerSize;
    sum += grad[idx];
  }
  cudaWarpReduce<T, OpCudaReduceSum>(sum);

  auto lane = tid & 0x1f;
  auto wid = tid >> 5;
  if (lane == 0) {
    warpShared[wid] = sum;
  }
  __syncthreads();

  T blockSum = 0;
  if (wid == 0) {
    blockSum = (tid < (blockDim.x >> 5)) ? warpShared[lane] : 0;
    cudaWarpReduce<T, OpCudaReduceSum>(blockSum);
    if (tid == 0) {
      warpShared[0] = blockSum;
    }
  }
  __syncthreads();
  blockSum = warpShared[0];

  // dL/dx_i = dL/dy_i - exp(y_i) * sum
  for (auto i = tid; i < dimSize; i += blockDim.x) {
    auto idx = base + i * innerSize;
    out[idx] = grad[idx] - ::expf(output[idx]) * blockSum;
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

template <typename T>
__global__ void kLayerNormForward(T* out, const T* self, const T* weight, const T* bias, int64_t d, float eps) {
  __shared__ T warpShared[32];

  auto row = blockIdx.x;
  auto tid = threadIdx.x;
  auto lane = tid & 0x1f;
  auto wid = tid >> 5;

  auto base = row * d;

  // mean
  T mean = 0;
  for (auto i = tid; i < d; i += blockDim.x) {
    mean += self[base + i];
  }
  cudaWarpReduce<T, OpCudaReduceSum>(mean);

  if (lane == 0) {
    warpShared[wid] = mean;
  }
  __syncthreads();

  T blockMean = 0;
  if (wid == 0) {
    blockMean = (tid < (blockDim.x >> 5)) ? warpShared[lane] : 0;
    cudaWarpReduce<T, OpCudaReduceSum>(blockMean);
    if (tid == 0) {
      warpShared[0] = blockMean;
    }
  }
  __syncthreads();
  blockMean = warpShared[0] / static_cast<T>(d);

  // var
  T var = 0;
  for (auto i = tid; i < d; i += blockDim.x) {
    T diff = self[base + i] - blockMean;
    var += diff * diff;
  }
  cudaWarpReduce<T, OpCudaReduceSum>(var);

  if (lane == 0) {
    warpShared[wid] = var;
  }
  __syncthreads();

  T blockVar = 0;
  if (wid == 0) {
    blockVar = (tid < (blockDim.x >> 5)) ? warpShared[lane] : 0;
    cudaWarpReduce<T, OpCudaReduceSum>(blockVar);
    if (tid == 0) {
      warpShared[0] = blockVar;
    }
  }
  __syncthreads();
  blockVar = warpShared[0] / static_cast<T>(d);
  T invStd = static_cast<T>(1) / sqrtf(blockVar + eps);

  // norm + affine
  for (auto i = tid; i < d; i += blockDim.x) {
    T normed = (self[base + i] - blockMean) * invStd;
    if (weight) normed *= weight[i];
    if (bias) normed += bias[i];
    out[base + i] = normed;
  }
}

template <typename T, SoftmaxType type>
void softmaxForwardCudaImpl(Tensor& out, const Tensor& self, int64_t dim) {
  ASSERT(out.shape() == self.shape());
  auto info = getSoftmaxDimInfo(self, dim);
  ASSERT(info.dimSize <= 1024);

  const T* selfPtr = self.dataPtr<T>();
  T* outPtr = out.dataPtr<T>();

  dim3 block(std::clamp(nextPow2(info.dimSize), 32u, 1024u));
  dim3 grid(info.outerSize, info.innerSize);
  const auto stream = cuda::getCurrentCUDAStream(self.device().index).stream;
  kSoftmaxForward<T, type><<<grid, block, 0, stream>>>(outPtr, selfPtr, info.dimSize, info.innerSize);
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

template <typename T>
Tensor softmaxOpBackwardCudaImpl(const Tensor& grad, const Tensor& output, int64_t dim) {
  ASSERT(output.shape() == grad.shape());
  auto info = getSoftmaxDimInfo(output, dim);
  ASSERT(info.dimSize <= 1024);
  Tensor out(output.shape(), output.options().noGrad());

  const T* outputPtr = output.dataPtr<T>();
  const T* gradPtr = grad.dataPtr<T>();
  T* outPtr = out.dataPtr<T>();

  dim3 block(std::clamp(nextPow2(info.dimSize), 32u, 1024u));
  dim3 grid(info.outerSize, info.innerSize);
  const auto stream = cuda::getCurrentCUDAStream(output.device().index).stream;
  kSoftmaxBackward<T><<<grid, block, 0, stream>>>(outPtr, outputPtr, gradPtr, info.dimSize, info.innerSize);
  CUDA_KERNEL_CHECK();
  return out;
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
  ASSERT(output.shape() == grad.shape());
  auto info = getSoftmaxDimInfo(output, dim);
  ASSERT(info.dimSize <= 1024);
  Tensor out(output.shape(), output.options().noGrad());

  const T* outputPtr = output.dataPtr<T>();
  const T* gradPtr = grad.dataPtr<T>();
  T* outPtr = out.dataPtr<T>();

  dim3 block(std::clamp(nextPow2(info.dimSize), 32u, 1024u));
  dim3 grid(info.outerSize, info.innerSize);
  const auto stream = cuda::getCurrentCUDAStream(output.device().index).stream;
  kLogSoftmaxBackward<T><<<grid, block, 0, stream>>>(outPtr, outputPtr, gradPtr, info.dimSize, info.innerSize);
  CUDA_KERNEL_CHECK();
  return out;
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

template <typename T>
Tensor layerNormOpCudaImpl(const Tensor& self, IntArrayView normalizedShape, const Tensor& weight, const Tensor& bias,
                           float eps) {
  int64_t d = self.shape().back();
  int64_t n = self.numel() / d;
  ASSERT(normalizedShape.size() == 1);
  ASSERT(normalizedShape.front() == d);
  ASSERT(d <= 1024);
  Tensor out(self.shape(), self.options().noGrad());

  const auto* selfPtr = self.dataPtr<T>();
  const auto* weightPtr = weight.defined() ? weight.dataPtr<T>() : nullptr;
  const auto* biasPtr = bias.defined() ? bias.dataPtr<T>() : nullptr;

  auto* outPtr = out.dataPtr<T>();

  dim3 block(std::clamp(nextPow2(d), 32u, 1024u));
  dim3 grid(n);
  auto stream = cuda::getCurrentCUDAStream(self.device().index).stream;
  kLayerNormForward<T><<<grid, block, 0, stream>>>(outPtr, selfPtr, weightPtr, biasPtr, d, eps);
  return out;
}

}  // namespace tinytorch::op
