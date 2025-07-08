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
Tensor dropoutOpCudaImpl(const Tensor& grad, const Tensor& mask, float p) {
  TensorIteratorCuda iterator(grad, mask);
  auto outShape = iterator.setupBroadcast();
  ASSERT(iterator.isBroadcastOk());
  Tensor out(outShape, grad.options().noGrad());
  iterator.template forEach<T>(out, [p] __device__(const T& a, const T& b) -> T { return a * b / p; });
  return out;
}

}  // namespace tinytorch::op
