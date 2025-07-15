/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "OpLinalg.h"
#include "Utils/CUDAUtils.h"

namespace tinytorch::op {

template <typename T>
__global__ void kIm2Col(T* ret, const T* self, const int64_t n, const int64_t channels, const int64_t height,
                        const int64_t width, const int64_t outH, const int64_t outW, const int64_t kernelH,
                        const int64_t kernelW, const int64_t strideH, const int64_t strideW, const int64_t paddingH,
                        const int64_t paddingW) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    const int64_t colH = outH * outW;
    const int64_t kernelSize = kernelH * kernelW;

    const int64_t kernelIdx = index % kernelSize;
    const int64_t colIdx = index / kernelSize;

    const int64_t b = colIdx / (channels * colH);
    const int64_t c = (colIdx % (channels * colH)) / colH;
    const int64_t h = (colIdx % colH) / outW;
    const int64_t w = colIdx % outW;

    const int64_t kh = kernelIdx / kernelW;
    const int64_t kw = kernelIdx % kernelW;

    const T* imPtr = self + (b * channels + c) * height * width;
    T* colPtr = ret + ((b * colH + h * outW + w) * channels + c) * kernelSize;

    const int64_t ih = h * strideH - paddingH + kh;
    const int64_t iw = w * strideW - paddingW + kw;
    T val = 0.f;
    if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
      val = imPtr[ih * width + iw];
    }
    colPtr[kh * kernelW + kw] = val;
  }
}

template <typename T>
__global__ void kCol2Im(T* ret, const T* self, const int64_t n, const int64_t channels, const int64_t height,
                        const int64_t width, const int64_t outH, const int64_t outW, const int64_t kernelH,
                        const int64_t kernelW, const int64_t strideH, const int64_t strideW, const int64_t paddingH,
                        const int64_t paddingW) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    const int64_t colH = outH * outW;
    const int64_t kernelSize = kernelH * kernelW;

    const int64_t b = index / (channels * colH);
    const int64_t c = (index % (channels * colH)) / colH;
    const int64_t h = (index % colH) / outW;
    const int64_t w = index % outW;

    const int64_t hStride = h * strideH - paddingH;
    const int64_t wStride = w * strideW - paddingW;

    T* imPtr = ret + (b * channels + c) * height * width;
    const T* colPtr = self + ((b * colH + h * outW + w) * channels + c) * kernelSize;

    for (int64_t i = 0; i < kernelH; i++) {
      for (int64_t j = 0; j < kernelW; j++) {
        const int64_t ih = hStride + i;
        const int64_t iw = wStride + j;
        if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
          atomicAdd(&imPtr[ih * width + iw], colPtr[i * kernelW + j]);
        }
      }
    }
  }
}

template <typename T>
__global__ void kDot(T* ret, const T* a, const T* b, const int64_t n) {
  extern __shared__ T sharedData[];

  auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const auto tid = threadIdx.x;

  T temp = 0;
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

template <typename T>
Tensor dotOpCudaImpl(const Tensor& self, const Tensor& other) {
  auto ret = Tensor::scalar(0, self.options().noGrad());

  auto params = cuda::getKernelLaunchParams(self.device().index, self.numel());
  params.sharedMemBytes = params.block.x * sizeof(T);
  CUDA_LAUNCH_KERNEL((kDot<T>), params, ret.dataPtr<T>(), self.dataPtr<T>(), other.dataPtr<T>(), self.numel());
  return ret;
}

template <typename T>
Tensor im2colOpCudaImpl(const Tensor& self, Dim2D kernel, Dim2D stride, Dim2D padding = 0) {
  // shape: [C, H, W], [N, C, H, W]
  ASSERT(self.dim() == 3 || self.dim() == 4);
  int64_t batch = (self.dim() == 4) ? self.shape(0) : 1;
  int64_t channels = (self.dim() == 4) ? self.shape(1) : self.shape(0);
  int64_t height = (self.dim() == 4) ? self.shape(2) : self.shape(1);
  int64_t width = (self.dim() == 4) ? self.shape(3) : self.shape(2);
  int64_t outH = (height - kernel.h + 2 * padding.h) / stride.h + 1;
  int64_t outW = (width - kernel.w + 2 * padding.w) / stride.w + 1;

  int64_t colH = outH * outW;
  int64_t colW = channels * kernel.h * kernel.w;

  auto ret = Tensor::empty({batch * colH, colW}, self.options().noGrad());
  const T* selfPtr = self.dataPtr<T>();
  T* retPtr = ret.dataPtr<T>();

  int64_t n = ret.numel();
  auto params = cuda::getKernelLaunchParams(self.device().index, n);

  CUDA_LAUNCH_KERNEL(kIm2Col<T>, params, retPtr, selfPtr, n, channels, height, width, outH, outW, kernel.h, kernel.w,
                     stride.h, stride.w, padding.h, padding.w);
  return ret;
}

template <typename T>
Tensor col2imOpCudaImpl(const Tensor& self, const IntArrayView shape, Dim2D kernel, Dim2D stride, Dim2D padding = 0) {
  // shape: [C, H, W], [N, C, H, W]
  ASSERT(shape.size() == 3 || shape.size() == 4);
  int64_t batch = (shape.size() == 4) ? shape[0] : 1;
  int64_t channels = (shape.size() == 4) ? shape[1] : shape[0];
  int64_t height = (shape.size() == 4) ? shape[2] : shape[1];
  int64_t width = (shape.size() == 4) ? shape[3] : shape[2];

  auto outH = (height - kernel.h + 2 * padding.h) / stride.h + 1;
  auto outW = (width - kernel.w + 2 * padding.w) / stride.w + 1;

  // int64_t colH = outH * outW;
  // int64_t colW = channels * kernel.h * kernel.w;

  auto ret = Tensor::zeros(shape, self.options().noGrad());
  const T* selfPtr = self.dataPtr<T>();
  T* retPtr = ret.dataPtr<T>();

  int64_t n = batch * channels * outH * outW;
  auto params = cuda::getKernelLaunchParams(self.device().index, n);

  CUDA_LAUNCH_KERNEL(kCol2Im<T>, params, retPtr, selfPtr, n, channels, height, width, outH, outW, kernel.h, kernel.w,
                     stride.h, stride.w, padding.h, padding.w);
  return ret;
}

void gemmCudaF32Impl(float* c, const float* a, const float* b, int64_t m, int64_t k, int64_t n, bool transA,
                     bool transB, DeviceIndex device) {
  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  int lda = static_cast<int>(transA ? m : k);
  int ldb = static_cast<int>(transB ? k : n);
  int ldc = static_cast<int>(n);

  constexpr float alpha = 1.f;
  constexpr float beta = 0.f;

  auto handle = cuda::getCublasHandle(device);
  CUBLAS_CHECK(cublasSgemm(handle, opB, opA, n, m, k, &alpha, b, ldb, a, lda, &beta, c, ldc));
}

template <>
void gemmImpl<float, DeviceType::CUDA>(float* c, const float* a, const float* b, int64_t m, int64_t k, int64_t n,
                                       bool transA, bool transB, DeviceIndex device) {
  gemmCudaF32Impl(c, a, b, m, k, n, transA, transB, device);
}

}  // namespace tinytorch::op
