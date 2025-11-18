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
    const int64_t c = (colIdx / colH) % channels;
    const int64_t h = (colIdx % colH) / outW;
    const int64_t w = (colIdx % outW);

    const int64_t kh = kernelIdx / kernelW;
    const int64_t kw = kernelIdx % kernelW;

    const T* imPtr = self + (b * channels + c) * height * width;
    T* colPtr = ret + ((b * colH + h * outW + w) * channels + c) * kernelSize;

    const int64_t ih = h * strideH - paddingH + kh;
    const int64_t iw = w * strideW - paddingW + kw;

    const bool inBounds = (ih >= 0) & (ih < height) & (iw >= 0) & (iw < width);
    const T val = inBounds ? imPtr[ih * width + iw] : static_cast<T>(0);

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
    const int64_t w = index % width;
    const int64_t h = (index / width) % height;
    const int64_t c = (index / (width * height)) % channels;
    const int64_t b = index / (width * height * channels);

    T val = static_cast<T>(0);
    const int64_t kernelSize = kernelH * kernelW;

    auto ohStart = cuda::max<int64_t>(0, (h + paddingH - kernelH + strideH) / strideH);
    auto ohEnd = cuda::min<int64_t>(outH - 1, (h + paddingH) / strideH);

    auto owStart = cuda::max<int64_t>(0, (w + paddingW - kernelW + strideW) / strideW);
    auto owEnd = cuda::min<int64_t>(outW - 1, (w + paddingW) / strideW);

    for (int64_t oh = ohStart; oh <= ohEnd; oh++) {
      for (int64_t ow = owStart; ow <= owEnd; ow++) {
        const int64_t kh = h - (oh * strideH - paddingH);
        const int64_t kw = w - (ow * strideW - paddingW);

        const int64_t colIdx = ((b * outH * outW + oh * outW + ow) * channels + c) * kernelSize + (kh * kernelW + kw);
        val += self[colIdx];
      }
    }
    ret[index] = val;
  }
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

  using CudaT = typename cuda::CudaTypeCast<T>::type;
  auto ret = Tensor::empty({batch * colH, colW}, self.options().noGrad());
  const auto* selfPtr = self.dataPtr<CudaT>();
  auto* retPtr = ret.dataPtr<CudaT>();

  int64_t n = ret.numel();
  auto params = cuda::getKernelLaunchParams(self.device().index, n);

  CUDA_LAUNCH_KERNEL(kIm2Col<CudaT>, params, retPtr, selfPtr, n, channels, height, width, outH, outW, kernel.h,
                     kernel.w, stride.h, stride.w, padding.h, padding.w);
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

  using CudaT = typename cuda::CudaTypeCast<T>::type;
  auto ret = Tensor::zeros(shape, self.options().noGrad());
  const auto* selfPtr = self.dataPtr<CudaT>();
  auto* retPtr = ret.dataPtr<CudaT>();

  int64_t n = batch * channels * height * width;
  auto params = cuda::getKernelLaunchParams(self.device().index, n);

  CUDA_LAUNCH_KERNEL(kCol2Im<CudaT>, params, retPtr, selfPtr, n, channels, height, width, outH, outW, kernel.h,
                     kernel.w, stride.h, stride.w, padding.h, padding.w);
  return ret;
}

template <typename T>
Tensor dotOpCudaImpl(const Tensor& self, const Tensor& other);

inline void gemmCudaF32Impl(float* c, const float* a, const float* b, int64_t m, int64_t k, int64_t n, bool transA,
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

inline void gemmCudaF16Impl(__half* c, const __half* a, const __half* b, int64_t m, int64_t k, int64_t n, bool transA,
                            bool transB, DeviceIndex device) {
  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  int lda = static_cast<int>(transA ? m : k);
  int ldb = static_cast<int>(transB ? k : n);
  int ldc = static_cast<int>(n);

  constexpr float alpha = 1.f;
  constexpr float beta = 0.f;

  auto handle = cuda::getCublasHandle(device);
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
  CUBLAS_CHECK(cublasGemmEx(handle, opB, opA, n, m, k, &alpha, b, CUDA_R_16F, ldb, a, CUDA_R_16F, lda, &beta, c,
                            CUDA_R_16F, ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

inline void gemmCudaBF16Impl(__nv_bfloat16* c, const __nv_bfloat16* a, const __nv_bfloat16* b, int64_t m, int64_t k,
                             int64_t n, bool transA, bool transB, DeviceIndex device) {
  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  int lda = static_cast<int>(transA ? m : k);
  int ldb = static_cast<int>(transB ? k : n);
  int ldc = static_cast<int>(n);

  constexpr float alpha = 1.f;
  constexpr float beta = 0.f;

  auto handle = cuda::getCublasHandle(device);
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
  CUBLAS_CHECK(cublasGemmEx(handle, opB, opA, n, m, k, &alpha, b, CUDA_R_16BF, ldb, a, CUDA_R_16BF, lda, &beta, c,
                            CUDA_R_16BF, ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

template <>
void gemmImpl<float, DeviceType::CUDA>(float* c, const float* a, const float* b, int64_t m, int64_t k, int64_t n,
                                       bool transA, bool transB, DeviceIndex device) {
  gemmCudaF32Impl(c, a, b, m, k, n, transA, transB, device);
}

template <>
void gemmImpl<Half, DeviceType::CUDA>(Half* c, const Half* a, const Half* b, int64_t m, int64_t k, int64_t n,
                                      bool transA, bool transB, DeviceIndex device) {
  gemmCudaF16Impl(reinterpret_cast<__half*>(c), reinterpret_cast<const __half*>(a), reinterpret_cast<const __half*>(b),
                  m, k, n, transA, transB, device);
}

template <>
void gemmImpl<BFloat16, DeviceType::CUDA>(BFloat16* c, const BFloat16* a, const BFloat16* b, int64_t m, int64_t k,
                                          int64_t n, bool transA, bool transB, DeviceIndex device) {
  gemmCudaBF16Impl(reinterpret_cast<__nv_bfloat16*>(c), reinterpret_cast<const __nv_bfloat16*>(a),
                   reinterpret_cast<const __nv_bfloat16*>(b), m, k, n, transA, transB, device);
}

}  // namespace tinytorch::op
