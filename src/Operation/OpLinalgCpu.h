/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "OpLinalg.h"

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#elif __BLAS__
#include <cblas.h>
#endif

namespace tinytorch::op {

template <typename T>
Tensor dotOpCpuImpl(const Tensor& self, const Tensor& other) {
  T ret = 0;
  const T* selfPtr = self.dataPtr<T>();
  const T* otherPtr = other.dataPtr<T>();
  for (auto i = 0; i < self.numel(); i++) {
    ret += selfPtr[i] * otherPtr[i];
  }
  return Tensor::scalar(ret, self.options().noGrad());
}

template <typename T>
Tensor im2colOpCpuImpl(const Tensor& self, Dim2D kernel, Dim2D stride, Dim2D padding = 0) {
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

  for (int64_t b = 0; b < batch; b++) {
    for (int64_t c = 0; c < channels; c++) {
      const T* imPtr = selfPtr + (b * channels + c) * height * width;

      for (int64_t h = 0; h < outH; h++) {
        for (int64_t w = 0; w < outW; w++) {
          int64_t hStride = h * stride.h - padding.h;
          int64_t wStride = w * stride.w - padding.w;
          T* colPtr = retPtr + ((b * colH + h * outW + w) * colW) + (c * kernel.h * kernel.w);

          for (int64_t i = 0; i < kernel.h; i++) {
            for (int64_t j = 0; j < kernel.w; j++) {
              int64_t ih = hStride + i;
              int64_t iw = wStride + j;
              colPtr[i * kernel.w + j] =
                  (ih >= 0 && ih < height && iw >= 0 && iw < width) ? imPtr[ih * width + iw] : T(0);
            }
          }
        }
      }
    }
  }
  return ret;
}

template <typename T>
Tensor col2imOpCpuImpl(const Tensor& self, const IntArrayView shape, Dim2D kernel, Dim2D stride, Dim2D padding = 0) {
  // shape: [C, H, W], [N, C, H, W]
  ASSERT(shape.size() == 3 || shape.size() == 4);
  int64_t batch = (shape.size() == 4) ? shape[0] : 1;
  int64_t channels = (shape.size() == 4) ? shape[1] : shape[0];
  int64_t height = (shape.size() == 4) ? shape[2] : shape[1];
  int64_t width = (shape.size() == 4) ? shape[3] : shape[2];

  auto outH = (height - kernel.h + 2 * padding.h) / stride.h + 1;
  auto outW = (width - kernel.w + 2 * padding.w) / stride.w + 1;

  int64_t colH = outH * outW;
  int64_t colW = channels * kernel.h * kernel.w;

  auto ret = Tensor::zeros(shape, self.options().noGrad());
  const T* selfPtr = self.dataPtr<T>();
  T* retPtr = ret.dataPtr<T>();

  for (int64_t b = 0; b < batch; b++) {
    for (int64_t c = 0; c < channels; c++) {
      T* imPtr = retPtr + (b * channels + c) * height * width;

      for (int64_t h = 0; h < outH; h++) {
        for (int64_t w = 0; w < outW; w++) {
          int64_t hStride = h * stride.h - padding.h;
          int64_t wStride = w * stride.w - padding.w;
          const T* colPtr = selfPtr + ((b * colH + h * outW + w) * colW) + (c * kernel.h * kernel.w);

          for (int64_t i = 0; i < kernel.h; i++) {
            for (int64_t j = 0; j < kernel.w; j++) {
              int64_t ih = hStride + i;
              int64_t iw = wStride + j;
              if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                imPtr[ih * width + iw] += colPtr[i * kernel.w + j];
              }
            }
          }
        }
      }
    }
  }
  return ret;
}

template <typename T>
void gemmCpuImpl(T* c, const T* a, const T* b, int64_t m, int64_t k, int64_t n, bool transA, bool transB) {
  // blas
#if defined(__APPLE__) || defined(__BLAS__)
  if constexpr (std::is_same_v<T, float>) {
    CBLAS_TRANSPOSE ta = transA ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE tb = transB ? CblasTrans : CblasNoTrans;
    cblas_sgemm(CblasRowMajor, ta, tb, (int)m, (int)n, (int)k, 1.0f, a, transA ? (int)m : (int)k, b,
                transB ? (int)k : (int)n, 0.0f, c, (int)n);
    return;
  }
#endif
  // basic
  std::memset(c, 0, m * n * sizeof(T));
  for (int64_t i = 0; i < m; i++) {
    for (int64_t p = 0; p < k; p++) {
      T aVal = transA ? a[p * m + i] : a[i * k + p];
      for (int64_t j = 0; j < n; j++) {
        T bVal = transB ? b[j * k + p] : b[p * n + j];
        c[i * n + j] += aVal * bVal;
      }
    }
  }
}

template <>
void gemmImpl<float, DeviceType::CPU>(float* c, const float* a, const float* b, int64_t m, int64_t k, int64_t n,
                                      bool transA, bool transB, DeviceIndex device) {
  gemmCpuImpl<float>(c, a, b, m, k, n, transA, transB);
}

template <>
void gemmImpl<Half, DeviceType::CPU>(Half* c, const Half* a, const Half* b, int64_t m, int64_t k, int64_t n,
                                     bool transA, bool transB, DeviceIndex device) {
  gemmCpuImpl<Half>(c, a, b, m, k, n, transA, transB);
}

template <>
void gemmImpl<BFloat16, DeviceType::CPU>(BFloat16* c, const BFloat16* a, const BFloat16* b, int64_t m, int64_t k,
                                         int64_t n, bool transA, bool transB, DeviceIndex device) {
  gemmCpuImpl<BFloat16>(c, a, b, m, k, n, transA, transB);
}

}  // namespace tinytorch::op
