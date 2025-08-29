/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "OpElemWise.h"
#include "Tensor/TensorIterator.cuh"
#include "Utils/CUDAUtils.h"

namespace tinytorch::op {

struct OpCudaAbs {
  template <typename T>
  __device__ static T apply(const T& a) {
    return cuda::abs(a);
  }
};

struct OpCudaNeg {
  template <typename T>
  __device__ static T apply(const T& a) {
    return -a;
  }
};

struct OpCudaSign {
  template <typename T>
  __device__ static T apply(const T& a) {
    return cuda::sign(a);
  }
};

struct OpCudaSqrt {
  template <typename T>
  __device__ static T apply(const T& a) {
    return cuda::sqrt(a);
  }
};

struct OpCudaSquare {
  template <typename T>
  __device__ static T apply(const T& a) {
    return a * a;
  }
};

struct OpCudaExp {
  template <typename T>
  __device__ static T apply(const T& a) {
    return cuda::exp(a);
  }
};

struct OpCudaLog {
  template <typename T>
  __device__ static T apply(const T& a) {
    return cuda::log(a);
  }
};

struct OpCudaSin {
  template <typename T>
  __device__ static T apply(const T& a) {
    return cuda::sin(a);
  }
};

struct OpCudaSinBackwardP1 {
  template <typename T>
  __device__ static T apply(const T& self, const T& grad) {
    return cuda::cos(self) * grad;
  }
};

struct OpCudaCos {
  template <typename T>
  __device__ static T apply(const T& a) {
    return cuda::cos(a);
  }
};

struct OpCudaCosBackwardP1 {
  template <typename T>
  __device__ static T apply(const T& self, const T& grad) {
    return -cuda::sin(self) * grad;
  }
};

struct OpCudaSigmoid {
  template <typename T>
  __device__ static T apply(const T& a) {
    return T(1) / (T(1) + cuda::exp(-a));
  }
};

struct OpCudaTanh {
  template <typename T>
  __device__ static T apply(const T& a) {
    return cuda::tanh(a);
  }
};

struct OpCudaRelu {
  template <typename T>
  __device__ static T apply(const T& a) {
    return a > T(0) ? a : T(0);
  }
};

struct OpCudaGelu {
  template <typename T>
  __device__ static T apply(const T& a) {
    auto fa = static_cast<float>(a);
    constexpr float sqrt2OverPi = 0.7978845608f;  // sqrt(2/pi)
    float tanhArg = sqrt2OverPi * (fa + 0.044715f * fa * fa * fa);
    auto ret = 0.5f * a * (1.0f + cuda::tanh(tanhArg));
    return static_cast<T>(ret);
  }
};

struct OpCudaSilu {
  template <typename T>
  __device__ static T apply(const T& a) {
    auto fa = static_cast<float>(a);
    auto ret = fa / (1.f + cuda::exp(-fa));
    return static_cast<T>(ret);
  }
};

struct OpCudaAdd {
  template <typename T>
  __device__ static T apply(const T& a, const T& b, const T& alpha) {
    return a + alpha * b;
  }

  template <typename T>
  __device__ static T apply(const T& a, const T& b) {
    return a + b;
  }
};

struct OpCudaSub {
  template <typename T>
  __device__ static T apply(const T& a, const T& b, const T& alpha) {
    return a - alpha * b;
  }

  template <typename T>
  __device__ static T apply(const T& a, const T& b) {
    return a - b;
  }
};

struct OpCudaMul {
  template <typename T>
  __device__ static T apply(const T& a, const T& b) {
    return a * b;
  }
};

struct OpCudaDiv {
  template <typename T>
  __device__ static T apply(const T& a, const T& b) {
    return a / b;
  }
};

struct OpCudaDivBackwardP2 {
  template <typename T>
  __device__ static T apply(const T& self, const T& other, const T& grad) {
    return -grad * self / (other * other);
  }
};

struct OpCudaPow {
  template <typename T>
  __device__ static T apply(const T& a, const T& b) {
    return cuda::pow(a, b);
  }
};

struct OpCudaPowBackwardP1 {
  template <typename T>
  __device__ static T apply(const T& a, const T& b, const T& grad) {
    return grad * b * cuda::pow(a, b - T(1));
  }
};

struct OpCudaPowBackwardP2 {
  template <typename T>
  __device__ static T apply(const T& a, const T& b, const T& grad) {
    return grad * cuda::pow(a, b) * cuda::log(a);
  }
};

struct OpCudaMaximum {
  template <typename T>
  __device__ static T apply(const T& a, const T& b) {
    return cuda::max(a, b);
  }
};

struct OpCudaMinimum {
  template <typename T>
  __device__ static T apply(const T& a, const T& b) {
    return cuda::min(a, b);
  }
};

struct OpCudaEq {
  template <typename T>
  __device__ static bool apply(const T& a, const T& b) {
    return a == b;
  }
};

struct OpCudaNe {
  template <typename T>
  __device__ static bool apply(const T& a, const T& b) {
    return a != b;
  }
};

struct OpCudaLt {
  template <typename T>
  __device__ static bool apply(const T& a, const T& b) {
    return a < b;
  }
};

struct OpCudaLe {
  template <typename T>
  __device__ static bool apply(const T& a, const T& b) {
    return a <= b;
  }
};

struct OpCudaGt {
  template <typename T>
  __device__ static bool apply(const T& a, const T& b) {
    return a > b;
  }
};

struct OpCudaGe {
  template <typename T>
  __device__ static bool apply(const T& a, const T& b) {
    return a >= b;
  }
};

struct OpCudaLogicNot {
  template <typename T>
  __device__ static T apply(const T& a) {
    return !a;
  }
};

struct OpCudaLogicAnd {
  template <typename T>
  __device__ static T apply(const T& a, const T& b) {
    return a && b;
  }
};

struct OpCudaLogicOr {
  template <typename T>
  __device__ static T apply(const T& a, const T& b) {
    return a || b;
  }
};

struct OpCudaClampMin {
  template <typename T>
  __device__ static T apply(const T& a, const T& minVal) {
    return a < minVal ? minVal : a;
  }
};

struct OpCudaClampMax {
  template <typename T>
  __device__ static T apply(const T& a, const T& maxVal) {
    return a > maxVal ? maxVal : a;
  }
};

struct OpCudaClamp {
  template <typename T>
  __device__ static T apply(const T& a, const T& minVal, const T& maxVal) {
    if (a < minVal) return minVal;
    if (a > maxVal) return maxVal;
    return a;
  }
};

template <typename T, typename OP>
__global__ void unaryOpKernel(T* out, const T* self, int64_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    out[index] = OP::template apply<T>(self[index]);
  }
}

template <typename T, typename OP>
__global__ void unaryOpInplaceKernel(T* self, int64_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    self[index] = OP::template apply<T>(self[index]);
  }
}

template <typename T, typename OP>
__global__ void unaryOpBackwardKernel(T* out, const T* self, const T* grad, int64_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    out[index] = OP::template apply<T>(self[index], grad[index]);
  }
}

template <typename T, typename OP>
Tensor unaryOpCudaImpl(const Tensor& self) {
  Tensor out(self.shape(), self.options().noGrad());
  T* outPtr = out.dataPtr<T>();
  const T* selfPtr = self.dataPtr<T>();
  int64_t n = self.numel();

  auto params = cuda::getKernelLaunchParams(self.device().index, n);
  CUDA_LAUNCH_KERNEL((unaryOpKernel<T, OP>), params, outPtr, selfPtr, n);
  return out;
}

template <typename T, typename OP>
void unaryOpOutCudaImpl(Tensor& out, const Tensor& self) {
  ASSERT(self.shape() == out.shape());
  T* outPtr = out.dataPtr<T>();
  const T* selfPtr = self.dataPtr<T>();
  int64_t n = self.numel();

  auto params = cuda::getKernelLaunchParams(self.device().index, n);
  CUDA_LAUNCH_KERNEL((unaryOpKernel<T, OP>), params, outPtr, selfPtr, n);
}

template <typename T, typename OP>
void unaryOpInplaceCudaImpl(Tensor& self) {
  self.copyOnWrite();
  T* selfPtr = self.dataPtr<T>();
  int64_t n = self.numel();

  auto params = cuda::getKernelLaunchParams(self.device().index, n);
  CUDA_LAUNCH_KERNEL((unaryOpInplaceKernel<T, OP>), params, selfPtr, n);
}

template <typename T, typename OP>
Tensor unaryOpBackwardCudaImpl(const Tensor& grad, const Tensor& self) {
  ASSERT(self.shape() == grad.shape());
  Tensor out(self.shape(), self.options().noGrad());
  T* outPtr = out.dataPtr<T>();
  const T* selfPtr = self.dataPtr<T>();
  const T* gradPtr = grad.dataPtr<T>();
  int64_t n = self.numel();

  auto params = cuda::getKernelLaunchParams(self.device().index, n);
  CUDA_LAUNCH_KERNEL((unaryOpBackwardKernel<T, OP>), params, outPtr, selfPtr, gradPtr, n);
  return out;
}

template <typename T, typename OP>
Tensor binaryOpAlphaCudaImpl(const Tensor& self, const Tensor& other, const Scalar& alpha = 1) {
  TensorIteratorCuda iterator(self, other);
  auto outShape = iterator.setupBroadcast();
  ASSERT(iterator.isBroadcastOk());
  Tensor out(outShape, self.options().noGrad());
  T alphaVal = alpha.to<T>();
  iterator.template forEach<T>(
      out, [alphaVal] __device__(const T& a, const T& b) -> T { return OP::template apply<T>(a, b, alphaVal); });
  return out;
}

template <typename T, typename OP>
void binaryOpAlphaOutCudaImpl(Tensor& out, const Tensor& self, const Tensor& other, const Scalar& alpha = 1) {
  TensorIteratorCuda iterator(self, other);
  auto outShape = iterator.setupBroadcast();
  ASSERT(iterator.isBroadcastOk());
  ASSERT(outShape == out.shape());
  out.copyOnWrite();
  T alphaVal = alpha.to<T>();
  iterator.template forEach<T>(
      out, [alphaVal] __device__(const T& a, const T& b) -> T { return OP::template apply<T>(a, b, alphaVal); });
}

template <typename T, typename OP>
void binaryOpAlphaInplaceCudaImpl(Tensor& self, const Tensor& other, const Scalar& alpha = 1) {
  TensorIteratorCuda iterator(self, other);
  auto outShape = iterator.setupBroadcast();
  ASSERT(iterator.isBroadcastOk());
  ASSERT(outShape == self.shape());
  self.copyOnWrite();
  T alphaVal = alpha.to<T>();
  iterator.template forEach<T>(
      self, [alphaVal] __device__(const T& a, const T& b) -> T { return OP::template apply<T>(a, b, alphaVal); });
}

template <typename T, typename OP>
Tensor binaryOpCudaImpl(const Tensor& self, const Tensor& other) {
  TensorIteratorCuda iterator(self, other);
  auto outShape = iterator.setupBroadcast();
  ASSERT(iterator.isBroadcastOk());
  Tensor out(outShape, self.options().noGrad());
  iterator.template forEach<T>(out, [] __device__(const T& a, const T& b) -> T { return OP::template apply<T>(a, b); });
  return out;
}

template <typename T, typename OP>
void binaryOpOutCudaImpl(Tensor& out, const Tensor& self, const Tensor& other) {
  TensorIteratorCuda iterator(self, other);
  auto outShape = iterator.setupBroadcast();
  ASSERT(iterator.isBroadcastOk());
  ASSERT(outShape == out.shape());
  out.copyOnWrite();
  iterator.template forEach<T>(out, [] __device__(const T& a, const T& b) -> T { return OP::template apply<T>(a, b); });
}

template <typename T, typename OP>
void binaryOpInplaceCudaImpl(Tensor& self, const Tensor& other) {
  TensorIteratorCuda iterator(self, other);
  auto outShape = iterator.setupBroadcast();
  ASSERT(iterator.isBroadcastOk());
  ASSERT(outShape == self.shape());
  self.copyOnWrite();
  iterator.template forEach<T>(self,
                               [] __device__(const T& a, const T& b) -> T { return OP::template apply<T>(a, b); });
}

template <typename T, typename OP>
Tensor binaryOpBackwardCudaImpl(const Tensor& grad, const Tensor& self, const Tensor& other) {
  TensorIteratorCuda iterator(self, other, grad);
  auto outShape = iterator.setupBroadcast();
  ASSERT(iterator.isBroadcastOk());
  Tensor out(outShape, self.options().noGrad());
  iterator.template forEach<T>(
      out, [] __device__(const T& a, const T& b, const T& g) -> T { return OP::template apply<T>(a, b, g); });
  return out;
}

template <typename T, typename OP>
Tensor binaryOpCompareCudaImpl(const Tensor& self, const Tensor& other) {
  TensorIteratorCuda iterator(self, other);
  auto outShape = iterator.setupBroadcast();
  ASSERT(iterator.isBroadcastOk());
  Options options = self.options();
  options.dtype(DType::Bool);
  options.requiresGrad(false);
  Tensor out(outShape, options);
  iterator.template forEach<DTypeToType_t<DType::Bool>, T>(
      out, [] __device__(const T& a, const T& b) -> bool { return OP::template apply<T>(a, b); });
  return out;
}

template <typename T, typename OP>
void binaryOpCompareOutCudaImpl(Tensor& out, const Tensor& self, const Tensor& other) {
  TensorIteratorCuda iterator(self, other);
  auto outShape = iterator.setupBroadcast();
  ASSERT(iterator.isBroadcastOk());
  ASSERT(outShape == out.shape());
  ASSERT(out.dtype() == DType::Bool);
  ASSERT(out.requiresGrad() == false);
  out.copyOnWrite();
  iterator.template forEach<DTypeToType_t<DType::Bool>, T>(
      out, [] __device__(const T& a, const T& b) -> bool { return OP::template apply<T>(a, b); });
}

template <typename T, typename OP>
Tensor ternaryOpCudaImpl(const Tensor& self, const Tensor& p1, const Tensor& p2) {
  TensorIteratorCuda iterator(self, p1, p2);
  auto outShape = iterator.setupBroadcast();
  ASSERT(iterator.isBroadcastOk());
  Tensor out(outShape, self.options().noGrad());
  iterator.template forEach<T>(
      out, [] __device__(const T& a, const T& b, const T& c) -> T { return OP::template apply<T>(a, b, c); });
  return out;
}

template <typename T, typename OP>
void ternaryOpOutCudaImpl(Tensor& out, const Tensor& self, const Tensor& p1, const Tensor& p2) {
  TensorIteratorCuda iterator(self, p1, p2);
  auto outShape = iterator.setupBroadcast();
  ASSERT(iterator.isBroadcastOk());
  ASSERT(outShape == out.shape());
  out.copyOnWrite();
  iterator.template forEach<T>(
      out, [] __device__(const T& a, const T& b, const T& c) -> T { return OP::template apply<T>(a, b, c); });
}

template <typename T, typename OP>
void ternaryOpInplaceCudaImpl(Tensor& self, const Tensor& p1, const Tensor& p2) {
  TensorIteratorCuda iterator(self, p1, p2);
  auto outShape = iterator.setupBroadcast();
  ASSERT(iterator.isBroadcastOk());
  ASSERT(outShape == self.shape());
  self.copyOnWrite();
  iterator.template forEach<T>(
      self, [] __device__(const T& a, const T& b, const T& c) -> T { return OP::template apply<T>(a, b, c); });
}

template <typename T>
Tensor addcmulOpCudaImpl(const Tensor& self, const Tensor& t1, const Tensor& t2, const Scalar& value) {
  TensorIteratorCuda iterator(self, t1, t2);
  auto outShape = iterator.setupBroadcast();
  ASSERT(iterator.isBroadcastOk());
  Tensor out(outShape, self.options().noGrad());
  T val = value.to<T>();
  iterator.template forEach<T>(out,
                               [val] __device__(const T& a, const T& b, const T& c) -> T { return a + val * b * c; });
  return out;
}

template <typename T>
void addcmulOpOutCudaImpl(Tensor& out, const Tensor& self, const Tensor& t1, const Tensor& t2, const Scalar& value) {
  TensorIteratorCuda iterator(self, t1, t2);
  auto outShape = iterator.setupBroadcast();
  ASSERT(iterator.isBroadcastOk());
  ASSERT(outShape == out.shape());
  out.copyOnWrite();
  T val = value.to<T>();
  iterator.template forEach<T>(out,
                               [val] __device__(const T& a, const T& b, const T& c) -> T { return a + val * b * c; });
}

template <typename T>
void addcmulOpInplaceCudaImpl(Tensor& self, const Tensor& t1, const Tensor& t2, const Scalar& value) {
  TensorIteratorCuda iterator(self, t1, t2);
  auto outShape = iterator.setupBroadcast();
  ASSERT(iterator.isBroadcastOk());
  ASSERT(outShape == self.shape());
  self.copyOnWrite();
  T val = value.to<T>();
  iterator.template forEach<T>(self,
                               [val] __device__(const T& a, const T& b, const T& c) -> T { return a + val * b * c; });
}

}  // namespace tinytorch::op
