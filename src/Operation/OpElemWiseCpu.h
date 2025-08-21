/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <cmath>

#include "OpElemWise.h"
#include "Tensor/TensorIterator.h"

namespace tinytorch::op {

struct OpCpuAbs {
  template <typename T>
  static T apply(const T& a) {
    return std::abs(a);
  }
};

struct OpCpuNeg {
  template <typename T>
  static T apply(const T& a) {
    return -a;
  }
};

struct OpCpuSign {
  template <typename T>
  static T apply(const T& a) {
    return a > 0 ? 1 : -1;
  }
};

struct OpCpuSqrt {
  template <typename T>
  static T apply(const T& a) {
    return std::sqrt(a);
  }
};

struct OpCpuSquare {
  template <typename T>
  static T apply(const T& a) {
    return a * a;
  }
};

struct OpCpuExp {
  template <typename T>
  static T apply(const T& a) {
    return std::exp(a);
  }
};

struct OpCpuLog {
  template <typename T>
  static T apply(const T& a) {
    return std::log(a);
  }
};

struct OpCpuSin {
  template <typename T>
  static T apply(const T& a) {
    return std::sin(a);
  }
};

struct OpCpuSinBackwardP1 {
  template <typename T>
  static T apply(const T& self, const T& grad) {
    return std::cos(self) * grad;
  }
};

struct OpCpuCos {
  template <typename T>
  static T apply(const T& a) {
    return std::cos(a);
  }
};

struct OpCpuCosBackwardP1 {
  template <typename T>
  static T apply(const T& self, const T& grad) {
    return -std::sin(self) * grad;
  }
};

struct OpCpuSigmoid {
  template <typename T>
  static T apply(const T& a) {
    return 1 / (1 + std::exp(-a));
  }
};

struct OpCpuTanh {
  template <typename T>
  static T apply(const T& a) {
    return std::tanh(a);
  }
};

struct OpCpuRelu {
  template <typename T>
  static T apply(const T& a) {
    return a > 0 ? a : T(0);
  }
};

struct OpCpuGelu {
  template <typename T>
  static T apply(const T& a) {
    constexpr float sqrt2OverPi = 0.7978845608f;  // sqrt(2/pi)
    float tanhArg = sqrt2OverPi * (a + 0.044715f * a * a * a);
    return 0.5f * a * (1.0f + std::tanh(tanhArg));
  }
};

struct OpCpuSilu {
  template <typename T>
  static T apply(const T& a) {
    return a / (1.0f + std::exp(-a));
  }
};

struct OpCpuAdd {
  template <typename T>
  static T apply(const T& a, const T& b, const T& alpha) {
    return a + alpha * b;
  }

  template <typename T>
  static T apply(const T& a, const T& b) {
    return a + b;
  }
};

struct OpCpuSub {
  template <typename T>
  static T apply(const T& a, const T& b, const T& alpha) {
    return a - alpha * b;
  }

  template <typename T>
  static T apply(const T& a, const T& b) {
    return a - b;
  }
};

struct OpCpuMul {
  template <typename T>
  static T apply(const T& a, const T& b) {
    return a * b;
  }
};

struct OpCpuDiv {
  template <typename T>
  static T apply(const T& a, const T& b) {
    return a / b;
  }
};

struct OpCpuDivBackwardP2 {
  template <typename T>
  static T apply(const T& self, const T& other, const T& grad) {
    return grad * (-self) / (other * other);
  }
};

struct OpCpuPow {
  template <typename T>
  static T apply(const T& a, const T& b) {
    return std::pow(a, b);
  }
};

struct OpCpuPowBackwardP1 {
  template <typename T>
  static T apply(const T& a, const T& b, const T& grad) {
    return grad * b * std::pow(a, b - 1);
  }
};

struct OpCpuPowBackwardP2 {
  template <typename T>
  static T apply(const T& a, const T& b, const T& grad) {
    return grad * std::pow(a, b) * std::log(a);
  }
};

struct OpCpuMaximum {
  template <typename T>
  static T apply(const T& a, const T& b) {
    return std::max(a, b);
  }
};

struct OpCpuMinimum {
  template <typename T>
  static T apply(const T& a, const T& b) {
    return std::min(a, b);
  }
};

struct OpCpuEq {
  template <typename T>
  static bool apply(const T& a, const T& b) {
    return a == b;
  }
};

struct OpCpuNe {
  template <typename T>
  static bool apply(const T& a, const T& b) {
    return a != b;
  }
};

struct OpCpuLt {
  template <typename T>
  static bool apply(const T& a, const T& b) {
    return a < b;
  }
};

struct OpCpuLe {
  template <typename T>
  static bool apply(const T& a, const T& b) {
    return a <= b;
  }
};

struct OpCpuGt {
  template <typename T>
  static bool apply(const T& a, const T& b) {
    return a > b;
  }
};

struct OpCpuGe {
  template <typename T>
  static bool apply(const T& a, const T& b) {
    return a >= b;
  }
};

struct OpCpuLogicNot {
  template <typename T>
  static T apply(const T& a) {
    return !a;
  }
};

struct OpCpuLogicAnd {
  template <typename T>
  static T apply(const T& a, const T& b) {
    return a && b;
  }
};

struct OpCpuLogicOr {
  template <typename T>
  static T apply(const T& a, const T& b) {
    return a || b;
  }
};

struct OpCpuClampMin {
  template <typename T>
  static T apply(const T& a, const T& minVal) {
    return a < minVal ? minVal : a;
  }
};

struct OpCpuClampMax {
  template <typename T>
  static T apply(const T& a, const T& maxVal) {
    return a > maxVal ? maxVal : a;
  }
};

struct OpCpuClamp {
  template <typename T>
  static T apply(const T& a, const T& minVal, const T& maxVal) {
    if (a < minVal) return minVal;
    if (a > maxVal) return maxVal;
    return a;
  }
};

template <typename T, typename OP>
Tensor unaryOpCpuImpl(const Tensor& self) {
  Tensor out(self.shape(), self.options().noGrad());
  T* outPtr = out.dataPtr<T>();
  const T* selfPtr = self.dataPtr<T>();
  for (int64_t i = 0; i < self.numel(); i++) {
    outPtr[i] = OP::template apply<T>(selfPtr[i]);
  }
  return out;
}

template <typename T, typename OP>
void unaryOpOutCpuImpl(Tensor& out, const Tensor& self) {
  ASSERT(self.shape() == out.shape());
  T* outPtr = out.dataPtr<T>();
  const T* selfPtr = self.dataPtr<T>();
  for (int64_t i = 0; i < self.numel(); i++) {
    outPtr[i] = OP::template apply<T>(selfPtr[i]);
  }
}

template <typename T, typename OP>
void unaryOpInplaceCpuImpl(Tensor& self) {
  self.copyOnWrite();
  T* selfPtr = self.dataPtr<T>();
  for (int64_t i = 0; i < self.numel(); i++) {
    selfPtr[i] = OP::template apply<T>(selfPtr[i]);
  }
}

template <typename T, typename OP>
Tensor unaryOpBackwardCpuImpl(const Tensor& grad, const Tensor& self) {
  ASSERT(self.shape() == grad.shape());
  Tensor out(self.shape(), self.options().noGrad());
  T* outPtr = out.dataPtr<T>();
  const T* selfPtr = self.dataPtr<T>();
  const T* gradPtr = grad.dataPtr<T>();
  for (int64_t i = 0; i < self.numel(); i++) {
    outPtr[i] = OP::template apply<T>(selfPtr[i], gradPtr[i]);
  }
  return out;
}

template <typename T, typename OP>
Tensor binaryOpAlphaCpuImpl(const Tensor& self, const Tensor& other, const Scalar& alpha = 1) {
  TensorIteratorCpu iterator(self, other);
  auto outShape = iterator.setupBroadcast();
  ASSERT(iterator.isBroadcastOk());
  Tensor out(outShape, self.options().noGrad());
  T alphaVal = alpha.to<T>();
  iterator.template forEach<T>(
      out, [alphaVal](const T& a, const T& b) -> T { return OP::template apply<T>(a, b, alphaVal); });
  return out;
}

template <typename T, typename OP>
void binaryOpAlphaOutCpuImpl(Tensor& out, const Tensor& self, const Tensor& other, const Scalar& alpha = 1) {
  TensorIteratorCpu iterator(self, other);
  auto outShape = iterator.setupBroadcast();
  ASSERT(iterator.isBroadcastOk());
  ASSERT(outShape == out.shape());
  out.copyOnWrite();
  T alphaVal = alpha.to<T>();
  iterator.template forEach<T>(
      out, [alphaVal](const T& a, const T& b) -> T { return OP::template apply<T>(a, b, alphaVal); });
}

template <typename T, typename OP>
void binaryOpAlphaInplaceCpuImpl(Tensor& self, const Tensor& other, const Scalar& alpha = 1) {
  TensorIteratorCpu iterator(self, other);
  auto outShape = iterator.setupBroadcast();
  ASSERT(iterator.isBroadcastOk());
  ASSERT(outShape == self.shape());
  self.copyOnWrite();
  T alphaVal = alpha.to<T>();
  iterator.template forEach<T>(
      self, [alphaVal](const T& a, const T& b) -> T { return OP::template apply<T>(a, b, alphaVal); });
}

template <typename T, typename OP>
Tensor binaryOpCpuImpl(const Tensor& self, const Tensor& other) {
  TensorIteratorCpu iterator(self, other);
  auto outShape = iterator.setupBroadcast();
  ASSERT(iterator.isBroadcastOk());
  Tensor out(outShape, self.options().noGrad());
  iterator.template forEach<T>(out, [](const T& a, const T& b) -> T { return OP::template apply<T>(a, b); });
  return out;
}

template <typename T, typename OP>
void binaryOpOutCpuImpl(Tensor& out, const Tensor& self, const Tensor& other) {
  TensorIteratorCpu iterator(self, other);
  auto outShape = iterator.setupBroadcast();
  ASSERT(iterator.isBroadcastOk());
  ASSERT(outShape == out.shape());
  out.copyOnWrite();
  iterator.template forEach<T>(out, [](const T& a, const T& b) -> T { return OP::template apply<T>(a, b); });
}

template <typename T, typename OP>
void binaryOpInplaceCpuImpl(Tensor& self, const Tensor& other) {
  TensorIteratorCpu iterator(self, other);
  auto outShape = iterator.setupBroadcast();
  ASSERT(iterator.isBroadcastOk());
  ASSERT(outShape == self.shape());
  self.copyOnWrite();
  iterator.template forEach<T>(self, [](const T& a, const T& b) -> T { return OP::template apply<T>(a, b); });
}

template <typename T, typename OP>
Tensor binaryOpBackwardCpuImpl(const Tensor& grad, const Tensor& self, const Tensor& other) {
  TensorIteratorCpu iterator(self, other, grad);
  auto outShape = iterator.setupBroadcast();
  ASSERT(iterator.isBroadcastOk());
  Tensor out(outShape, self.options().noGrad());
  iterator.template forEach<T>(out,
                               [](const T& a, const T& b, const T& g) -> T { return OP::template apply<T>(a, b, g); });
  return out;
}

template <typename T, typename OP>
Tensor binaryOpCompareCpuImpl(const Tensor& self, const Tensor& other) {
  TensorIteratorCpu iterator(self, other);
  auto outShape = iterator.setupBroadcast();
  ASSERT(iterator.isBroadcastOk());
  Options options = self.options();
  options.dtype(DType::Bool);
  options.requiresGrad(false);
  Tensor out(outShape, options);
  iterator.template forEach<DTypeToType_t<DType::Bool>, T>(
      out, [](const T& a, const T& b) -> bool { return OP::template apply<T>(a, b); });
  return out;
}

template <typename T, typename OP>
void binaryOpCompareOutCpuImpl(Tensor& out, const Tensor& self, const Tensor& other) {
  TensorIteratorCpu iterator(self, other);
  auto outShape = iterator.setupBroadcast();
  ASSERT(iterator.isBroadcastOk());
  ASSERT(outShape == out.shape());
  ASSERT(out.dtype() == DType::Bool);
  ASSERT(out.requiresGrad() == false);
  out.copyOnWrite();
  iterator.template forEach<DTypeToType_t<DType::Bool>, T>(
      out, [](const T& a, const T& b) -> bool { return OP::template apply<T>(a, b); });
}

template <typename T, typename OP>
Tensor ternaryOpCpuImpl(const Tensor& self, const Tensor& p1, const Tensor& p2) {
  TensorIteratorCpu iterator(self, p1, p2);
  auto outShape = iterator.setupBroadcast();
  ASSERT(iterator.isBroadcastOk());
  Tensor out(outShape, self.options().noGrad());
  iterator.template forEach<T>(out,
                               [](const T& a, const T& b, const T& c) -> T { return OP::template apply<T>(a, b, c); });
  return out;
}

template <typename T, typename OP>
void ternaryOpOutCpuImpl(Tensor& out, const Tensor& self, const Tensor& p1, const Tensor& p2) {
  TensorIteratorCpu iterator(self, p1, p2);
  auto outShape = iterator.setupBroadcast();
  ASSERT(iterator.isBroadcastOk());
  ASSERT(outShape == out.shape());
  out.copyOnWrite();
  iterator.template forEach<T>(out,
                               [](const T& a, const T& b, const T& c) -> T { return OP::template apply<T>(a, b, c); });
}

template <typename T, typename OP>
void ternaryOpInplaceCpuImpl(Tensor& self, const Tensor& p1, const Tensor& p2) {
  TensorIteratorCpu iterator(self, p1, p2);
  auto outShape = iterator.setupBroadcast();
  ASSERT(iterator.isBroadcastOk());
  ASSERT(outShape == self.shape());
  self.copyOnWrite();
  iterator.template forEach<T>(self,
                               [](const T& a, const T& b, const T& c) -> T { return OP::template apply<T>(a, b, c); });
}

template <typename T>
Tensor addcmulOpCpuImpl(const Tensor& self, const Tensor& t1, const Tensor& t2, const Scalar& value) {
  TensorIteratorCpu iterator(self, t1, t2);
  auto outShape = iterator.setupBroadcast();
  ASSERT(iterator.isBroadcastOk());
  Tensor out(outShape, self.options().noGrad());
  T val = value.to<T>();
  iterator.template forEach<T>(out, [val](const T& a, const T& b, const T& c) -> T { return a + val * b * c; });
  return out;
}

template <typename T>
void addcmulOpOutCpuImpl(Tensor& out, const Tensor& self, const Tensor& t1, const Tensor& t2, const Scalar& value) {
  TensorIteratorCpu iterator(self, t1, t2);
  auto outShape = iterator.setupBroadcast();
  ASSERT(iterator.isBroadcastOk());
  ASSERT(outShape == out.shape());
  out.copyOnWrite();
  T val = value.to<T>();
  iterator.template forEach<T>(out, [val](const T& a, const T& b, const T& c) -> T { return a + val * b * c; });
}

template <typename T>
void addcmulOpInplaceCpuImpl(Tensor& self, const Tensor& t1, const Tensor& t2, const Scalar& value) {
  TensorIteratorCpu iterator(self, t1, t2);
  auto outShape = iterator.setupBroadcast();
  ASSERT(iterator.isBroadcastOk());
  ASSERT(outShape == self.shape());
  self.copyOnWrite();
  T val = value.to<T>();
  iterator.template forEach<T>(self, [val](const T& a, const T& b, const T& c) -> T { return a + val * b * c; });
}

}  // namespace tinytorch::op
