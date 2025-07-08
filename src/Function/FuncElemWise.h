/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Operations.h"
#include "Tensor/Function.h"

namespace tinytorch::function {

Tensor reduceGrad(const Tensor& self, const Tensor& grad);

class FuncSin : public Function<FuncSin> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self) { return op::sin(self); }

  static TensorList backward(AutogradContext* ctx, const Tensor& grad) {
    auto& self = ctx->savedInputs[0];

    TensorList ret;
    if (self.requiresGrad()) {
      ret.push_back(std::move(op::sinBackwardP1(grad, self)));
    }
    return ret;
  }
};

class FuncCos : public Function<FuncCos> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self) { return op::cos(self); }

  static TensorList backward(AutogradContext* ctx, const Tensor& grad) {
    auto& self = ctx->savedInputs[0];

    TensorList ret;
    if (self.requiresGrad()) {
      ret.push_back(std::move(op::cosBackwardP1(grad, self)));
    }
    return ret;
  }
};

class FuncAdd : public Function<FuncAdd> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, const Tensor& other, const Scalar& alpha) {
    return op::add(self, other, alpha);
  }

  static TensorList backward(AutogradContext* ctx, const Tensor& grad) {
    auto& self = ctx->savedInputs[0];
    auto& other = ctx->savedInputs[1];

    TensorList ret;
    if (self.requiresGrad()) {
      ret.push_back(std::move(reduceGrad(self, grad)));
    }
    if (other.requiresGrad()) {
      ret.push_back(std::move(reduceGrad(other, grad)));
    }
    return ret;
  }
};

class FuncSub : public Function<FuncSub> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, const Tensor& other, const Scalar& alpha) {
    return op::sub(self, other, alpha);
  }

  static TensorList backward(AutogradContext* ctx, const Tensor& grad) {
    auto& self = ctx->savedInputs[0];
    auto& other = ctx->savedInputs[1];

    TensorList ret;
    if (self.requiresGrad()) {
      ret.push_back(std::move(reduceGrad(self, grad)));
    }
    if (other.requiresGrad()) {
      ret.push_back(std::move(reduceGrad(other, op::neg(grad))));
    }
    return ret;
  }
};

class FuncMul : public Function<FuncMul> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, const Tensor& other) { return op::mul(self, other); }

  static TensorList backward(AutogradContext* ctx, const Tensor& grad) {
    auto& self = ctx->savedInputs[0];
    auto& other = ctx->savedInputs[1];

    TensorList ret;
    if (self.requiresGrad()) {
      ret.push_back(std::move(reduceGrad(self, op::mul(grad, other))));
    }
    if (other.requiresGrad()) {
      ret.push_back(std::move(reduceGrad(other, op::mul(grad, self))));
    }
    return ret;
  }
};

class FuncDiv : public Function<FuncDiv> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, const Tensor& other) { return op::div(self, other); }

  static TensorList backward(AutogradContext* ctx, const Tensor& grad) {
    auto& self = ctx->savedInputs[0];
    auto& other = ctx->savedInputs[1];

    TensorList ret;
    if (self.requiresGrad()) {
      ret.push_back(std::move(reduceGrad(self, op::div(grad, other))));
    }
    if (other.requiresGrad()) {
      ret.push_back(std::move(reduceGrad(other, op::divBackwardP2(grad, self, other))));
    }
    return ret;
  }
};

class FuncPow : public Function<FuncPow> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, const Tensor& other) { return op::pow(self, other); }

  static TensorList backward(AutogradContext* ctx, const Tensor& grad) {
    auto& self = ctx->savedInputs[0];
    auto& other = ctx->savedInputs[1];

    TensorList ret;
    if (self.requiresGrad()) {
      ret.push_back(std::move(reduceGrad(self, op::powBackwardP1(grad, self, other))));
    }
    if (other.requiresGrad()) {
      ret.push_back(std::move(reduceGrad(other, op::powBackwardP2(grad, self, other))));
    }
    return ret;
  }
};

class FuncMaximum : public Function<FuncMaximum> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, const Tensor& other) {
    return op::maximum(self, other);
  }

  static TensorList backward(AutogradContext* ctx, const Tensor& grad) {
    auto& self = ctx->savedInputs[0];
    auto& other = ctx->savedInputs[1];

    TensorList ret;
    // TODO
    NOT_IMPLEMENTED();
    return ret;
  }
};

class FuncMinimum : public Function<FuncMinimum> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, const Tensor& other) {
    return op::minimum(self, other);
  }

  static TensorList backward(AutogradContext* ctx, const Tensor& grad) {
    auto& self = ctx->savedInputs[0];
    auto& other = ctx->savedInputs[1];

    TensorList ret;
    // TODO
    NOT_IMPLEMENTED();
    return ret;
  }
};

inline Tensor add(const Tensor& self, const Tensor& other, const Scalar& alpha = 1) {
  return FuncAdd::apply(self, other, alpha);
}
inline Tensor sub(const Tensor& self, const Tensor& other, const Scalar& alpha = 1) {
  return FuncSub::apply(self, other, alpha);
}
inline Tensor mul(const Tensor& self, const Tensor& other) { return FuncMul::apply(self, other); }
inline Tensor div(const Tensor& self, const Tensor& other) { return FuncDiv::apply(self, other); }

inline Tensor sin(const Tensor& self) { return FuncSin::apply(self); }
inline Tensor cos(const Tensor& self) { return FuncCos::apply(self); }
inline Tensor pow(const Tensor& self, const Tensor& other) { return FuncPow::apply(self, other); }

inline Tensor maximum(const Tensor& self, const Tensor& other) { return FuncMaximum::apply(self, other); }
inline Tensor minimum(const Tensor& self, const Tensor& other) { return FuncMinimum::apply(self, other); }

}  // namespace tinytorch::function
