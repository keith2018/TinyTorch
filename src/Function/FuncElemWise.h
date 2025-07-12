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

  static void backward(AutogradContext* ctx, const Tensor& grad) {
    auto& self = ctx->savedInputs[0];

    if (self.requiresGrad()) {
      self.addGrad(std::move(op::sinBackwardP1(grad, self)));
    }
  }
};

class FuncCos : public Function<FuncCos> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self) { return op::cos(self); }

  static void backward(AutogradContext* ctx, const Tensor& grad) {
    auto& self = ctx->savedInputs[0];

    if (self.requiresGrad()) {
      self.addGrad(std::move(op::cosBackwardP1(grad, self)));
    }
  }
};

class FuncAdd : public Function<FuncAdd> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, const Tensor& other, const Scalar& alpha) {
    return op::add(self, other, alpha);
  }

  static void backward(AutogradContext* ctx, const Tensor& grad) {
    auto& self = ctx->savedInputs[0];
    auto& other = ctx->savedInputs[1];

    if (self.requiresGrad()) {
      self.addGrad(std::move(reduceGrad(self, grad)));
    }
    if (other.requiresGrad()) {
      other.addGrad(std::move(reduceGrad(other, grad)));
    }
  }
};

class FuncSub : public Function<FuncSub> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, const Tensor& other, const Scalar& alpha) {
    return op::sub(self, other, alpha);
  }

  static void backward(AutogradContext* ctx, const Tensor& grad) {
    auto& self = ctx->savedInputs[0];
    auto& other = ctx->savedInputs[1];

    if (self.requiresGrad()) {
      self.addGrad(std::move(reduceGrad(self, grad)));
    }
    if (other.requiresGrad()) {
      other.addGrad(std::move(reduceGrad(other, op::neg(grad))));
    }
  }
};

class FuncMul : public Function<FuncMul> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, const Tensor& other) { return op::mul(self, other); }

  static void backward(AutogradContext* ctx, const Tensor& grad) {
    auto& self = ctx->savedInputs[0];
    auto& other = ctx->savedInputs[1];

    if (self.requiresGrad()) {
      self.addGrad(std::move(reduceGrad(self, op::mul(grad, other))));
    }
    if (other.requiresGrad()) {
      other.addGrad(std::move(reduceGrad(other, op::mul(grad, self))));
    }
  }
};

class FuncDiv : public Function<FuncDiv> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, const Tensor& other) { return op::div(self, other); }

  static void backward(AutogradContext* ctx, const Tensor& grad) {
    auto& self = ctx->savedInputs[0];
    auto& other = ctx->savedInputs[1];

    if (self.requiresGrad()) {
      self.addGrad(std::move(reduceGrad(self, op::div(grad, other))));
    }
    if (other.requiresGrad()) {
      other.addGrad(std::move(reduceGrad(other, op::divBackwardP2(grad, self, other))));
    }
  }
};

class FuncPow : public Function<FuncPow> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, const Tensor& other) { return op::pow(self, other); }

  static void backward(AutogradContext* ctx, const Tensor& grad) {
    auto& self = ctx->savedInputs[0];
    auto& other = ctx->savedInputs[1];

    if (self.requiresGrad()) {
      self.addGrad(std::move(reduceGrad(self, op::powBackwardP1(grad, self, other))));
    }
    if (other.requiresGrad()) {
      other.addGrad(std::move(reduceGrad(other, op::powBackwardP2(grad, self, other))));
    }
  }
};

class FuncMaximum : public Function<FuncMaximum> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, const Tensor& other) {
    return op::maximum(self, other);
  }

  static void backward(AutogradContext* ctx, const Tensor& grad) {
    // TODO
    NOT_IMPLEMENTED();
  }
};

class FuncMinimum : public Function<FuncMinimum> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, const Tensor& other) {
    return op::minimum(self, other);
  }

  static void backward(AutogradContext* ctx, const Tensor& grad) {
    // TODO
    NOT_IMPLEMENTED();
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
