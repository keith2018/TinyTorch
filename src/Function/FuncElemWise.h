/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Operations.h"
#include "Tensor/Function.h"

namespace tinytorch::function {

static Tensor reduceGrad(const Tensor& self, const Tensor& grad) {
  auto ndim = std::max(self.dim(), grad.dim());
  SizeVector dims;
  dims.reserve(ndim);

  auto offsetSelf = ndim - self.dim();
  auto offsetGrad = ndim - grad.dim();

  for (auto i = 0; i < ndim; i++) {
    auto s = i < offsetSelf ? 1 : self.shape(i - offsetSelf);
    auto g = i < offsetGrad ? 1 : grad.shape(i - offsetGrad);
    if (s == 1 && g != 1) {
      dims.pushBack(i);
    }
  }

  if (dims.empty()) {
    return grad;
  }

  auto ret = op::sumOnDims(grad, dims, true);
  op::reshapeInplace(ret, self.shape());
  return ret;
}

class FuncSin : public Function<FuncSin> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self) { return op::sin(self); }

  static void backward(AutogradContext* ctx, const Tensor& grad) {
    auto& self = ctx->savedInputs[0];

    if (self.requiresGrad()) {
      self.addGrad(op::sinBackwardP1(grad, self));
    }
  }
};

class FuncCos : public Function<FuncCos> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self) { return op::cos(self); }

  static void backward(AutogradContext* ctx, const Tensor& grad) {
    auto& self = ctx->savedInputs[0];

    if (self.requiresGrad()) {
      self.addGrad(op::cosBackwardP1(grad, self));
    }
  }
};

class FuncSqrt : public Function<FuncSqrt> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self) { return op::sqrt(self); }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
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
      self.addGrad(reduceGrad(self, grad));
    }
    if (other.requiresGrad()) {
      other.addGrad(reduceGrad(other, grad));
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
      self.addGrad(reduceGrad(self, grad));
    }
    if (other.requiresGrad()) {
      other.addGrad(reduceGrad(other, op::neg(grad)));
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
      self.addGrad(reduceGrad(self, op::mul(grad, other)));
    }
    if (other.requiresGrad()) {
      other.addGrad(reduceGrad(other, op::mul(grad, self)));
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
      self.addGrad(reduceGrad(self, op::div(grad, other)));
    }
    if (other.requiresGrad()) {
      other.addGrad(reduceGrad(other, op::divBackwardP2(grad, self, other)));
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
      self.addGrad(reduceGrad(self, op::powBackwardP1(grad, self, other)));
    }
    if (other.requiresGrad()) {
      other.addGrad(reduceGrad(other, op::powBackwardP2(grad, self, other)));
    }
  }
};

class FuncMaximum : public Function<FuncMaximum> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, const Tensor& other) {
    return op::maximum(self, other);
  }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncMinimum : public Function<FuncMinimum> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, const Tensor& other) {
    return op::minimum(self, other);
  }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

#define DEFINE_COMPARE_FUNCTION(CLASSNAME, OPNAME)                                         \
  class CLASSNAME : public Function<CLASSNAME> {                                           \
   public:                                                                                 \
    static Tensor forward(AutogradContext* ctx, const Tensor& self, const Tensor& other) { \
      return op::OPNAME(self, other);                                                      \
    }                                                                                      \
    static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }  \
  }

DEFINE_COMPARE_FUNCTION(FuncLt, lt);
DEFINE_COMPARE_FUNCTION(FuncLe, le);
DEFINE_COMPARE_FUNCTION(FuncGt, gt);
DEFINE_COMPARE_FUNCTION(FuncGe, ge);
DEFINE_COMPARE_FUNCTION(FuncEq, eq);
DEFINE_COMPARE_FUNCTION(FuncNe, ne);

class FuncLogicNot : public Function<FuncLogicNot> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self) { return op::logicNot(self); }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncLogicAnd : public Function<FuncLogicAnd> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, const Tensor& other) {
    return op::logicAnd(self, other);
  }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncLogicOr : public Function<FuncLogicOr> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, const Tensor& other) {
    return op::logicOr(self, other);
  }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
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
inline Tensor sqrt(const Tensor& self) { return FuncSqrt::apply(self); }
inline Tensor pow(const Tensor& self, const Tensor& other) { return FuncPow::apply(self, other); }

inline Tensor maximum(const Tensor& self, const Tensor& other) { return FuncMaximum::apply(self, other); }
inline Tensor minimum(const Tensor& self, const Tensor& other) { return FuncMinimum::apply(self, other); }

inline Tensor lt(const Tensor& self, const Tensor& other) { return FuncLt::apply(self, other); }
inline Tensor le(const Tensor& self, const Tensor& other) { return FuncLe::apply(self, other); }
inline Tensor gt(const Tensor& self, const Tensor& other) { return FuncGt::apply(self, other); }
inline Tensor ge(const Tensor& self, const Tensor& other) { return FuncGe::apply(self, other); }
inline Tensor eq(const Tensor& self, const Tensor& other) { return FuncEq::apply(self, other); }
inline Tensor ne(const Tensor& self, const Tensor& other) { return FuncNe::apply(self, other); }

inline Tensor logicNot(const Tensor& self) { return FuncLogicNot::apply(self); }
inline Tensor logicAnd(const Tensor& self, const Tensor& other) { return FuncLogicAnd::apply(self, other); }
inline Tensor logicOr(const Tensor& self, const Tensor& other) { return FuncLogicOr::apply(self, other); }

}  // namespace tinytorch::function
