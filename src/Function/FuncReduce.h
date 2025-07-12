/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Operations.h"
#include "Tensor/Function.h"

namespace tinytorch::function {

class FuncSum : public Function<FuncSum> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self) { return op::sum(self); }

  static void backward(AutogradContext* ctx, const Tensor& grad) {
    auto& self = ctx->savedInputs[0];

    if (self.requiresGrad()) {
      auto selfGrad = op::mul(grad, Tensor::onesLike(self, self.options().noGrad()));
      self.addGrad(std::move(selfGrad));
    }
  }
};

class FuncSumOnDim : public Function<FuncSumOnDim> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, int64_t dim, bool keepDim) {
    return op::sumOnDim(self, dim, keepDim);
  }

  static void backward(AutogradContext* ctx, const Tensor& grad) {
    // TODO
    NOT_IMPLEMENTED();
  }
};

class FuncSumOnDims : public Function<FuncSumOnDims> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, const IntArrayView dims, bool keepDim) {
    return op::sumOnDims(self, dims, keepDim);
  }

  static void backward(AutogradContext* ctx, const Tensor& grad) {
    // TODO
    NOT_IMPLEMENTED();
  }
};

inline Tensor sum(const Tensor& self) { return FuncSum::apply(self); }
inline Tensor sum(const Tensor& self, int64_t dim, bool keepDim = false) {
  return FuncSumOnDim::apply(self, dim, keepDim);
}
inline Tensor sum(const Tensor& self, const IntArrayView dims, bool keepDim = false) {
  return FuncSumOnDims::apply(self, dims, keepDim);
}

}  // namespace tinytorch::function
