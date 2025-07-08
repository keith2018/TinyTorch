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

  static TensorList backward(AutogradContext* ctx, const Tensor& grad) {
    auto& self = ctx->savedInputs[0];

    TensorList ret;
    if (self.requiresGrad()) {
      auto selfGrad = op::mul(grad, Tensor::onesLike(self, self.options().noGrad()));
      ret.push_back(std::move(selfGrad));
    }
    return ret;
  }
};

class FuncSumOnDim : public Function<FuncSumOnDim> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, int64_t dim, bool keepDim) {
    return op::sumOnDim(self, dim, keepDim);
  }

  static TensorList backward(AutogradContext* ctx, const Tensor& grad) {
    auto& self = ctx->savedInputs[0];

    TensorList ret;
    // TODO
    NOT_IMPLEMENTED();
    return ret;
  }
};

class FuncSumOnDims : public Function<FuncSumOnDims> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, const IntArrayView dims, bool keepDim) {
    return op::sumOnDims(self, dims, keepDim);
  }

  static TensorList backward(AutogradContext* ctx, const Tensor& grad) {
    auto& self = ctx->savedInputs[0];

    TensorList ret;
    // TODO
    NOT_IMPLEMENTED();
    return ret;
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
