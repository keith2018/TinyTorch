/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Operations.h"
#include "Tensor/Function.h"

namespace tinytorch::function {

class FuncRelu : public Function<FuncRelu> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self) {
    return op::clampMin(self, Tensor::scalar(0, self.options().noGrad()));
  }

  static void backward(AutogradContext* ctx, const Tensor& grad) {
    auto& self = ctx->savedInputs[0];

    if (self.requiresGrad()) {
      auto selfGrad = op::fillMasked(grad, self < 0, 0);
      self.addGrad(std::move(selfGrad));
    }
  }
};

class FuncGelu : public Function<FuncGelu> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self) { return op::gelu(self); }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncSilu : public Function<FuncSilu> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self) { return op::silu(self); }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncSoftmax : public Function<FuncSoftmax> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, int64_t dim) {
    auto output = op::softmax(self, dim);
    if (ctx) {
      ctx->pushData(dim);
      ctx->pushData(output);
    }
    return output;
  }

  static void backward(AutogradContext* ctx, const Tensor& grad) {
    auto& self = ctx->savedInputs[0];
    auto dim = ctx->popData().toInt64();
    auto output = ctx->popData().toTensor();

    if (self.requiresGrad()) {
      auto selfGrad = op::softmaxBackward(grad, output, dim);
      self.addGrad(std::move(selfGrad));
    }
  }
};

class FuncLogSoftmax : public Function<FuncLogSoftmax> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, int64_t dim) {
    auto output = op::logSoftmax(self, dim);
    if (ctx) {
      ctx->pushData(dim);
      ctx->pushData(output);
    }
    return output;
  }

  static void backward(AutogradContext* ctx, const Tensor& grad) {
    auto& self = ctx->savedInputs[0];
    auto dim = ctx->popData().toInt64();
    auto output = ctx->popData().toTensor();

    if (self.requiresGrad()) {
      auto selfGrad = op::logSoftmaxBackward(grad, output, dim);
      self.addGrad(std::move(selfGrad));
    }
  }
};

inline Tensor relu(const Tensor& self) { return FuncRelu::apply(self); }
inline Tensor gelu(const Tensor& self) { return FuncGelu::apply(self); }
inline Tensor silu(const Tensor& self) { return FuncSilu::apply(self); }
inline Tensor softmax(const Tensor& self, int64_t dim = -1) { return FuncSoftmax::apply(self, dim); }
inline Tensor logSoftmax(const Tensor& self, int64_t dim = -1) { return FuncLogSoftmax::apply(self, dim); }

}  // namespace tinytorch::function
