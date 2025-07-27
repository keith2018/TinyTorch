/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Operations.h"
#include "Tensor/Function.h"

namespace tinytorch::function {

class FuncMatmul : public Function<FuncMatmul> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& a, const Tensor& b, bool transA, bool transB) {
    return op::matmul(a, b, transA, transB);
  }

  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

inline Tensor matmul(const Tensor& a, const Tensor& b, bool transA = false, bool transB = false) {
  return FuncMatmul::apply(a, b, transA, transB);
}

}  // namespace tinytorch::function
