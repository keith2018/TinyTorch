/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Operations.h"
#include "Tensor/Function.h"

namespace tinytorch::function {

class FuncSiluMul : public Function<FuncSiluMul> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& x) { return op::siluMul(x); }

  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

inline Tensor siluMul(const Tensor& x) { return FuncSiluMul::apply(x); }

}  // namespace tinytorch::function
