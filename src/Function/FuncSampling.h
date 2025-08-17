/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Operations.h"
#include "Tensor/Function.h"

namespace tinytorch::function {

class FuncTopk : public Function<FuncTopk> {
 public:
  static TensorPair forward(AutogradContext* ctx, const Tensor& self, int64_t k, int64_t dim, bool largest,
                            bool sorted) {
    return op::topk(self, k, dim, largest, sorted);
  }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncMultinomial : public Function<FuncMultinomial> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, int64_t numSamples, bool replacement) {
    return op::multinomial(self, numSamples, replacement);
  }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncSort : public Function<FuncSort> {
 public:
  static TensorPair forward(AutogradContext* ctx, const Tensor& self, int64_t dim, bool descending) {
    return op::sort(self, dim, descending);
  }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncCumsum : public Function<FuncCumsum> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, int64_t dim) { return op::cumsum(self, dim); }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

inline TensorPair topk(const Tensor& self, int64_t k, int64_t dim, bool largest = true, bool sorted = true) {
  return FuncTopk::apply(self, k, dim, largest, sorted);
}
inline Tensor multinomial(const Tensor& self, int64_t numSamples, bool replacement = false) {
  return FuncMultinomial::apply(self, numSamples, replacement);
}
inline TensorPair sort(const Tensor& self, int64_t dim = -1, bool descending = false) {
  return FuncSort::apply(self, dim, descending);
}
inline Tensor cumsum(const Tensor& self, int64_t dim) { return FuncCumsum::apply(self, dim); }

}  // namespace tinytorch::function