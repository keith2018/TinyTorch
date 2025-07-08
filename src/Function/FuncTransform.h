/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Operations.h"
#include "Tensor/Function.h"

namespace tinytorch::function {

#define DEFINE_TRANSFORM_FUNCTION(CLASSNAME, FORWARD_SIGNATURE, FORWARD_BODY) \
  class CLASSNAME : public Function<CLASSNAME> {                              \
   public:                                                                    \
    static Tensor forward FORWARD_SIGNATURE { FORWARD_BODY }                  \
    static TensorList backward(AutogradContext* ctx, const Tensor& grad) {    \
      auto& self = ctx->savedInputs[0];                                       \
      TensorList ret;                                                         \
      if (self.requiresGrad()) {                                              \
        ret.push_back(std::move(op::reshape(grad, self.shape())));            \
      }                                                                       \
      return ret;                                                             \
    }                                                                         \
  }

DEFINE_TRANSFORM_FUNCTION(FuncReshape, (AutogradContext * ctx, const Tensor& self, const IntArrayView shape),
                          return op::reshape(self, shape););

DEFINE_TRANSFORM_FUNCTION(FuncPermute, (AutogradContext * ctx, const Tensor& self, const IntArrayView dims),
                          return op::permute(self, dims););

DEFINE_TRANSFORM_FUNCTION(FuncPermuteAll, (AutogradContext * ctx, const Tensor& self), return op::permuteAll(self););

DEFINE_TRANSFORM_FUNCTION(FuncFlatten,
                          (AutogradContext * ctx, const Tensor& self, int64_t startDim = 0, int64_t endDim = -1),
                          return op::flatten(self, startDim, endDim););

DEFINE_TRANSFORM_FUNCTION(FuncUnflatten,
                          (AutogradContext * ctx, const Tensor& self, int64_t dim, const IntArrayView shape),
                          return op::unflatten(self, dim, shape););

DEFINE_TRANSFORM_FUNCTION(FuncSqueeze, (AutogradContext * ctx, const Tensor& self, IntArrayView dims),
                          return op::squeeze(self, dims););

DEFINE_TRANSFORM_FUNCTION(FuncUnsqueeze, (AutogradContext * ctx, const Tensor& self, int64_t dim),
                          return op::unsqueeze(self, dim););

inline Tensor reshape(const Tensor& self, const IntArrayView shape) { return FuncReshape::apply(self, shape); }
inline Tensor permute(const Tensor& self, const IntArrayView dims) { return FuncPermute::apply(self, dims); }
inline Tensor permute(const Tensor& self) { return FuncPermuteAll::apply(self); }
inline Tensor flatten(const Tensor& self, int64_t startDim = 0, int64_t endDim = -1) {
  return FuncFlatten::apply(self, startDim, endDim);
}
inline Tensor unflatten(const Tensor& self, int64_t dim, const IntArrayView shape) {
  return FuncUnflatten::apply(self, dim, shape);
}
inline Tensor squeeze(const Tensor& self, int64_t dim = -1) { return FuncSqueeze::apply(self, IntArrayView{dim}); }
inline Tensor squeeze(const Tensor& self, const IntArrayView dims) { return FuncSqueeze::apply(self, dims); }
inline Tensor unsqueeze(const Tensor& self, int64_t dim) { return FuncUnsqueeze::apply(self, dim); }

}  // namespace tinytorch::function
