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
    static void backward(AutogradContext* ctx, const Tensor& grad) {          \
      auto& self = ctx->savedInputs[0];                                       \
                                                                              \
      if (self.requiresGrad()) {                                              \
        self.addGrad(op::reshape(grad, self.shape()));                        \
      }                                                                       \
    }                                                                         \
  }

DEFINE_TRANSFORM_FUNCTION(FuncReshape, (AutogradContext * ctx, const Tensor& self, const IntArrayView shape),
                          return op::reshape(self, shape););

DEFINE_TRANSFORM_FUNCTION(FuncView, (AutogradContext * ctx, const Tensor& self, const IntArrayView shape),
                          return op::view(self, shape););

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

class FuncTranspose : public Function<FuncTranspose> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, int64_t dim0, int64_t dim1) {
    return op::transpose(self, dim0, dim1);
  }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncTranspose2D : public Function<FuncTranspose2D> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self) { return op::transpose2d(self); }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

inline Tensor reshape(const Tensor& self, const IntArrayView shape) { return FuncReshape::apply(self, shape); }
inline Tensor view(const Tensor& self, const IntArrayView shape) { return FuncView::apply(self, shape); }
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
inline Tensor transpose(const Tensor& self, int64_t dim0, int64_t dim1) {
  return FuncTranspose::apply(self, dim0, dim1);
}
inline Tensor transpose2d(const Tensor& self) { return FuncTranspose2D::apply(self); }

}  // namespace tinytorch::function
