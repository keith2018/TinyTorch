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

class FuncSplit : public Function<FuncSplit> {
 public:
  static std::vector<Tensor> forward(AutogradContext* ctx, const Tensor& self, int64_t splitSize, int64_t dim) {
    return op::split(self, splitSize, dim);
  }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncConcat : public Function<FuncConcat> {
 public:
  static Tensor forward(AutogradContext* ctx, ArrayView<Tensor> tensors, int64_t dim) {
    return op::concat(tensors, dim);
  }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncStack : public Function<FuncStack> {
 public:
  static Tensor forward(AutogradContext* ctx, ArrayView<Tensor> tensors, int64_t dim) {
    return op::stack(tensors, dim);
  }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncHStack : public Function<FuncHStack> {
 public:
  static Tensor forward(AutogradContext* ctx, ArrayView<Tensor> tensors) { return op::hstack(tensors); }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncVStack : public Function<FuncVStack> {
 public:
  static Tensor forward(AutogradContext* ctx, ArrayView<Tensor> tensors) { return op::vstack(tensors); }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncNarrow : public Function<FuncNarrow> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, int64_t dim, int64_t start, int64_t length) {
    return op::narrow(self, dim, start, length);
  }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

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

class FuncGather : public Function<FuncGather> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, int64_t dim, const Tensor& index) {
    return op::gather(self, dim, index);
  }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncScatter : public Function<FuncScatter> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
    return op::scatter(self, dim, index, src);
  }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncScatterInplace : public Function<FuncScatterInplace> {
 public:
  static void forward(AutogradContext* ctx, Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
    return op::scatterInplace(self, dim, index, src);
  }
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
inline std::vector<Tensor> split(const Tensor& self, int64_t splitSize, int64_t dim = 0) {
  return FuncSplit::apply(self, splitSize, dim);
}
inline Tensor concat(ArrayView<Tensor> tensors, int64_t dim = 0) { return FuncConcat::apply(tensors, dim); }
inline Tensor stack(ArrayView<Tensor> tensors, int64_t dim = 0) { return FuncStack::apply(tensors, dim); }
inline Tensor hstack(ArrayView<Tensor> tensors) { return FuncHStack::apply(tensors); }
inline Tensor vstack(ArrayView<Tensor> tensors) { return FuncVStack::apply(tensors); }
inline Tensor narrow(const Tensor& self, int64_t dim, int64_t start, int64_t length) {
  return FuncNarrow::apply(self, dim, start, length);
}
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
inline Tensor gather(const Tensor& self, int64_t dim, const Tensor& index) {
  return FuncGather::apply(self, dim, index);
}
inline Tensor scatter(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
  return FuncScatter::apply(self, dim, index, src);
}
inline void scatter_(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
  return FuncScatterInplace::apply(self, dim, index, src);
}

}  // namespace tinytorch::function
