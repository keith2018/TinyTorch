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

class FuncTril : public Function<FuncTril> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, int64_t diagonal) { return op::tril(self, diagonal); }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncTriu : public Function<FuncTriu> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, int64_t diagonal) { return op::triu(self, diagonal); }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncSplit : public Function<FuncSplit> {
 public:
  static std::vector<Tensor> forward(AutogradContext* ctx, const Tensor& self, int64_t splitSize, int64_t dim) {
    return op::split(self, splitSize, dim);
  }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncSplitSections : public Function<FuncSplitSections> {
 public:
  static std::vector<Tensor> forward(AutogradContext* ctx, const Tensor& self, IntArrayView sections, int64_t dim) {
    return op::splitSections(self, sections, dim);
  }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncChunk : public Function<FuncChunk> {
 public:
  static std::vector<Tensor> forward(AutogradContext* ctx, const Tensor& self, int64_t chunks, int64_t dim) {
    return op::chunk(self, chunks, dim);
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

class FuncExpand : public Function<FuncExpand> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, IntArrayView sizes) {
    return op::expand(self, sizes);
  }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncIndexSelect : public Function<FuncIndexSelect> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, int64_t dim, const Tensor& index) {
    return op::indexSelect(self, dim, index);
  }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncRepeatInterleave : public Function<FuncRepeatInterleave> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, int64_t repeats, int64_t dim) {
    return op::repeatInterleave(self, repeats, dim);
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
inline Tensor tril(const Tensor& self, int64_t diagonal = 0) { return FuncTril::apply(self, diagonal); }
inline Tensor triu(const Tensor& self, int64_t diagonal = 0) { return FuncTriu::apply(self, diagonal); }
inline std::vector<Tensor> split(const Tensor& self, int64_t splitSize, int64_t dim = 0) {
  return FuncSplit::apply(self, splitSize, dim);
}
inline std::vector<Tensor> split(const Tensor& self, IntArrayView sections, int64_t dim = 0) {
  return FuncSplitSections::apply(self, sections, dim);
}
inline std::vector<Tensor> chunk(const Tensor& self, int64_t chunks, int64_t dim = 0) {
  return FuncChunk::apply(self, chunks, dim);
}
inline Tensor concat(ArrayView<Tensor> tensors, int64_t dim = 0) { return FuncConcat::apply(tensors, dim); }
inline Tensor stack(ArrayView<Tensor> tensors, int64_t dim = 0) { return FuncStack::apply(tensors, dim); }
inline Tensor hstack(ArrayView<Tensor> tensors) { return FuncHStack::apply(tensors); }
inline Tensor vstack(ArrayView<Tensor> tensors) { return FuncVStack::apply(tensors); }
inline Tensor narrow(const Tensor& self, int64_t dim, int64_t start, int64_t length) {
  return FuncNarrow::apply(self, dim, start, length);
}
inline Tensor gather(const Tensor& self, int64_t dim, const Tensor& index) {
  return FuncGather::apply(self, dim, index);
}
inline Tensor scatter(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
  return FuncScatter::apply(self, dim, index, src);
}
inline void scatter_(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
  return FuncScatterInplace::apply(self, dim, index, src);
}
inline Tensor expand(const Tensor& self, IntArrayView sizes) { return FuncExpand::apply(self, sizes); }
inline Tensor indexSelect(const Tensor& self, int64_t dim, const Tensor& index) {
  return FuncIndexSelect::apply(self, dim, index);
}
inline Tensor repeatInterleave(const Tensor& self, int64_t repeats, int64_t dim) {
  return FuncRepeatInterleave::apply(self, repeats, dim);
}

}  // namespace tinytorch::function
