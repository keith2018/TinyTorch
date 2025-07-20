/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Operations.h"
#include "Tensor/Function.h"

namespace tinytorch::function {

class FuncMin : public Function<FuncMin> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self) { return op::min(self); }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncArgmin : public Function<FuncArgmin> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self) { return op::argmin(self); }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncMinOnDim : public Function<FuncMinOnDim> {
 public:
  static TensorPair forward(AutogradContext* ctx, const Tensor& self, int64_t dim, bool keepDim) {
    return op::minOnDim(self, dim, keepDim);
  }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncMax : public Function<FuncMax> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self) { return op::max(self); }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncArgmax : public Function<FuncArgmax> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self) { return op::argmax(self); }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncMaxOnDim : public Function<FuncMaxOnDim> {
 public:
  static TensorPair forward(AutogradContext* ctx, const Tensor& self, int64_t dim, bool keepDim) {
    return op::maxOnDim(self, dim, keepDim);
  }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

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
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncSumOnDims : public Function<FuncSumOnDims> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, const IntArrayView dims, bool keepDim) {
    return op::sumOnDims(self, dims, keepDim);
  }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncMean : public Function<FuncMean> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self) { return op::mean(self); }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncMeanOnDim : public Function<FuncMeanOnDim> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, int64_t dim, bool keepDim) {
    return op::meanOnDim(self, dim, keepDim);
  }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncMeanOnDims : public Function<FuncMeanOnDims> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, const IntArrayView dims, bool keepDim) {
    return op::meanOnDims(self, dims, keepDim);
  }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncVar : public Function<FuncVar> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, bool unbiased) {
    return op::varMean(self, unbiased).first;
  }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncVarOnDim : public Function<FuncVarOnDim> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, int64_t dim, bool unbiased, bool keepDim) {
    return op::varMeanOnDim(self, dim, unbiased, keepDim).first;
  }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncVarOnDims : public Function<FuncVarOnDims> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& self, const IntArrayView dims, bool unbiased,
                        bool keepDim) {
    return op::varMeanOnDims(self, dims, unbiased, keepDim).first;
  }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncVarMean : public Function<FuncVarMean> {
 public:
  static TensorPair forward(AutogradContext* ctx, const Tensor& self, bool unbiased) {
    return op::varMean(self, unbiased);
  }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncVarMeanOnDim : public Function<FuncVarMeanOnDim> {
 public:
  static TensorPair forward(AutogradContext* ctx, const Tensor& self, int64_t dim, bool unbiased, bool keepDim) {
    return op::varMeanOnDim(self, dim, unbiased, keepDim);
  }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncVarMeanOnDims : public Function<FuncVarMeanOnDims> {
 public:
  static TensorPair forward(AutogradContext* ctx, const Tensor& self, const IntArrayView dims, bool unbiased,
                            bool keepDim) {
    return op::varMeanOnDims(self, dims, unbiased, keepDim);
  }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

inline Tensor min(const Tensor& self) { return FuncMin::apply(self); }
inline Tensor argmin(const Tensor& self) { return FuncArgmin::apply(self); }
inline TensorPair min(const Tensor& self, int64_t dim, bool keepDim) { return FuncMinOnDim::apply(self, dim, keepDim); }

inline Tensor max(const Tensor& self) { return FuncMax::apply(self); }
inline Tensor argmax(const Tensor& self) { return FuncArgmax::apply(self); }
inline TensorPair max(const Tensor& self, int64_t dim, bool keepDim) { return FuncMaxOnDim::apply(self, dim, keepDim); }

inline Tensor sum(const Tensor& self) { return FuncSum::apply(self); }
inline Tensor sum(const Tensor& self, int64_t dim, bool keepDim = false) {
  return FuncSumOnDim::apply(self, dim, keepDim);
}
inline Tensor sum(const Tensor& self, const IntArrayView dims, bool keepDim = false) {
  return FuncSumOnDims::apply(self, dims, keepDim);
}

inline Tensor mean(const Tensor& self) { return FuncMean::apply(self); }
inline Tensor mean(const Tensor& self, int64_t dim, bool keepDim = false) {
  return FuncMeanOnDim::apply(self, dim, keepDim);
}
inline Tensor mean(const Tensor& self, const IntArrayView dims, bool keepDim = false) {
  return FuncMeanOnDims::apply(self, dims, keepDim);
}

inline Tensor var(const Tensor& self, bool unbiased) { return FuncVar::apply(self, unbiased); }
inline Tensor var(const Tensor& self, int64_t dim, bool unbiased, bool keepDim = false) {
  return FuncVarOnDim::apply(self, dim, unbiased, keepDim);
}
inline Tensor var(const Tensor& self, const IntArrayView dims, bool unbiased, bool keepDim = false) {
  return FuncVarOnDims::apply(self, dims, unbiased, keepDim);
}

inline TensorPair varMean(const Tensor& self, bool unbiased) { return FuncVarMean::apply(self, unbiased); }
inline TensorPair varMean(const Tensor& self, int64_t dim, bool unbiased, bool keepDim = false) {
  return FuncVarMeanOnDim::apply(self, dim, unbiased, keepDim);
}
inline TensorPair varMean(const Tensor& self, const IntArrayView dims, bool unbiased, bool keepDim = false) {
  return FuncVarMeanOnDims::apply(self, dims, unbiased, keepDim);
}

}  // namespace tinytorch::function
