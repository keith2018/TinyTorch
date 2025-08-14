/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Operation/OpLoss.h"
#include "Tensor/Tensor.h"

namespace tinytorch::nn {

// TODO derived from Module
class Loss {
 public:
  explicit Loss(const LossReduction reduction = LossReduction::MEAN) : reduction_(reduction) {}
  virtual ~Loss() = default;

  virtual Tensor forward(Tensor &input, Tensor &target) = 0;

  template <typename... Args>
  Tensor operator()(Args &&...args) {
    return forward(std::forward<Args>(args)...);
  }

 protected:
  LossReduction reduction_;
};

class MSELoss : public Loss {
 public:
  explicit MSELoss(const LossReduction reduction = LossReduction::MEAN) : Loss(reduction) {}
  Tensor forward(Tensor &input, Tensor &target) override;
};

class NLLLoss : public Loss {
 public:
  explicit NLLLoss(const LossReduction reduction = LossReduction::MEAN) : Loss(reduction) {}
  Tensor forward(Tensor &input, Tensor &target) override;
};

// TODO CrossEntropyLoss

}  // namespace tinytorch::nn
