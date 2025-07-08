/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Module.h"

namespace tinytorch {

enum class LossReduction : int8_t {
  NONE = 0,
  MEAN = 1,
  SUM = 2,
};

namespace nn {
class Loss : public Module {
 public:
  explicit Loss(const LossReduction reduction = LossReduction::MEAN) : reduction_(reduction) {}
  ~Loss() override = default;

 private:
  std::vector<TensorPtr> parameters() override { return {}; }
  void resetParameters() override {}
  void zeroGrad() override {}

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

}  // namespace nn
}  // namespace tinytorch
