/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Module.h"

namespace tinytorch {

typedef enum {
  NONE = 0,
  MEAN = 1,
  SUM = 2,
} LossReduction;

namespace nn {
class Loss : public Module {
 public:
  explicit Loss(const LossReduction reduction = MEAN)
      : reduction_(reduction) {};
  ~Loss() override = default;

 private:
  std::vector<Tensor *> parameters() override { return {}; }
  void resetParameters() override {}
  void zeroGrad() override {}

 protected:
  LossReduction reduction_;
};

class MSELoss : public Loss {
 public:
  explicit MSELoss(const LossReduction reduction = MEAN) : Loss(reduction) {}
  Tensor forward(Tensor &input, Tensor &target) override;
};

class NLLLoss : public Loss {
 public:
  explicit NLLLoss(const LossReduction reduction = MEAN) : Loss(reduction) {}
  Tensor forward(Tensor &input, Tensor &target) override;
};

}  // namespace nn
}  // namespace tinytorch
