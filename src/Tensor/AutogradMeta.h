/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <memory>
#include <vector>

#include "Tensor.h"

namespace tinytorch {

struct AutogradMeta {
  AutogradMeta() = default;
  Tensor grad_;
  std::shared_ptr<FunctionBase> gradFn_;
  std::vector<std::shared_ptr<FunctionBase>> backwardGraph_;

  void backward(const Tensor &grad);
  void buildBackwardGraph();
};

}  // namespace tinytorch
