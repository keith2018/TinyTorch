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

class AutogradMeta : public std::enable_shared_from_this<AutogradMeta> {
 public:
  AutogradMeta() = default;

  std::shared_ptr<FunctionBase>& gradFn() { return gradFn_; }
  void setGradFn(const std::shared_ptr<FunctionBase>& fn);

  const Tensor& grad() const { return grad_; }
  void setGrad(const Tensor& grad);
  void setGrad(Tensor&& grad);
  void addGrad(const Tensor& grad);
  void addGrad(Tensor&& grad);
  void zeroGrad(const Tensor& owner);

  bool isLeaf() const;
  void backward(const Tensor& grad);
  void buildBackwardGraph();

 private:
  Tensor grad_;
  std::shared_ptr<FunctionBase> gradFn_;
  std::vector<std::shared_ptr<FunctionBase>> backwardGraph_;
};

}  // namespace tinytorch
