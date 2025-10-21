/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Module.h"

namespace tinytorch::nn {

class LayerNorm : public Module {
 public:
  explicit LayerNorm(IntArrayView normalizedShape, float eps = 1e-5, bool bias = true, Options options = {});

  Tensor forward(const Tensor &input) override;
  void resetParameters() override;

  Tensor &weight() { return weight_; }
  Tensor &bias() { return bias_; }

 protected:
  std::vector<std::pair<std::string, TensorPtr>> namedParameters_() override;

  SizeVector normalizedShape_;
  float eps_;
  bool useBias_;
  Tensor weight_;
  Tensor bias_;
};

class RMSNorm : public Module {
 public:
  explicit RMSNorm(IntArrayView normalizedShape, float eps = 1e-5, Options options = {});

  Tensor forward(const Tensor &input) override;
  void resetParameters() override;

  Tensor &weight() { return weight_; }

 protected:
  std::vector<std::pair<std::string, TensorPtr>> namedParameters_() override;

  SizeVector normalizedShape_;
  float eps_;
  Tensor weight_;
};

}  // namespace tinytorch::nn
