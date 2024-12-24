/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <unordered_map>

#include "Tensor.h"

namespace TinyTorch::optim {

class Optimizer {
 public:
  Optimizer(std::vector<Tensor *> &&parameters, const float &lr);
  virtual ~Optimizer() = default;

  virtual void step() = 0;

  float getLr() const { return lr_; }
  void setLr(const float &lr) { this->lr_ = lr; }

  void zeroGrad();

 protected:
  std::vector<Tensor *> parameters_;
  float lr_;
};

class SGD : public Optimizer {
 public:
  explicit SGD(std::vector<Tensor *> &&parameters, const float &lr = 0.001f)
      : Optimizer(std::move(parameters), lr) {}
  void step() override;
};

class RMSprop : public Optimizer {
 public:
  explicit RMSprop(std::vector<Tensor *> &&parameters, const float &lr = 0.01f,
                   const float &alpha = 0.99f, const float &eps = 1e-8f);
  void step() override;

 private:
  float alpha_;
  float eps_;
  std::vector<TensorImpl> cache_;
};

class Adadelta : public Optimizer {
 public:
  explicit Adadelta(std::vector<Tensor *> &&parameters, const float &lr = 1.0f,
                    const float &rho = 0.9f, const float &eps = 1e-6f);
  void step() override;

 private:
  float rho_;
  float eps_;
  std::vector<TensorImpl> eg2_;
  std::vector<TensorImpl> edx2_;
};

}  // namespace TinyTorch::optim
