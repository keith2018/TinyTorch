/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Tensor.h"

namespace tinytorch::optim {

class Optimizer {
 public:
  Optimizer(std::vector<TensorPtr> &&parameters, float lr, float weightDecay = 0.f);
  virtual ~Optimizer() = default;

  void step() {
    NoGradGuard guard;
    step_++;
    doStep();
  }

  virtual void doStep() = 0;

  float getLr() const { return lr_; }
  void setLr(float lr) { this->lr_ = lr; }

  void zeroGrad() const;

 protected:
  virtual Tensor getDecayedGrad(TensorPtr param) const;
  void initCache(std::vector<Tensor> &cache, bool setZero = false);

  std::vector<TensorPtr> parameters_;
  int64_t step_;
  float lr_;
  float weightDecay_;
};

class SGD : public Optimizer {
 public:
  explicit SGD(std::vector<TensorPtr> &&parameters, float lr = 0.001f, float momentum = 0.f, float dampening = 0.f,
               float weightDecay = 0.f, bool nesterov = false);
  void doStep() override;

 private:
  float momentum_;
  float dampening_;
  bool nesterov_;

  std::vector<Tensor> momentumBuffer_;
};

class Adagrad : public Optimizer {
 public:
  explicit Adagrad(std::vector<TensorPtr> &&parameters, float lr = 0.01f, float lrDecay = 0.f, float weightDecay = 0.f,
                   float initAcc = 0.f, float eps = 1e-10);
  void doStep() override;

 private:
  float lrDecay_;
  float initAcc_;
  float eps_;

  std::vector<Tensor> stateSums_;
};

class RMSprop : public Optimizer {
 public:
  explicit RMSprop(std::vector<TensorPtr> &&parameters, float lr = 0.01f, float alpha = 0.99f, float eps = 1e-8f,
                   float weightDecay = 0.f, float momentum = 0.f, bool centered = false);
  void doStep() override;

 private:
  float alpha_;
  float eps_;
  float momentum_;
  bool centered_;

  std::vector<Tensor> squareAvg_;
  std::vector<Tensor> momentumBuffer_;
  std::vector<Tensor> gradAvg_;
};

class AdaDelta : public Optimizer {
 public:
  explicit AdaDelta(std::vector<TensorPtr> &&parameters, float lr = 1.0f, float rho = 0.9f, float eps = 1e-6f,
                    float weightDecay = 0.f);
  void doStep() override;

 private:
  float rho_;
  float eps_;
  std::vector<Tensor> squareAvg_;
  std::vector<Tensor> accDelta_;
};

class Adam : public Optimizer {
 public:
  explicit Adam(std::vector<TensorPtr> &&parameters, float lr = 0.001f,
                const std::pair<float, float> &betas = {0.9f, 0.999f}, float eps = 1e-8f, float weightDecay = 0.f,
                bool amsGrad = false);
  void doStep() override;

 protected:
  float beta1_;
  float beta2_;
  float eps_;
  bool amsGrad_;
  std::vector<Tensor> expAvg_;
  std::vector<Tensor> expAvgSq_;
  std::vector<Tensor> maxExpAvgSq_;
};

class AdamW : public Adam {
 public:
  explicit AdamW(std::vector<TensorPtr> &&parameters, float lr = 0.001f,
                 const std::pair<float, float> &betas = {0.9f, 0.999f}, float eps = 1e-8f, float weightDecay = 0.01f,
                 bool amsGrad = false);
  Tensor getDecayedGrad(TensorPtr param) const override;
};

}  // namespace tinytorch::optim
