/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Optimizer.h"

#include "Operations.h"
#include "Utils/Logger.h"

namespace tinytorch::optim {

Optimizer::Optimizer(std::vector<TensorPtr> &&parameters, float lr, float weightDecay)
    : parameters_(parameters), step_(0), lr_(lr), weightDecay_(weightDecay) {
  if (lr_ <= 0) {
    LOGE("Invalid learning rate: %f", lr_);
    return;
  }
}

void Optimizer::zeroGrad() const {
  for (auto &param : parameters_) {
    param->zeroGrad();
  }
}

Tensor Optimizer::getDecayedGrad(TensorPtr param) const {
  auto ret = param->grad();
  if (!ret.defined()) {
    return ret;
  }
  if (weightDecay_ != 0.f) {
    ret += weightDecay_ * (*param);
  }
  return ret;
}

void Optimizer::initCache(std::vector<Tensor> &cache, bool setZero) {
  cache.resize(parameters_.size());
  if (setZero) {
    for (size_t i = 0; i < parameters_.size(); i++) {
      cache[i] = Tensor::zeros(parameters_[i]->shape(), parameters_[i]->options());
    }
  }
}

SGD::SGD(std::vector<TensorPtr> &&parameters, float lr, float momentum, float dampening, float weightDecay,
         bool nesterov)
    : Optimizer(std::move(parameters), lr, weightDecay),
      momentum_(momentum),
      dampening_(dampening),
      nesterov_(nesterov) {
  if (momentum_ != 0.f) {
    initCache(momentumBuffer_);
  }

  if (nesterov && (momentum <= 0 || dampening != 0)) {
    LOGE("SGD error: Nesterov momentum requires a momentum and zero dampening");
  }
}

void SGD::doStep() {
  for (size_t i = 0; i < parameters_.size(); i++) {
    auto &param = parameters_[i];
    auto grad = getDecayedGrad(param);
    if (!grad.defined()) {
      continue;
    }
    if (momentum_ != 0.f) {
      auto &buf = momentumBuffer_[i];
      buf = buf.defined() ? (momentum_ * buf + (1.f - dampening_) * grad) : grad;
      if (nesterov_) {
        grad += momentum_ * buf;
      } else {
        grad = buf;
      }
    }
    *param -= lr_ * grad;
  }
}

Adagrad::Adagrad(std::vector<TensorPtr> &&parameters, float lr, float lrDecay, float weightDecay, float initAcc,
                 float eps)
    : Optimizer(std::move(parameters), lr, weightDecay), lrDecay_(lrDecay), initAcc_(initAcc), eps_(eps) {
  UNUSED(initAcc_);
  stateSums_.resize(parameters_.size());
  for (size_t i = 0; i < parameters_.size(); i++) {
    stateSums_[i] = Tensor::empty(parameters_[i]->shape(), parameters_[i]->options());
    stateSums_[i].fill_(initAcc);
  }
}

void Adagrad::doStep() {
  for (size_t i = 0; i < parameters_.size(); i++) {
    auto &param = parameters_[i];
    auto grad = getDecayedGrad(param);
    if (!grad.defined()) {
      continue;
    }
    auto &s = stateSums_[i];
    auto clr = lr_ / (1 + static_cast<float>(step_ - 1) * lrDecay_);
    s += grad * grad;
    *param -= clr * grad / (s.sqrt() + eps_);
  }
}

RMSprop::RMSprop(std::vector<TensorPtr> &&parameters, float lr, float alpha, float eps, float weightDecay,
                 float momentum, bool centered)
    : Optimizer(std::move(parameters), lr, weightDecay),
      alpha_(alpha),
      eps_(eps),
      momentum_(momentum),
      centered_(centered) {
  initCache(squareAvg_, true);
  if (momentum_ != 0.f) {
    initCache(momentumBuffer_, true);
  }
  if (centered_) {
    initCache(gradAvg_, true);
  }
}

void RMSprop::doStep() {
  for (size_t i = 0; i < parameters_.size(); i++) {
    auto &param = parameters_[i];
    auto grad = getDecayedGrad(param);
    if (!grad.defined()) {
      continue;
    }
    auto &v = squareAvg_[i];
    v = alpha_ * v + (1.f - alpha_) * grad * grad;
    Tensor avg;
    if (centered_) {
      auto &g = gradAvg_[i];
      g = alpha_ * g + (1.f - alpha_) * grad;
      avg = v - g * g;
    } else {
      avg = v.sqrt();
    }
    avg += eps_;
    if (momentum_ != 0.f) {
      auto &buf = momentumBuffer_[i];
      buf = momentum_ * buf + grad / avg;
      *param -= lr_ * buf;
    } else {
      *param -= lr_ * grad / avg;
    }
  }
}

AdaDelta::AdaDelta(std::vector<TensorPtr> &&parameters, float lr, float rho, float eps, float weightDecay)
    : Optimizer(std::move(parameters), lr, weightDecay), rho_(rho), eps_(eps) {
  initCache(squareAvg_, true);
  initCache(accDelta_, true);
}

void AdaDelta::doStep() {
  for (size_t i = 0; i < parameters_.size(); i++) {
    auto &param = parameters_[i];
    auto grad = getDecayedGrad(param);
    if (!grad.defined()) {
      continue;
    }
    auto &v = squareAvg_[i];
    auto &u = accDelta_[i];
    v = rho_ * v + (1.f - rho_) * grad * grad;
    auto delta = (u + eps_).sqrt() / (v + eps_).sqrt() * grad;
    u = rho_ * u + (1.f - rho_) * delta * delta;
    *param -= lr_ * delta;
  }
}

Adam::Adam(std::vector<TensorPtr> &&parameters, float lr, const std::pair<float, float> &betas, float eps,
           float weightDecay, bool amsGrad)
    : Optimizer(std::move(parameters), lr, weightDecay),
      beta1_(betas.first),
      beta2_(betas.second),
      eps_(eps),
      amsGrad_(amsGrad) {
  initCache(expAvg_, true);
  initCache(expAvgSq_, true);
  if (amsGrad_) {
    initCache(maxExpAvgSq_, true);
  }
}

void Adam::doStep() {
  float b1t = 1.f - std::pow(beta1_, static_cast<float>(step_));
  float b2t = 1.f - std::pow(beta2_, static_cast<float>(step_));

  for (size_t i = 0; i < parameters_.size(); i++) {
    auto &param = parameters_[i];
    auto grad = getDecayedGrad(param);
    if (!grad.defined()) {
      continue;
    }
    auto &m = expAvg_[i];
    auto &v = expAvgSq_[i];
    m = beta1_ * m + (1.f - beta1_) * grad;
    v = beta2_ * v + (1.f - beta2_) * grad * grad;
    auto mh = m / b1t;
    auto vh = v / b2t;
    if (amsGrad_) {
      auto &vMax = maxExpAvgSq_[i];
      vh = vMax = Tensor::maximum(vMax, vh);
    }
    // auto clr = lr_ * std::sqrt(b2t) / b1t;
    *param -= lr_ * mh / (vh.sqrt() + eps_);
  }
}

AdamW::AdamW(std::vector<TensorPtr> &&parameters, float lr, const std::pair<float, float> &betas, float eps,
             float weightDecay, bool amsGrad)
    : Adam(std::move(parameters), lr, betas, eps, weightDecay, amsGrad) {}

Tensor AdamW::getDecayedGrad(TensorPtr param) const {
  *param *= (1.f - lr_ * weightDecay_);
  return param->grad();
}

}  // namespace tinytorch::optim
