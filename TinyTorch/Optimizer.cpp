/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Optimizer.h"

#include "Logger.h"

namespace TinyTorch::optim {

Optimizer::Optimizer(std::vector<Tensor *> &&parameters, const float &lr)
    : parameters_(parameters), lr_(lr) {
  if (lr_ <= 0) {
    LOGE("Invalid learning rate: %f", lr_);
    return;
  }
}

void Optimizer::zeroGrad() {
  for (auto &param : parameters_) {
    param->zeroGrad();
  }
}

void SGD::step() {
  for (auto &param : parameters_) {
    auto &grad = param->getGrad().data();
    param->data() -= lr_ * grad;
  }
}

RMSprop::RMSprop(std::vector<Tensor *> &&parameters, const float &lr,
                 const float &alpha, const float &eps)
    : Optimizer(std::move(parameters), lr), alpha_(alpha), eps_(eps) {
  cache_.resize(parameters_.size());
  for (uint32_t i = 0; i < parameters_.size(); i++) {
    cache_[i] = TensorImpl::zeros(parameters_[i]->shape());
  }
}

void RMSprop::step() {
  for (uint32_t i = 0; i < parameters_.size(); i++) {
    auto &param = parameters_[i];
    auto &cache = cache_[i];
    auto &grad = param->getGrad().data();
    cache = alpha_ * cache + (1.f - alpha_) * grad * grad;
    param->data() -= lr_ * grad / (TensorImpl::sqrt(cache) + eps_);
  }
}

Adadelta::Adadelta(std::vector<Tensor *> &&parameters, const float &lr,
                   const float &rho, const float &eps)
    : Optimizer(std::move(parameters), lr), rho_(rho), eps_(eps) {
  eg2_.resize(parameters_.size());
  edx2_.resize(parameters_.size());
  for (uint32_t i = 0; i < parameters_.size(); i++) {
    eg2_[i] = TensorImpl::zeros(parameters_[i]->shape());
    edx2_[i] = TensorImpl::zeros(parameters_[i]->shape());
  }
}

void Adadelta::step() {
  for (uint32_t i = 0; i < parameters_.size(); i++) {
    auto &param = parameters_[i];
    auto &grad = param->getGrad().data();
    eg2_[i] = rho_ * eg2_[i] + (1 - rho_) * grad * grad;
    auto delta = lr_ * TensorImpl::sqrt(edx2_[i] + eps_) /
                 TensorImpl::sqrt(eg2_[i] + eps_) * grad;
    param->data() -= delta;
    edx2_[i] = rho_ * edx2_[i] + (1 - rho_) * delta * delta;
  }
}

}  // namespace TinyTorch::optim
