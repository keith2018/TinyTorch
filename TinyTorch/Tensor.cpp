/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Tensor.h"

#include <deque>
#include <set>
#include <unordered_map>

#include "Function.h"
#include "Logger.h"

namespace TinyTorch {

thread_local bool NoGradScope::gradEnabled = true;

Tensor::Tensor(const Array1d &values1d, bool requiresGrad)
    : data_(std::make_shared<TensorImpl>(values1d)) {
  initAutograd(requiresGrad);
}

Tensor::Tensor(const Array2d &values2d, bool requiresGrad)
    : data_(std::make_shared<TensorImpl>(values2d)) {
  initAutograd(requiresGrad);
}

Tensor::Tensor(const Array3d &values3d, bool requiresGrad)
    : data_(std::make_shared<TensorImpl>(values3d)) {
  initAutograd(requiresGrad);
}

Tensor Tensor::shape(const Shape &shape, bool requiresGrad) {
  auto ret = TensorImpl::shape(shape);
  return Tensor(std::move(ret), requiresGrad);
}

Tensor Tensor::scalar(const float &value, bool requiresGrad) {
  auto ret = TensorImpl::scalar(value);
  return Tensor(std::move(ret), requiresGrad);
}

Tensor Tensor::ones(const Shape &shape, bool requiresGrad) {
  auto ret = TensorImpl::ones(shape);
  return Tensor(std::move(ret), requiresGrad);
}

Tensor Tensor::onesLike(const Tensor &t, bool requiresGrad) {
  auto ret = TensorImpl::onesLike(*t.data_);
  return Tensor(std::move(ret), requiresGrad);
}

Tensor Tensor::zeros(const Shape &shape, bool requiresGrad) {
  auto ret = TensorImpl::zeros(shape);
  return Tensor(std::move(ret), requiresGrad);
}

Tensor Tensor::randn(const Shape &shape, bool requiresGrad) {
  auto ret = TensorImpl::randn(shape);
  return Tensor(std::move(ret), requiresGrad);
}

Tensor Tensor::arange(float start, float stop, float step, bool requiresGrad) {
  auto ret = TensorImpl::arange(start, stop, step);
  return Tensor(std::move(ret), requiresGrad);
}

Tensor Tensor::linspace(float start, float end, int steps, bool requiresGrad) {
  auto ret = TensorImpl::linspace(start, end, steps);
  return Tensor(std::move(ret), requiresGrad);
}

Tensor Tensor::reshape(const Shape &shape) const {
  return Function::reshape(*this, shape);
}

Tensor Tensor::reshape(const Tensor &t, const Shape &shape) {
  return Function::reshape(t, shape);
}

Tensor Tensor::flatten(int32_t startDim, int32_t endDim) const {
  return Function::flatten(*this, startDim, endDim);
}

Tensor Tensor::flatten(const Tensor &t, int32_t startDim, int32_t endDim) {
  return Function::flatten(t, startDim, endDim);
}

Tensor Tensor::unflatten(int32_t dim, const std::vector<int32_t> &sizes) const {
  return Function::unflatten(*this, dim, sizes);
}

Tensor Tensor::unflatten(const Tensor &t, int32_t dim,
                         const std::vector<int32_t> &sizes) {
  return Function::unflatten(t, dim, sizes);
}

Tensor Tensor::operator+(const Tensor &other) const {
  return Function::add(*this, other);
}

Tensor Tensor::operator-(const Tensor &other) const {
  return Function::sub(*this, other);
}

Tensor Tensor::operator*(const Tensor &other) const {
  return Function::mul(*this, other);
}

Tensor Tensor::operator/(const Tensor &other) const {
  return Function::div(*this, other);
}

Tensor Tensor::operator+(const float &other) const {
  return Function::add(*this, scalar(other));
}

Tensor Tensor::operator-(const float &other) const {
  return Function::sub(*this, scalar(other));
}

Tensor Tensor::operator*(const float &other) const {
  return Function::mul(*this, scalar(other));
}

Tensor Tensor::operator/(const float &other) const {
  return Function::div(*this, scalar(other));
}

Tensor operator+(const float &other, const Tensor &obj) {
  return Function::add(Tensor::scalar(other), obj);
}

Tensor operator-(const float &other, const Tensor &obj) {
  return Function::sub(Tensor::scalar(other), obj);
}

Tensor operator*(const float &other, const Tensor &obj) {
  return Function::mul(Tensor::scalar(other), obj);
}

Tensor operator/(const float &other, const Tensor &obj) {
  return Function::div(Tensor::scalar(other), obj);
}

void Tensor::operator+=(const Tensor &other) {
  *this = Function::add(*this, other);
}

void Tensor::operator-=(const Tensor &other) {
  *this = Function::sub(*this, other);
}

void Tensor::operator*=(const Tensor &other) {
  *this = Function::mul(*this, other);
}

void Tensor::operator/=(const Tensor &other) {
  *this = Function::div(*this, other);
}

void Tensor::operator+=(const float &other) {
  *this = Function::add(*this, scalar(other));
}

void Tensor::operator-=(const float &other) {
  *this = Function::sub(*this, scalar(other));
}

void Tensor::operator*=(const float &other) {
  *this = Function::mul(*this, scalar(other));
}

void Tensor::operator/=(const float &other) {
  *this = Function::div(*this, scalar(other));
}

Tensor Tensor::sin() const { return Function::sin(*this); }

Tensor Tensor::cos() const { return Function::cos(*this); }

Tensor Tensor::pow(const float &exp) const { return Function::pow(*this, exp); }

Tensor Tensor::pow(const Tensor &exp) const {
  return Function::pow(*this, exp);
}

Tensor Tensor::sum() const { return Function::sum(*this); }

Tensor Tensor::squeeze(int32_t dim) const {
  return Function::squeeze(*this, dim);
}

Tensor Tensor::unsqueeze(int32_t dim) const {
  return Function::unsqueeze(*this, dim);
}

Tensor::Tensor(TensorImpl &&data, bool requiresGrad,
               const std::shared_ptr<Function> &gradFunc)
    : data_(std::make_shared<TensorImpl>()) {
  *data_ = std::move(data);
  initAutograd(requiresGrad, gradFunc);
}

bool Tensor::isRequiresGrad() const { return requiresGrad_; }

void Tensor::setRequiresGrad(bool requires) { initAutograd(requires); }

void Tensor::backward(const Tensor &grad) {
  if (requiresGrad_) {
    gradMeta_->backward(grad);
  }
}

const Tensor &Tensor::getGrad() const {
  if (isLeaf()) {
    return gradMeta_->grad_;
  }
  LOGE(
      "The 'grad' attribute of a Tensor that is not a leaf Tensor is being "
      "accessed");
  static Tensor empty;
  return empty;
}

void Tensor::setGrad(const Tensor &grad) {
  if (requiresGrad_) {
    gradMeta_->grad_.data_ = grad.data_;
  }
}

void Tensor::zeroGrad() {
  if (requiresGrad_) {
    *gradMeta_->grad_.data_ =
        TensorImpl::zeros(data_->shape(), data_->device());
  }
}

void Tensor::initAutograd(bool requiresGrad,
                          const std::shared_ptr<Function> &gradFunc) {
  requiresGrad_ = requiresGrad;
  if (requiresGrad_) {
    gradMeta_ = std::make_shared<AutogradMeta>();
    gradMeta_->setGradFunc(gradFunc);
    *gradMeta_->grad_.data_ =
        TensorImpl::zeros(data_->shape(), data_->device());

    if (isLeaf()) {
      gradMeta_->gradLeaf_ = std::make_shared<FuncLeaf>();
      gradMeta_->gradLeaf_->setOwner(gradMeta_);
    } else {
      gradMeta_->gradLeaf_ = nullptr;
    }
  } else {
    gradMeta_ = nullptr;
  }
}

bool Tensor::isLeaf() const {
  return requiresGrad_ && gradMeta_->gradFunc_ == nullptr;
}

std::shared_ptr<Function> Tensor::getGradFunc() const {
  if (requiresGrad_) {
    return isLeaf() ? gradMeta_->gradLeaf_ : gradMeta_->gradFunc_;
  }
  return nullptr;
}

void AutogradMeta::setGradFunc(const std::shared_ptr<Function> &gradFunc) {
  gradFunc_ = gradFunc;
  if (gradFunc_) {
    gradFunc_->setOwner(shared_from_this());
  }
}

void AutogradMeta::backward(const Tensor &grad) {
  if (gradFunc_ == nullptr) {
    LOGE("error call backward: gradFunc == nullptr");
    return;
  }

  if (grad.empty()) {
    if (grad_.data_->dim() != 0) {
      LOGE("error call backward: input grad must not be omitted");
      return;
    }
  } else {
    if (grad.data_->shape() != grad_.data_->shape()) {
      LOGE("error call backward: input grad shape mismatch");
      return;
    }
  }

  if (backwardGraph_.empty()) {
    buildBackwardGraph();
  }

  std::unordered_map<std::shared_ptr<Function>, TensorImpl> inputs = {
      {gradFunc_,
       grad.empty() ? TensorImpl::scalar(1.f, grad_.device()) : *grad.data_}};
  for (auto &currFunc : backwardGraph_) {
    auto outputs = currFunc->callBackward(inputs[currFunc]);

    for (int i = 0; i < currFunc->nextFuncs().size(); i++) {
      auto &nextFunc = currFunc->nextFuncs()[i];

      if (inputs.find(nextFunc) != inputs.end()) {
        inputs[nextFunc] += outputs[i];
      } else {
        inputs[nextFunc] = outputs[i];
      }
    }
  }
}

void AutogradMeta::buildBackwardGraph() {
  std::unordered_map<std::shared_ptr<Function>, int> deps;
  std::deque<std::shared_ptr<Function>> q;

  std::set traversed = {gradFunc_};
  q.push_back(gradFunc_);
  while (!q.empty()) {
    const auto curr = q.front();
    q.pop_front();
    for (const auto &next : curr->nextFuncs()) {
      deps[next] += 1;
      if (traversed.find(next) == traversed.end()) {
        q.push_back(next);
        traversed.insert(next);
      }
    }
  }

  q.push_back(gradFunc_);
  while (!q.empty()) {
    const auto currFunc = q.front();
    backwardGraph_.push_back(currFunc);
    q.pop_front();
    for (const auto &nextFunc : currFunc->nextFuncs()) {
      deps[nextFunc] -= 1;
      if (deps[nextFunc] == 0) {
        q.push_back(nextFunc);
      }
    }
  }
}

}  // namespace TinyTorch
