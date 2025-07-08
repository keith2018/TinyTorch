/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Tensor.h"

#include "AutogradMeta.h"
#include "Function.h"
#include "Functions.h"
#include "Operations.h"
#include "Utils/Macros.h"

namespace tinytorch {

Tensor::Tensor(const IntArrayView shape, Options options) {
  impl_ = std::make_shared<TensorImpl>(shape, options);
  initAutogradMeta();
}

Tensor::Tensor(const IntArrayView shape, Options options, const std::shared_ptr<Storage> &storage, int64_t offset) {
  impl_ = std::make_shared<TensorImpl>(shape, options, storage, offset);
  initAutogradMeta();
}

Tensor Tensor::empty(const IntArrayView shape, Options options) { return {shape, options}; }

Tensor Tensor::scalar(const Scalar &scalar, Options options) {
  Tensor ret({}, options);
  op::fill(ret, scalar);
  return ret;
}

Tensor Tensor::ones(const IntArrayView shape, Options options) {
  Tensor ret(shape, options);
  op::fill(ret, 1);
  return ret;
}

Tensor Tensor::onesLike(const Tensor &t, Options options) { return ones(t.shape(), options); }

Tensor Tensor::zeros(const IntArrayView shape, Options options) {
  Tensor ret(shape, options);
  op::fill(ret, 0);
  return ret;
}

Tensor Tensor::zerosLike(const Tensor &t, Options options) { return zeros(t.shape(), options); }

Tensor Tensor::rand(const IntArrayView shape, Options options) {
  Tensor ret(shape, options);
  op::fillRandUniform(ret, 0.f, 1.f);
  return ret;
}

Tensor Tensor::randn(const IntArrayView shape, Options options) {
  Tensor ret(shape, options);
  op::fillRandNormal(ret);
  return ret;
}

Tensor Tensor::uniform(const IntArrayView shape, float min, float max, Options options) {
  Tensor ret(shape, options);
  op::fillRandUniform(ret, min, max);
  return ret;
}

Tensor Tensor::bernoulli(const IntArrayView shape, float p, Options options) {
  Tensor ret(shape, options);
  op::fillRandBernoulli(ret, p);
  return ret;
}

void Tensor::setRequiresGrad(bool require, const std::shared_ptr<FunctionBase> &fn) {
  impl_->options().requiresGrad_ = require;
  initAutogradMeta(fn);
}

void Tensor::initAutogradMeta(const std::shared_ptr<FunctionBase> &fn) {
  if (requiresGrad()) {
    if (gradMeta_ == nullptr) {
      gradMeta_ = std::make_unique<AutogradMeta>();
    }
    setGradFn(fn ? fn : std::make_shared<FuncLeaf>());
  } else {
    gradMeta_.reset();
  }
}

void Tensor::inplaceSet(Tensor &&val) {
  auto reqGrad = requiresGrad();
  impl_ = std::move(val.impl_);
  impl_->options().requiresGrad(reqGrad);
}

void Tensor::backwardImpl(const Tensor &grad) const {
  if (!requiresGrad()) {
    LOGE("call backward while requiresGrad == false");
    return;
  }
  ASSERT(gradMeta_ != nullptr);
  gradMeta_->backward(grad);
}

Tensor Tensor::clone() const {
  Tensor ret;
  ret.impl_ = std::make_shared<TensorImpl>(*this->impl_);
  ret.initAutogradMeta();
  return ret;
}

std::shared_ptr<FunctionBase> &Tensor::gradFn() const {
  ASSERT(gradMeta_ != nullptr);
  return gradMeta_->gradFn_;
}

void Tensor::setGradFn(const std::shared_ptr<FunctionBase> &fn) const {
  ASSERT(gradMeta_ != nullptr);
  ASSERT(fn != nullptr);
  gradMeta_->gradFn_ = fn;
  fn->weakOwner = gradMeta_;
}

const Tensor &Tensor::grad() const {
  ASSERT(gradMeta_ != nullptr);
  return gradMeta_->grad_;
}

void Tensor::setGrad(const Tensor &grad) const {
  ASSERT(gradMeta_ != nullptr);
  gradMeta_->grad_ = grad;
}

void Tensor::setGrad(Tensor &&grad) const {
  ASSERT(gradMeta_ != nullptr);
  gradMeta_->grad_ = std::move(grad);
}

void Tensor::zeroGrad() const {
  ASSERT(gradMeta_ != nullptr);
  if (!gradMeta_->grad_.defined()) {
    gradMeta_->grad_ = empty(shape(), options().noGrad());
  }
  op::fill(gradMeta_->grad_, 0);
}

void Tensor::backward(const Tensor &grad) const {
  ASSERT(grad.defined());
  if (shape() != grad.shape()) {
    LOGE("error call backward: input grad shape mismatch");
    return;
  }
  backwardImpl(grad);
}

void Tensor::backward() const {
  if (dim() != 0) {
    LOGE("error call backward: input grad must not be omitted");
    return;
  }
  backwardImpl(scalar(1, options().noGrad()));
}

bool Tensor::isLeaf() const {
  if (!requiresGrad()) {
    return false;
  }
  ASSERT(gradMeta_ != nullptr);
  ASSERT(gradMeta_->gradFn_ != nullptr);
  return typeid(*gradMeta_->gradFn_) == typeid(FuncLeaf);
}

Tensor Tensor::to(DType type) const {
  if (dtype() == type) {
    return *this;
  }
  Options opt = options();
  opt.dtype(type);
  Tensor ret(shape(), opt);
  op::dtypeCast(ret, *this);
  return ret;
}

Tensor Tensor::to(Device device) const {
  if (device == this->device()) {
    return *this;
  }
  Options opt = options();
  opt.device(device);
  Tensor ret(shape(), opt);
  int64_t nbytes = numel() * static_cast<int64_t>(dtypeSize(dtype()));
  Storage::copyOnDevice(ret.dataPtr<>(), ret.device(), this->dataPtr<>(), this->device(), nbytes);
  return ret;
}

void Tensor::fill(const Scalar &val) { op::fill(*this, val); }
void Tensor::fillMasked(const Tensor &mask, const Scalar &val) { op::fillMaskedInplace(*this, mask, val); }
void Tensor::fillLinSpace(const Scalar &start, const Scalar &step, int64_t steps) {
  op::fillLinSpace(*this, start, step, steps);
}
void Tensor::fillUniform(float min, float max) { op::fillRandUniform(*this, min, max); }
void Tensor::fillNormal() { op::fillRandNormal(*this); }
void Tensor::fillBernoulli(float p) { op::fillRandBernoulli(*this, p); }

Tensor Tensor::operator+(const Tensor &other) const { return function::add(*this, other); }
Tensor Tensor::operator-(const Tensor &other) const { return function::sub(*this, other); }
Tensor Tensor::operator*(const Tensor &other) const { return function::mul(*this, other); }
Tensor Tensor::operator/(const Tensor &other) const { return function::div(*this, other); }

Tensor Tensor::operator+(const Scalar &other) const { return function::add(*this, scalar(other, options().noGrad())); }
Tensor Tensor::operator-(const Scalar &other) const { return function::sub(*this, scalar(other, options().noGrad())); }
Tensor Tensor::operator*(const Scalar &other) const { return function::mul(*this, scalar(other, options().noGrad())); }
Tensor Tensor::operator/(const Scalar &other) const { return function::div(*this, scalar(other, options().noGrad())); }

Tensor operator+(const Scalar &self, const Tensor &other) {
  return function::add(Tensor::scalar(self, other.options().noGrad()), other);
}
Tensor operator-(const Scalar &self, const Tensor &other) {
  return function::sub(Tensor::scalar(self, other.options().noGrad()), other);
}
Tensor operator*(const Scalar &self, const Tensor &other) {
  return function::mul(Tensor::scalar(self, other.options().noGrad()), other);
}
Tensor operator/(const Scalar &self, const Tensor &other) {
  return function::div(Tensor::scalar(self, other.options().noGrad()), other);
}

void Tensor::operator+=(const Tensor &other) { inplaceSet(function::add(*this, other)); }
void Tensor::operator-=(const Tensor &other) { inplaceSet(function::sub(*this, other)); }
void Tensor::operator*=(const Tensor &other) { inplaceSet(function::mul(*this, other)); }
void Tensor::operator/=(const Tensor &other) { inplaceSet(function::div(*this, other)); }

void Tensor::operator+=(const Scalar &other) { inplaceSet(function::add(*this, scalar(other, options().noGrad()))); }
void Tensor::operator-=(const Scalar &other) { inplaceSet(function::sub(*this, scalar(other, options().noGrad()))); }
void Tensor::operator*=(const Scalar &other) { inplaceSet(function::mul(*this, scalar(other, options().noGrad()))); }
void Tensor::operator/=(const Scalar &other) { inplaceSet(function::div(*this, scalar(other, options().noGrad()))); }

Tensor Tensor::operator<(const Tensor &other) const { return op::lt(*this, other); }
Tensor Tensor::operator<=(const Tensor &other) const { return op::le(*this, other); }
Tensor Tensor::operator>(const Tensor &other) const { return op::gt(*this, other); }
Tensor Tensor::operator>=(const Tensor &other) const { return op::ge(*this, other); }
Tensor Tensor::operator==(const Tensor &other) const { return op::eq(*this, other); }
Tensor Tensor::operator!=(const Tensor &other) const { return op::ne(*this, other); }

Tensor Tensor::operator<(const Scalar &other) const { return op::lt(*this, scalar(other, options().noGrad())); }
Tensor Tensor::operator<=(const Scalar &other) const { return op::le(*this, scalar(other, options().noGrad())); }
Tensor Tensor::operator>(const Scalar &other) const { return op::gt(*this, scalar(other, options().noGrad())); }
Tensor Tensor::operator>=(const Scalar &other) const { return op::ge(*this, scalar(other, options().noGrad())); }
Tensor Tensor::operator==(const Scalar &other) const { return op::eq(*this, scalar(other, options().noGrad())); }
Tensor Tensor::operator!=(const Scalar &other) const { return op::ne(*this, scalar(other, options().noGrad())); }

Tensor Tensor::sin() const { return function::sin(*this); }
Tensor Tensor::cos() const { return function::cos(*this); }
Tensor Tensor::pow(const Scalar &exp) const { return function::pow(*this, scalar(exp, options().noGrad())); }
Tensor Tensor::pow(const Tensor &exp) const { return function::pow(*this, exp); }

Tensor Tensor::maximum(const Tensor &a, const Tensor &b) { return function::maximum(a, b); }
Tensor Tensor::minimum(const Tensor &a, const Tensor &b) { return function::minimum(a, b); }

Tensor Tensor::sum() const { return function::sum(*this); }

void Tensor::reshape(const IntArrayView shape) { inplaceSet(function::reshape(*this, shape)); }
Tensor Tensor::permute(const IntArrayView dims) const { return function::permute(*this, dims); }
Tensor Tensor::permute() const { return function::permute(*this); }
Tensor Tensor::flatten(int64_t startDim, int64_t endDim) const { return function::flatten(*this, startDim, endDim); }
Tensor Tensor::unflatten(int64_t dim, IntArrayView shape) const { return function::unflatten(*this, dim, shape); }
Tensor Tensor::squeeze(int64_t dim) const { return function::squeeze(*this, dim); }
Tensor Tensor::squeeze(IntArrayView dims) const { return function::squeeze(*this, dims); }
Tensor Tensor::unsqueeze(int64_t dim) const { return function::unsqueeze(*this, dim); }

}  // namespace tinytorch
