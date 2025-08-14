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

Tensor Tensor::zeros(const IntArrayView shape, Options options) {
  Tensor ret(shape, options);
  ret.fillZero_();
  return ret;
}

Tensor Tensor::full(const IntArrayView shape, const Scalar &scalar, Options options) {
  Tensor ret(shape, options);
  op::fill(ret, scalar);
  return ret;
}

Tensor Tensor::rand(const IntArrayView shape, Options options) {
  Tensor ret(shape, options);
  op::fillRandUniform(ret, 0.f, 1.f);
  return ret;
}

Tensor Tensor::randn(const IntArrayView shape, Options options) {
  Tensor ret(shape, options);
  op::fillRandNormal(ret, 0.f, 1.f);
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

Tensor Tensor::onesLike(const Tensor &t, std::optional<Options> options) {
  auto ops = options ? *options : t.options();
  return ones(t.shape(), ops);
}

Tensor Tensor::zerosLike(const Tensor &t, std::optional<Options> options) {
  auto ops = options ? *options : t.options();
  return zeros(t.shape(), ops);
}

Tensor Tensor::fullLike(const Tensor &t, const Scalar &scalar, std::optional<Options> options) {
  auto ops = options ? *options : t.options();
  Tensor ret(t.shape(), ops);
  op::fill(ret, scalar);
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
  return gradMeta_->gradFn();
}

void Tensor::setGradFn(const std::shared_ptr<FunctionBase> &fn) const {
  ASSERT(gradMeta_ != nullptr);
  ASSERT(fn != nullptr);
  gradMeta_->setGradFn(fn);
}

const Tensor &Tensor::grad() const {
  ASSERT(gradMeta_ != nullptr);
  return gradMeta_->grad();
}

void Tensor::setGrad(const Tensor &grad) const {
  ASSERT(gradMeta_ != nullptr);
  gradMeta_->setGrad(grad);
}

void Tensor::setGrad(Tensor &&grad) const {
  ASSERT(gradMeta_ != nullptr);
  gradMeta_->setGrad(std::move(grad));
}

void Tensor::addGrad(const Tensor &grad) const {
  ASSERT(gradMeta_ != nullptr);
  gradMeta_->addGrad(grad);
}

void Tensor::addGrad(Tensor &&grad) const {
  ASSERT(gradMeta_ != nullptr);
  gradMeta_->addGrad(std::move(grad));
}

void Tensor::zeroGrad() const {
  ASSERT(gradMeta_ != nullptr);
  gradMeta_->zeroGrad(*this);
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
  return gradMeta_->isLeaf();
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

void Tensor::fill_(const Scalar &val) { op::fill(*this, val); }
void Tensor::fillMasked_(const Tensor &mask, const Scalar &val) { op::fillMaskedInplace(*this, mask, val); }
void Tensor::fillZero_() { op::fill(*this, 0); }
void Tensor::fillOne_() { op::fill(*this, 1); }
void Tensor::fillLinSpace_(const Scalar &start, const Scalar &step, int64_t steps) {
  op::fillLinSpace(*this, start, step, steps);
}
void Tensor::fillUniform_(float min, float max) { op::fillRandUniform(*this, min, max); }
void Tensor::fillNormal_(float mean, float stddev) { op::fillRandNormal(*this, mean, stddev); }
void Tensor::fillBernoulli_(float p) { op::fillRandBernoulli(*this, p); }

void Tensor::scatter_(int64_t dim, const Tensor &index, const Tensor &src) {
  function::scatter_(*this, dim, index, src);
}

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

Tensor Tensor::operator<(const Tensor &other) const { return function::lt(*this, other); }
Tensor Tensor::operator<=(const Tensor &other) const { return function::le(*this, other); }
Tensor Tensor::operator>(const Tensor &other) const { return function::gt(*this, other); }
Tensor Tensor::operator>=(const Tensor &other) const { return function::ge(*this, other); }
Tensor Tensor::operator==(const Tensor &other) const { return function::eq(*this, other); }
Tensor Tensor::operator!=(const Tensor &other) const { return function::ne(*this, other); }

Tensor Tensor::operator<(const Scalar &other) const { return function::lt(*this, scalar(other, options().noGrad())); }
Tensor Tensor::operator<=(const Scalar &other) const { return function::le(*this, scalar(other, options().noGrad())); }
Tensor Tensor::operator>(const Scalar &other) const { return function::gt(*this, scalar(other, options().noGrad())); }
Tensor Tensor::operator>=(const Scalar &other) const { return function::ge(*this, scalar(other, options().noGrad())); }
Tensor Tensor::operator==(const Scalar &other) const { return function::eq(*this, scalar(other, options().noGrad())); }
Tensor Tensor::operator!=(const Scalar &other) const { return function::ne(*this, scalar(other, options().noGrad())); }

Tensor Tensor::operator~() const { return function::logicNot(*this); }
Tensor Tensor::operator&(const Tensor &other) const { return function::logicAnd(*this, other); }
Tensor Tensor::operator|(const Tensor &other) const { return function::logicOr(*this, other); }

Tensor Tensor::sin() const { return function::sin(*this); }
Tensor Tensor::cos() const { return function::cos(*this); }
Tensor Tensor::sqrt() const { return function::sqrt(*this); }
Tensor Tensor::pow(const Scalar &exp) const { return function::pow(*this, scalar(exp, options().noGrad())); }
Tensor Tensor::pow(const Tensor &exp) const { return function::pow(*this, exp); }

Tensor Tensor::maximum(const Tensor &a, const Tensor &b) { return function::maximum(a, b); }
Tensor Tensor::minimum(const Tensor &a, const Tensor &b) { return function::minimum(a, b); }

Tensor Tensor::matmul(const Tensor &other, bool transA, bool transB) const {
  return function::matmul(*this, other, transA, transB);
}

Tensor Tensor::min() const { return function::min(*this); }
Tensor Tensor::max() const { return function::max(*this); }
Tensor Tensor::argmin() const { return function::argmin(*this); }
Tensor Tensor::argmax() const { return function::argmax(*this); }
Tensor Tensor::sum() const { return function::sum(*this); }
Tensor Tensor::mean() const { return function::mean(*this); }
Tensor Tensor::var(bool unbiased) const { return function::var(*this, unbiased); }
TensorPair Tensor::varMean(bool unbiased) const { return function::varMean(*this, unbiased); }

TensorPair Tensor::min(int64_t dim, bool keepDims) const { return function::min(*this, dim, keepDims); }
TensorPair Tensor::max(int64_t dim, bool keepDims) const { return function::max(*this, dim, keepDims); }
Tensor Tensor::argmin(int64_t dim, bool keepDims) const { return function::min(*this, dim, keepDims).second; }
Tensor Tensor::argmax(int64_t dim, bool keepDims) const { return function::max(*this, dim, keepDims).second; }
Tensor Tensor::sum(int64_t dim, bool keepDims) const { return function::sum(*this, dim, keepDims); }
Tensor Tensor::mean(int64_t dim, bool keepDims) const { return function::mean(*this, dim, keepDims); }
Tensor Tensor::var(int64_t dim, bool unbiased, bool keepDims) const {
  return function::var(*this, dim, unbiased, keepDims);
}
TensorPair Tensor::varMean(int64_t dim, bool unbiased, bool keepDims) const {
  return function::varMean(*this, dim, unbiased, keepDims);
}

Tensor Tensor::sum(IntArrayView dims, bool keepDims) const { return function::sum(*this, dims, keepDims); }
Tensor Tensor::mean(IntArrayView dims, bool keepDims) const { return function::mean(*this, dims, keepDims); }
Tensor Tensor::var(IntArrayView dims, bool unbiased, bool keepDims) const {
  return function::var(*this, dims, unbiased, keepDims);
}
TensorPair Tensor::varMean(IntArrayView dims, bool unbiased, bool keepDims) const {
  return function::varMean(*this, dims, unbiased, keepDims);
}

void Tensor::reshape_(const IntArrayView shape) { inplaceSet(function::reshape(*this, shape)); }
void Tensor::permute_(const IntArrayView dims) { inplaceSet(function::permute(*this, dims)); }
void Tensor::permute_() { inplaceSet(function::permute(*this)); }
void Tensor::flatten_(int64_t startDim, int64_t endDim) { inplaceSet(function::flatten(*this, startDim, endDim)); }
void Tensor::unflatten_(int64_t dim, IntArrayView shape) { inplaceSet(function::unflatten(*this, dim, shape)); }
void Tensor::squeeze_(int64_t dim) { inplaceSet(function::squeeze(*this, dim)); }
void Tensor::squeeze_(IntArrayView dims) { inplaceSet(function::squeeze(*this, dims)); }
void Tensor::unsqueeze_(int64_t dim) { inplaceSet(function::unsqueeze(*this, dim)); }
void Tensor::transpose_(int64_t dim0, int64_t dim1) { inplaceSet(function::transpose(*this, dim0, dim1)); }
void Tensor::t_() { inplaceSet(function::transpose(*this, 0, 1)); }

Tensor Tensor::reshape(const IntArrayView shape) const { return function::reshape(*this, shape); }
Tensor Tensor::view(const IntArrayView shape) const { return function::view(*this, shape); }
Tensor Tensor::permute(const IntArrayView dims) const { return function::permute(*this, dims); }
Tensor Tensor::permute() const { return function::permute(*this); }
Tensor Tensor::flatten(int64_t startDim, int64_t endDim) const { return function::flatten(*this, startDim, endDim); }
Tensor Tensor::unflatten(int64_t dim, IntArrayView shape) const { return function::unflatten(*this, dim, shape); }
Tensor Tensor::squeeze(int64_t dim) const { return function::squeeze(*this, dim); }
Tensor Tensor::squeeze(IntArrayView dims) const { return function::squeeze(*this, dims); }
Tensor Tensor::unsqueeze(int64_t dim) const { return function::unsqueeze(*this, dim); }
Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const { return function::transpose(*this, dim0, dim1); }
Tensor Tensor::t() const { return function::transpose(*this, 0, 1); }

}  // namespace tinytorch
