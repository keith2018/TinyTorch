/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "TensorImpl.h"

#include <cassert>
#include <numeric>

#include "TensorImpl_cpu.h"
#ifdef USE_CUDA
#include "TensorImpl_cuda.cuh"
#endif

namespace tinytorch {

#define TENSOR_DEVICE_AVAILABLE(device, ret)                            \
  do {                                                                  \
    if (!deviceAvailable(device)) {                                     \
      TensorOperations::error(__FUNCTION__, TensorError_InvalidDevice); \
      return ret;                                                       \
    }                                                                   \
  } while (0)

#define TENSOR_CHECK_DEVICE_RET(a, b, ret)                                 \
  do {                                                                     \
    if ((a).device() != (b).device()) {                                    \
      TensorOperations::error(__FUNCTION__, TensorError_DeviceNotAligned); \
      return ret;                                                          \
    }                                                                      \
  } while (0)

#define TENSOR_CHECK_EMPTY_RET(t, ret)                                \
  do {                                                                \
    if ((t).empty()) {                                                \
      TensorOperations::error(__FUNCTION__, TensorError_EmptyTensor); \
      return ret;                                                     \
    }                                                                 \
  } while (0)

Storage::Storage(size_t nbytes, Device device) {
  nbytes_ = nbytes;
  ops_ = getOps(device);
  if (ops_ == nullptr) {
    TensorOperations::error(__FUNCTION__, TensorError_InvalidDevice);
  } else {
    ops_->allocate(reinterpret_cast<void **>(&data_), nbytes);
  }
}

Storage::~Storage() {
  if (ops_ && data_) {
    ops_->deallocate(data_);
    data_ = nullptr;
  }
}

TensorOperations *Storage::getOps(Device device) {
#ifdef USE_CUDA
  static auto opsCUDA_ = std::make_shared<TensorOpsCUDA>();
#endif
  static auto opsCPU_ = std::make_shared<TensorOpsCPU>();
  switch (device) {
    case Device::CUDA:
#ifdef USE_CUDA
      return opsCUDA_.get();
#else
      return nullptr;
#endif
    default:
      break;
  }
  return opsCPU_.get();
}

Device TensorImpl::defaultDevice_ = Device::CPU;

TensorImpl::TensorImpl(const TensorImpl &other) { shareFrom(other); }

TensorImpl::TensorImpl(TensorImpl &&other) noexcept {
  moveFrom(std::move(other));
}

TensorImpl &TensorImpl::operator=(const TensorImpl &other) {
  if (this != &other) {
    shareFrom(other);
  }
  return *this;
}

TensorImpl &TensorImpl::operator=(TensorImpl &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  moveFrom(std::move(other));
  return *this;
}

TensorImpl::TensorImpl(const Array1d &values1d, Device device) {
  device_ = device;
  shape_ = {static_cast<int32_t>(values1d.size())};
  initMeta();
  initData(values1d.data());
}

TensorImpl::TensorImpl(const Array2d &values2d, Device device) {
  device_ = device;
  shape_ = {static_cast<int32_t>(values2d.size()),
            static_cast<int32_t>(values2d[0].size())};
  initMeta();
  initData();
  for (int32_t idx = 0; idx < shape_[0]; idx++) {
    ops_->copyHostToDevice(data_ + idx * strides_[0], values2d[idx].data(),
                           values2d[idx].size() * sizeof(float));
  }
}

TensorImpl::TensorImpl(const Array3d &values3d, Device device) {
  device_ = device;
  shape_ = {static_cast<int32_t>(values3d.size()),
            static_cast<int32_t>(values3d[0].size()),
            static_cast<int32_t>(values3d[0][0].size())};
  initMeta();
  initData();
  for (int32_t idx = 0; idx < shape_[0]; idx++) {
    for (int32_t k = 0; k < shape_[1]; k++) {
      ops_->copyHostToDevice(data_ + idx * strides_[0] + k * strides_[1],
                             values3d[idx][k].data(),
                             values3d[idx][k].size() * sizeof(float));
    }
  }
}

void TensorImpl::initMeta() {
  dimCount_ = static_cast<int32_t>(shape_.size());
  elemCount_ = 1;
  strides_.resize(dimCount_);
  for (auto dim = dimCount_ - 1; dim >= 0; dim--) {
    strides_[dim] = elemCount_;
    elemCount_ *= shape_[dim];
  }
}

void TensorImpl::initMeta(const TensorImpl &other) {
  dimCount_ = other.dimCount_;
  elemCount_ = other.elemCount_;
  shape_ = other.shape_;
  strides_ = other.strides_;

  device_ = other.device_;
}

void TensorImpl::initData(const float *ptr, Device device) {
  if (elemCount_ == 0) {
    return;
  }
  storage_ = std::make_shared<Storage>(sizeof(float) * elemCount_, device_);
  data_ = storage_->data_;
  ops_ = storage_->ops_;
  if (ptr) {
    copyToDevice(data_, ptr, storage_->nbytes_, device);
  }
}

void TensorImpl::cow() {
  if (storage_ && storage_.use_count() > 1) {
    auto oldData = data_;
    storage_ = std::make_shared<Storage>(sizeof(float) * elemCount_, device_);
    data_ = storage_->data_;
    ops_ = storage_->ops_;
    ops_->copyOnDevice(data_, oldData, storage_->nbytes_);
  }
}

void TensorImpl::shareFrom(const TensorImpl &other) {
  dimCount_ = other.dimCount_;
  elemCount_ = other.elemCount_;
  shape_ = other.shape_;
  strides_ = other.strides_;

  data_ = other.data_;
  device_ = other.device_;
  ops_ = other.ops_;
  storage_ = other.storage_;
}

void TensorImpl::moveFrom(TensorImpl &&other) {
  dimCount_ = other.dimCount_;
  elemCount_ = other.elemCount_;
  shape_ = std::move(other.shape_);
  strides_ = std::move(other.strides_);

  data_ = other.data_;
  device_ = other.device_;
  ops_ = other.ops_;
  storage_ = std::move(other.storage_);

  other.dimCount_ = 0;
  other.elemCount_ = 0;
  other.data_ = nullptr;
  other.ops_ = nullptr;
}

void TensorImpl::copyToDevice(void *dst, const void *src, size_t count,
                              Device srcDevice) {
  if (srcDevice == Device::CPU) {
    ops_->copyHostToDevice(dst, src, count);
  } else {
    ops_->copyOnDevice(dst, src, count);
  }
}

TensorImpl TensorImpl::shape(const Shape &s, Device device) {
  TensorImpl ret;
  ret.device_ = device;
  ret.shape_ = s;
  ret.initMeta();
  ret.initData();
  return ret;
}

TensorImpl TensorImpl::scalar(Device device) { return shape({}, device); }

TensorImpl TensorImpl::scalar(const float &value, Device device) {
  TensorImpl ret = shape({}, device);
  ret.ops_->fillConstant_(ret.data_, value, 1);
  return ret;
}

TensorImpl TensorImpl::ones(const Shape &s, Device device) {
  TensorImpl ret = shape(s, device);
  ret.ops_->fillConstant_(ret, 1.f);
  return ret;
}

TensorImpl TensorImpl::onesLike(const TensorImpl &t, Device device) {
  return ones(t.shape(), device);
}

TensorImpl TensorImpl::zeros(const Shape &s, Device device) {
  TensorImpl ret = shape(s, device);
  ret.ops_->fillConstant_(ret, 0.f);
  return ret;
}

TensorImpl TensorImpl::zerosLike(const TensorImpl &t, Device device) {
  return zeros(t.shape(), device);
}

TensorImpl TensorImpl::rand(const Shape &s, Device device) {
  TensorImpl ret = shape(s, device);
  ret.ops_->fillRandUniform_(ret, 0.f, 1.f);
  return ret;
}

TensorImpl TensorImpl::uniform(const Shape &s, float min, float max,
                               Device device) {
  TensorImpl ret = shape(s, device);
  ret.ops_->fillRandUniform_(ret, min, max);
  return ret;
}

TensorImpl TensorImpl::randn(const Shape &s, Device device) {
  TensorImpl ret = shape(s, device);
  ret.ops_->fillRandNormal_(ret);
  return ret;
}

TensorImpl TensorImpl::bernoulli(const Shape &s, float p, Device device) {
  TensorImpl ret = shape(s, device);
  ret.ops_->fillRandBernoulli_(ret, p);
  return ret;
}

TensorImpl TensorImpl::arange(float start, float stop, float step,
                              Device device) {
  auto steps = static_cast<int32_t>(std::ceil((stop - start) / step));
  auto ret = shape({steps}, device);
  ret.ops_->fillLinSpace_(ret.data_, start, step, steps);
  return ret;
}

TensorImpl TensorImpl::linspace(float start, float end, int32_t steps,
                                Device device) {
  assert(steps > 0);
  float step = 0;
  if (steps > 1) {
    step = (end - start) / static_cast<float>(steps - 1);
  }
  auto ret = shape({steps}, device);
  ret.ops_->fillLinSpace_(ret.data_, start, step, steps);
  return ret;
}

TensorImpl TensorImpl::to(Device device) {
  TENSOR_DEVICE_AVAILABLE(device, {});
  if (device_ == device) {
    return *this;
  }

  TensorImpl ret;
  ret.dimCount_ = dimCount_;
  ret.elemCount_ = elemCount_;
  ret.shape_ = shape_;
  ret.strides_ = strides_;
  ret.device_ = device;
  ret.initData();

  if (!empty()) {
    if (device == Device::CPU) {
      ops_->copyDeviceToHost(ret.data_, data_, elemCount_ * sizeof(float));
    } else {
      ret.ops_->copyHostToDevice(ret.data_, data_, elemCount_ * sizeof(float));
    }
  }
  return ret;
}

void TensorImpl::to_(Device device) {
  TENSOR_DEVICE_AVAILABLE(device, );
  if (device_ == device) {
    return;
  }
  auto oldStorage = storage_;
  auto oldData = data_;

  device_ = device;
  initData();

  if (!empty()) {
    if (device == Device::CPU) {
      oldStorage->ops_->copyDeviceToHost(data_, oldData,
                                         elemCount_ * sizeof(float));
    } else {
      ops_->copyHostToDevice(data_, oldData, elemCount_ * sizeof(float));
    }
  }
}

std::vector<float> TensorImpl::toList() const {
  if (device_ == Device::CPU) {
    return {data_, data_ + elemCount_};
  }

  if (device_ == Device::CUDA) {
    std::vector<float> hostData(elemCount_);
    ops_->copyDeviceToHost(&hostData[0], data_, elemCount_ * sizeof(float));
    return hostData;
  }

  return {};
}

float TensorImpl::item() const {
  assert(elemCount_ == 1);
  if (device_ == Device::CPU) {
    return data_[0];
  }
  if (device_ == Device::CUDA) {
    float ret;
    ops_->copyDeviceToHost(&ret, data_, sizeof(float));
    return ret;
  }
  return 0.f;
}

void TensorImpl::reshape_(const Shape &shape) {
  // set scalar
  if (shape.empty() && elemCount_ == 1) {
    dimCount_ = 0;
    shape_.clear();
    strides_.clear();
    return;
  }

  shape_.resize(shape.size());

  int32_t inferredIdx = -1;
  int32_t cnt = 1;
  for (int32_t i = 0; i < shape.size(); i++) {
    if (shape[i] == -1) {
      if (inferredIdx >= 0) {
        TensorOperations::error(__FUNCTION__, TensorError_InvalidShape);
        return;
      }
      inferredIdx = i;
      shape_[i] = 0;
    } else {
      cnt *= shape[i];
      shape_[i] = shape[i];
    }
  }
  if (inferredIdx >= 0) {
    shape_[inferredIdx] = elemCount_ / cnt;
  }

  initMeta();
}

TensorImpl TensorImpl::reshape(const TensorImpl &t, const Shape &shape) {
  TensorImpl ret = t;
  ret.reshape_(shape);
  return ret;
}

void TensorImpl::flatten_(int32_t startDim, int32_t endDim) {
  Shape retShape;
  for (int32_t i = 0; i < startDim; i++) {
    retShape.push_back(shape_[i]);
  }
  int32_t flattenDims = 1;
  if (endDim < 0) {
    endDim = dimCount_ - 1;
  }
  for (int32_t i = startDim; i <= endDim; i++) {
    flattenDims *= shape_[i];
  }
  retShape.push_back(flattenDims);
  for (int32_t i = endDim + 1; i < dimCount_; i++) {
    retShape.push_back(shape_[i]);
  }

  reshape_(retShape);
}

TensorImpl TensorImpl::flatten(const TensorImpl &t, int32_t startDim,
                               int32_t endDim) {
  TensorImpl ret = t;
  ret.flatten_(startDim, endDim);
  return ret;
}

void TensorImpl::unflatten_(int32_t dim, const std::vector<int32_t> &sizes) {
  if (dim < 0) {
    dim += dimCount_;
  }
  Shape retShape;
  for (int32_t i = 0; i < dim; i++) {
    retShape.push_back(shape_[i]);
  }
  int32_t unflattenDims = 1;
  int32_t inferredIdx = -1;
  for (int32_t i = 0; i < sizes.size(); i++) {
    if (sizes[i] == -1) {
      inferredIdx = dim + i;
      retShape.push_back(0);
    } else {
      unflattenDims *= sizes[i];
      retShape.push_back(sizes[i]);
    }
  }
  if (inferredIdx >= 0) {
    retShape[inferredIdx] = shape_[dim] / unflattenDims;
  } else if (unflattenDims != shape_[dim]) {
    TensorOperations::error(__FUNCTION__, TensorError_InvalidAxis);
    return;
  }
  for (int32_t i = dim + 1; i < dimCount_; i++) {
    retShape.push_back(shape_[i]);
  }
  reshape_(retShape);
}

TensorImpl TensorImpl::unflatten(const TensorImpl &t, int32_t dim,
                                 const std::vector<int32_t> &sizes) {
  TensorImpl ret = t;
  ret.unflatten_(dim, sizes);
  return ret;
}

void TensorImpl::squeeze_(int32_t dim) {
  if (dim >= dimCount_) {
    return;
  }
  if (dim >= 0 && shape_[dim] != 1) {
    return;
  }
  Shape retShape;
  if (dim >= 0) {
    for (int32_t i = 0; i < dim; i++) {
      retShape.push_back(shape_[i]);
    }
    for (int32_t i = dim + 1; i < dimCount_; i++) {
      retShape.push_back(shape_[i]);
    }
  } else {
    for (auto d : shape_) {
      if (d != 1) {
        retShape.push_back(d);
      }
    }
  }
  reshape_(retShape);
}

void TensorImpl::squeeze_(const std::vector<int32_t> &dims) {
  for (const auto d : dims) {
    squeeze_(d);
  }
}

TensorImpl TensorImpl::squeeze(const TensorImpl &t, int32_t dim) {
  TensorImpl ret = t;
  ret.squeeze_(dim);
  return ret;
}

TensorImpl TensorImpl::squeeze(const TensorImpl &t,
                               const std::vector<int32_t> &dims) {
  TensorImpl ret = t;
  ret.squeeze_(dims);
  return ret;
}

void TensorImpl::unsqueeze_(int32_t dim) {
  if (dim > dimCount_ || dim < -dimCount_ - 1) {
    TensorOperations::error(__FUNCTION__, TensorError_InvalidAxis);
    return;
  }
  if (dim < 0) {
    dim += dimCount_ + 1;
  }
  Shape retShape;
  for (int32_t i = 0; i < dim; i++) {
    retShape.push_back(shape_[i]);
  }
  retShape.push_back(1);
  for (int32_t i = dim; i < dimCount_; i++) {
    retShape.push_back(shape_[i]);
  }
  reshape_(retShape);
}

TensorImpl TensorImpl::unsqueeze(const TensorImpl &t, int32_t dim) {
  TensorImpl ret = t;
  ret.unsqueeze_(dim);
  return ret;
}

void TensorImpl::fill_(float value) {
  cow();
  ops_->fillConstant_(*this, value);
}

void TensorImpl::fillUniform_(float min, float max) {
  cow();
  ops_->fillRandUniform_(*this, min, max);
}

TensorImpl TensorImpl::operator+(const TensorImpl &other) const {
  TENSOR_CHECK_EMPTY_RET(*this, {});
  TENSOR_CHECK_EMPTY_RET(other, {});
  TENSOR_CHECK_DEVICE_RET(*this, other, {});
  return ops_->add(*this, other);
}

TensorImpl TensorImpl::operator-(const TensorImpl &other) const {
  TENSOR_CHECK_EMPTY_RET(*this, {});
  TENSOR_CHECK_EMPTY_RET(other, {});
  TENSOR_CHECK_DEVICE_RET(*this, other, {});
  return ops_->sub(*this, other);
}

TensorImpl TensorImpl::operator*(const TensorImpl &other) const {
  TENSOR_CHECK_EMPTY_RET(*this, {});
  TENSOR_CHECK_EMPTY_RET(other, {});
  TENSOR_CHECK_DEVICE_RET(*this, other, {});
  return ops_->mul(*this, other);
}

TensorImpl TensorImpl::operator/(const TensorImpl &other) const {
  TENSOR_CHECK_EMPTY_RET(*this, {});
  TENSOR_CHECK_EMPTY_RET(other, {});
  TENSOR_CHECK_DEVICE_RET(*this, other, {});
  return ops_->div(*this, other);
}

TensorImpl TensorImpl::operator+(const float &other) const {
  TENSOR_CHECK_EMPTY_RET(*this, {});
  return ops_->add(*this, other);
}

TensorImpl TensorImpl::operator-(const float &other) const {
  TENSOR_CHECK_EMPTY_RET(*this, {});
  return ops_->sub(*this, other);
}

TensorImpl TensorImpl::operator*(const float &other) const {
  TENSOR_CHECK_EMPTY_RET(*this, {});
  return ops_->mul(*this, other);
}

TensorImpl TensorImpl::operator/(const float &other) const {
  TENSOR_CHECK_EMPTY_RET(*this, {});
  return ops_->div(*this, other);
}

TensorImpl operator+(const float &other, const TensorImpl &obj) {
  TENSOR_CHECK_EMPTY_RET(obj, {});
  return obj.ops_->add(other, obj);
}

TensorImpl operator-(const float &other, const TensorImpl &obj) {
  TENSOR_CHECK_EMPTY_RET(obj, {});
  return obj.ops_->sub(other, obj);
}

TensorImpl operator*(const float &other, const TensorImpl &obj) {
  TENSOR_CHECK_EMPTY_RET(obj, {});
  return obj.ops_->mul(other, obj);
}

TensorImpl operator/(const float &other, const TensorImpl &obj) {
  TENSOR_CHECK_EMPTY_RET(obj, {});
  return obj.ops_->div(other, obj);
}

void TensorImpl::operator+=(const TensorImpl &other) {
  TENSOR_CHECK_EMPTY_RET(*this, );
  TENSOR_CHECK_EMPTY_RET(other, );
  TENSOR_CHECK_DEVICE_RET(*this, other, );
  cow();
  ops_->add_(*this, other);
}

void TensorImpl::operator-=(const TensorImpl &other) {
  TENSOR_CHECK_EMPTY_RET(*this, );
  TENSOR_CHECK_EMPTY_RET(other, );
  TENSOR_CHECK_DEVICE_RET(*this, other, );
  cow();
  ops_->sub_(*this, other);
}

void TensorImpl::operator*=(const TensorImpl &other) {
  TENSOR_CHECK_EMPTY_RET(*this, );
  TENSOR_CHECK_EMPTY_RET(other, );
  TENSOR_CHECK_DEVICE_RET(*this, other, );
  cow();
  ops_->mul_(*this, other);
}

void TensorImpl::operator/=(const TensorImpl &other) {
  TENSOR_CHECK_EMPTY_RET(*this, );
  TENSOR_CHECK_EMPTY_RET(other, );
  TENSOR_CHECK_DEVICE_RET(*this, other, );
  cow();
  ops_->div_(*this, other);
}

void TensorImpl::operator+=(const float &other) {
  TENSOR_CHECK_EMPTY_RET(*this, );
  cow();
  ops_->add_(*this, other);
}

void TensorImpl::operator-=(const float &other) {
  TENSOR_CHECK_EMPTY_RET(*this, );
  cow();
  ops_->sub_(*this, other);
}

void TensorImpl::operator*=(const float &other) {
  TENSOR_CHECK_EMPTY_RET(*this, );
  cow();
  ops_->mul_(*this, other);
}

void TensorImpl::operator/=(const float &other) {
  TENSOR_CHECK_EMPTY_RET(*this, );
  cow();
  ops_->div_(*this, other);
}

TensorImpl TensorImpl::pow(const TensorImpl &other) const {
  TENSOR_CHECK_EMPTY_RET(*this, {});
  TENSOR_CHECK_EMPTY_RET(other, {});
  TENSOR_CHECK_DEVICE_RET(*this, other, {});
  return ops_->pow(*this, other);
}

TensorImpl TensorImpl::pow(const float &other) const {
  TENSOR_CHECK_EMPTY_RET(*this, {});
  return ops_->pow(*this, other);
}

void TensorImpl::sin_() {
  TENSOR_CHECK_EMPTY_RET(*this, );
  cow();
  ops_->sin_(*this);
}

void TensorImpl::cos_() {
  TENSOR_CHECK_EMPTY_RET(*this, );
  cow();
  ops_->cos_(*this);
}

void TensorImpl::sqrt_() {
  TENSOR_CHECK_EMPTY_RET(*this, );
  cow();
  ops_->sqrt_(*this);
}

void TensorImpl::tanh_() {
  TENSOR_CHECK_EMPTY_RET(*this, );
  cow();
  ops_->tanh_(*this);
}

void TensorImpl::exp_() {
  TENSOR_CHECK_EMPTY_RET(*this, );
  cow();
  ops_->exp_(*this);
}

void TensorImpl::log_() {
  TENSOR_CHECK_EMPTY_RET(*this, );
  cow();
  ops_->log_(*this);
}

TensorImpl TensorImpl::sin(const TensorImpl &t) {
  TENSOR_CHECK_EMPTY_RET(t, {});
  return t.ops_->sin(t);
}

TensorImpl TensorImpl::cos(const TensorImpl &t) {
  TENSOR_CHECK_EMPTY_RET(t, {});
  return t.ops_->cos(t);
}

TensorImpl TensorImpl::sqrt(const TensorImpl &t) {
  TENSOR_CHECK_EMPTY_RET(t, {});
  return t.ops_->sqrt(t);
}

TensorImpl TensorImpl::tanh(const TensorImpl &t) {
  TENSOR_CHECK_EMPTY_RET(t, {});
  return t.ops_->tanh(t);
}

TensorImpl TensorImpl::exp(const TensorImpl &t) {
  TENSOR_CHECK_EMPTY_RET(t, {});
  return t.ops_->exp(t);
}

TensorImpl TensorImpl::log(const TensorImpl &t) {
  TENSOR_CHECK_EMPTY_RET(t, {});
  return t.ops_->log(t);
}

TensorImpl TensorImpl::operator<(const TensorImpl &other) const {
  TENSOR_CHECK_EMPTY_RET(*this, {});
  TENSOR_CHECK_EMPTY_RET(other, {});
  TENSOR_CHECK_DEVICE_RET(*this, other, {});
  return ops_->lt(*this, other);
}

TensorImpl TensorImpl::operator<=(const TensorImpl &other) const {
  TENSOR_CHECK_EMPTY_RET(*this, {});
  TENSOR_CHECK_EMPTY_RET(other, {});
  TENSOR_CHECK_DEVICE_RET(*this, other, {});
  return ops_->le(*this, other);
}

TensorImpl TensorImpl::operator>(const TensorImpl &other) const {
  TENSOR_CHECK_EMPTY_RET(*this, {});
  TENSOR_CHECK_EMPTY_RET(other, {});
  TENSOR_CHECK_DEVICE_RET(*this, other, {});
  return ops_->gt(*this, other);
}

TensorImpl TensorImpl::operator>=(const TensorImpl &other) const {
  TENSOR_CHECK_EMPTY_RET(*this, {});
  TENSOR_CHECK_EMPTY_RET(other, {});
  TENSOR_CHECK_DEVICE_RET(*this, other, {});
  return ops_->ge(*this, other);
}

TensorImpl TensorImpl::operator==(const TensorImpl &other) const {
  TENSOR_CHECK_EMPTY_RET(*this, {});
  TENSOR_CHECK_EMPTY_RET(other, {});
  TENSOR_CHECK_DEVICE_RET(*this, other, {});
  return ops_->eq(*this, other);
}

TensorImpl TensorImpl::operator!=(const TensorImpl &other) const {
  TENSOR_CHECK_EMPTY_RET(*this, {});
  TENSOR_CHECK_EMPTY_RET(other, {});
  TENSOR_CHECK_DEVICE_RET(*this, other, {});
  return ops_->ne(*this, other);
}

TensorImpl TensorImpl::operator<(const float &other) const {
  TENSOR_CHECK_EMPTY_RET(*this, {});
  return ops_->lt(*this, other);
}

TensorImpl TensorImpl::operator<=(const float &other) const {
  TENSOR_CHECK_EMPTY_RET(*this, {});
  return ops_->le(*this, other);
}

TensorImpl TensorImpl::operator>(const float &other) const {
  TENSOR_CHECK_EMPTY_RET(*this, {});
  return ops_->gt(*this, other);
}

TensorImpl TensorImpl::operator>=(const float &other) const {
  TENSOR_CHECK_EMPTY_RET(*this, {});
  return ops_->ge(*this, other);
}

TensorImpl TensorImpl::operator==(const float &other) const {
  TENSOR_CHECK_EMPTY_RET(*this, {});
  return ops_->eq(*this, other);
}

TensorImpl TensorImpl::operator!=(const float &other) const {
  TENSOR_CHECK_EMPTY_RET(*this, {});
  return ops_->ne(*this, other);
}

TensorImpl TensorImpl::maximum(const TensorImpl &a, const TensorImpl &b) {
  TENSOR_CHECK_EMPTY_RET(a, {});
  TENSOR_CHECK_EMPTY_RET(b, {});
  TENSOR_CHECK_DEVICE_RET(a, b, {});
  return a.ops_->maximum(a, b);
}

TensorImpl TensorImpl::minimum(const TensorImpl &a, const TensorImpl &b) {
  TENSOR_CHECK_EMPTY_RET(a, {});
  TENSOR_CHECK_EMPTY_RET(b, {});
  TENSOR_CHECK_DEVICE_RET(a, b, {});
  return a.ops_->minimum(a, b);
}

void TensorImpl::clampMin_(float min) {
  TENSOR_CHECK_EMPTY_RET(*this, );
  cow();
  ops_->clampMin_(*this, min);
}

void TensorImpl::clampMax_(float max) {
  TENSOR_CHECK_EMPTY_RET(*this, );
  cow();
  ops_->clampMax_(*this, max);
}

void TensorImpl::clamp_(float min, float max) {
  TENSOR_CHECK_EMPTY_RET(*this, );
  cow();
  ops_->clamp_(*this, min, max);
}

TensorImpl TensorImpl::clampMin(const TensorImpl &t, float min) {
  TENSOR_CHECK_EMPTY_RET(t, {});
  return t.ops_->clampMin(t, min);
}

TensorImpl TensorImpl::clampMax(const TensorImpl &t, float max) {
  TENSOR_CHECK_EMPTY_RET(t, {});
  return t.ops_->clampMax(t, max);
}

TensorImpl TensorImpl::clamp(const TensorImpl &t, float min, float max) {
  TENSOR_CHECK_EMPTY_RET(t, {});
  return t.ops_->clamp(t, min, max);
}

TensorImpl TensorImpl::min(const TensorImpl &t) {
  TENSOR_CHECK_EMPTY_RET(t, {});
  return t.ops_->min(t);
}

TensorImpl TensorImpl::max(const TensorImpl &t) {
  TENSOR_CHECK_EMPTY_RET(t, {});
  return t.ops_->max(t);
}

TensorImpl TensorImpl::argmin(const TensorImpl &t) {
  TENSOR_CHECK_EMPTY_RET(t, {});
  return t.ops_->argmin(t);
}

TensorImpl TensorImpl::argmax(const TensorImpl &t) {
  TENSOR_CHECK_EMPTY_RET(t, {});
  return t.ops_->argmax(t);
}

TensorImpl TensorImpl::sum(const TensorImpl &t) {
  TENSOR_CHECK_EMPTY_RET(t, {});
  return t.ops_->sum(t);
}

TensorImpl TensorImpl::mean(const TensorImpl &t) {
  TENSOR_CHECK_EMPTY_RET(t, {});
  return t.ops_->mean(t);
}

TensorImpl TensorImpl::var(const TensorImpl &t, bool unbiased) {
  TENSOR_CHECK_EMPTY_RET(t, {});
  return t.ops_->varMean(t, unbiased).first;
}

std::pair<TensorImpl, TensorImpl> TensorImpl::varMean(const TensorImpl &t,
                                                      bool unbiased) {
  TENSOR_CHECK_EMPTY_RET(t, {});
  return t.ops_->varMean(t, unbiased);
}

std::pair<TensorImpl, TensorImpl> TensorImpl::min(const TensorImpl &t,
                                                  int32_t dim, bool keepDims) {
  TENSOR_CHECK_EMPTY_RET(t, {});
  return t.ops_->min(t, dim, keepDims);
}

std::pair<TensorImpl, TensorImpl> TensorImpl::max(const TensorImpl &t,
                                                  int32_t dim, bool keepDims) {
  TENSOR_CHECK_EMPTY_RET(t, {});
  return t.ops_->max(t, dim, keepDims);
}

TensorImpl TensorImpl::argmin(const TensorImpl &t, int32_t dim, bool keepDims) {
  TENSOR_CHECK_EMPTY_RET(t, {});
  return t.ops_->min(t, dim, keepDims).second;
}

TensorImpl TensorImpl::argmax(const TensorImpl &t, int32_t dim, bool keepDims) {
  TENSOR_CHECK_EMPTY_RET(t, {});
  return t.ops_->max(t, dim, keepDims).second;
}

TensorImpl TensorImpl::sum(const TensorImpl &t, int32_t dim, bool keepDims) {
  TENSOR_CHECK_EMPTY_RET(t, {});
  return t.ops_->sum(t, {dim}, keepDims);
}

TensorImpl TensorImpl::mean(const TensorImpl &t, int32_t dim, bool keepDims) {
  TENSOR_CHECK_EMPTY_RET(t, {});
  return t.ops_->mean(t, {dim}, keepDims);
}

TensorImpl TensorImpl::var(const TensorImpl &t, int32_t dim, bool unbiased,
                           bool keepDims) {
  TENSOR_CHECK_EMPTY_RET(t, {});
  return t.ops_->varMean(t, {dim}, unbiased, keepDims).first;
}

std::pair<TensorImpl, TensorImpl> TensorImpl::varMean(const TensorImpl &t,
                                                      int32_t dim,
                                                      bool unbiased,
                                                      bool keepDims) {
  TENSOR_CHECK_EMPTY_RET(t, {});
  return t.ops_->varMean(t, {dim}, unbiased, keepDims);
}

TensorImpl TensorImpl::sum(const TensorImpl &t,
                           const std::vector<int32_t> &dims, bool keepDims) {
  TENSOR_CHECK_EMPTY_RET(t, {});
  return t.ops_->sum(t, dims, keepDims);
}

TensorImpl TensorImpl::mean(const TensorImpl &t,
                            const std::vector<int32_t> &dims, bool keepDims) {
  TENSOR_CHECK_EMPTY_RET(t, {});
  return t.ops_->mean(t, dims, keepDims);
}

TensorImpl TensorImpl::var(const TensorImpl &t,
                           const std::vector<int32_t> &dims, bool unbiased,
                           bool keepDims) {
  TENSOR_CHECK_EMPTY_RET(t, {});
  return t.ops_->varMean(t, dims, unbiased, keepDims).first;
}

std::pair<TensorImpl, TensorImpl> TensorImpl::varMean(
    const TensorImpl &t, const std::vector<int32_t> &dims, bool unbiased,
    bool keepDims) {
  TENSOR_CHECK_EMPTY_RET(t, {});
  return t.ops_->varMean(t, dims, unbiased, keepDims);
}

void TensorImpl::transpose_(int32_t dim0, int32_t dim1) {
  TENSOR_CHECK_EMPTY_RET(*this, );
  auto ret = transpose(*this, dim0, dim1);
  if (!ret.empty()) {
    *this = std::move(ret);
  }
}

TensorImpl TensorImpl::transpose(const TensorImpl &t, int32_t dim0,
                                 int32_t dim1) {
  TENSOR_CHECK_EMPTY_RET(t, {});
  if (dim0 < 0) {
    dim0 += t.dimCount_;
  }
  if (dim1 < 0) {
    dim1 += t.dimCount_;
  }
  if (dim0 < 0 || dim0 >= t.dimCount_ || dim1 < 0 || dim1 >= t.dimCount_) {
    TensorOperations::error(__FUNCTION__, TensorError_InvalidAxis);
    return {};
  }

  std::vector<int32_t> dims(t.dimCount_);
  std::iota(dims.begin(), dims.end(), 0);
  dims[dim0] = dim1;
  dims[dim1] = dim0;
  return t.ops_->permute(t, dims);
}

void TensorImpl::t_() {
  TENSOR_CHECK_EMPTY_RET(*this, );
  if (dimCount_ < 2) {
    return;
  }
  if (dimCount_ > 2) {
    TensorOperations::error(__FUNCTION__, TensorError_InvalidShape);
    return;
  }
  auto ret = t(*this);
  if (!ret.empty()) {
    *this = std::move(ret);
  }
}

TensorImpl TensorImpl::t(const TensorImpl &t) {
  TENSOR_CHECK_EMPTY_RET(t, {});
  if (t.dimCount_ < 2) {
    return {};
  }
  if (t.dimCount_ > 2) {
    TensorOperations::error(__FUNCTION__, TensorError_InvalidShape);
    return {};
  }
  return t.ops_->transpose2D(t);
}

TensorImpl TensorImpl::permute(const std::vector<int32_t> &dims) const {
  TENSOR_CHECK_EMPTY_RET(*this, {});
  if (!dims.empty() && dims.size() != dimCount_) {
    TensorOperations::error(__FUNCTION__, TensorError_InvalidShape);
    return {};
  }
  std::vector<int32_t> retDims;
  retDims.reserve(dimCount_);
  if (dims.empty()) {
    for (int32_t i = dimCount_ - 1; i >= 0; i--) {
      retDims.push_back(i);
    }
  } else {
    for (int32_t i = 0; i < dimCount_; i++) {
      auto d = dims[i];
      if (d < 0) {
        d += dimCount_;
      }
      if (d < 0 || d >= dimCount_) {
        TensorOperations::error(__FUNCTION__, TensorError_InvalidAxis);
        return {};
      }
      retDims.push_back(d);
    }
  }
  // 2D transpose
  if (retDims.size() == 2 && retDims[0] == 1 && retDims[1] == 0) {
    return ops_->transpose2D(*this);
  }
  return ops_->permute(*this, retDims);
}

TensorImpl TensorImpl::index(const std::vector<int32_t> &indices) const {
  TENSOR_CHECK_EMPTY_RET(*this, {});
  auto len = static_cast<int32_t>(indices.size());
  int32_t dataIdx = 0;
  for (int32_t i = 0; i < len; i++) {
    auto idx = indices[i];
    dataIdx += (idx >= 0 ? idx : idx + shape_[i]) * strides_[i];
  }
  int32_t dimStride = strides_[len - 1];
  Shape retShape;
  retShape.reserve(dimCount_);
  for (int32_t i = len; i < dimCount_; i++) {
    retShape.push_back(shape_[i]);
  }
  auto retTensor = shape(retShape, device_);
  assert(dimStride == retTensor.elemCount_);
  ops_->copyOnDevice(retTensor.data_, &data_[dataIdx],
                     dimStride * sizeof(float));
  return retTensor;
}

TensorImpl TensorImpl::index(
    const std::vector<std::reference_wrapper<TensorImpl>> &indices) const {
  TENSOR_CHECK_EMPTY_RET(*this, {});
  for (auto &t : indices) {
    if (t.get().device_ != device_) {
      TensorOperations::error(__FUNCTION__, TensorError_DeviceNotAligned);
      return {};
    }
  }
  return ops_->index(*this, indices);
}

void TensorImpl::indexPut_(const std::vector<int32_t> &indices, float val) {
  TENSOR_CHECK_EMPTY_RET(*this, );
  cow();
  auto len = static_cast<int32_t>(indices.size());
  int32_t dataIdx = 0;
  for (int32_t i = 0; i < len; i++) {
    auto idx = indices[i];
    dataIdx += (idx >= 0 ? idx : idx + shape_[i]) * strides_[i];
  }
  int32_t dimStride = strides_[len - 1];
  ops_->fillConstant_(&data_[dataIdx], val, dimStride);
}

void TensorImpl::indexPut_(const std::vector<int32_t> &indices,
                           const TensorImpl &val) {
  TENSOR_CHECK_EMPTY_RET(*this, );
  cow();
  auto len = static_cast<int32_t>(indices.size());
  int32_t dataIdx = 0;
  for (int32_t i = 0; i < len; i++) {
    auto idx = indices[i];
    dataIdx += (idx >= 0 ? idx : idx + shape_[i]) * strides_[i];
  }
  int32_t dimStride = strides_[len - 1];
  assert(val.elemCount_ == dimStride);
  copyToDevice(&data_[dataIdx], val.data_, dimStride * sizeof(float),
               val.device_);
}

void TensorImpl::indexPut_(
    const std::vector<std::reference_wrapper<TensorImpl>> &indices, float val) {
  TENSOR_CHECK_EMPTY_RET(*this, );
  for (auto &t : indices) {
    if (t.get().device_ != device_) {
      TensorOperations::error(__FUNCTION__, TensorError_DeviceNotAligned);
      return;
    }
  }
  cow();
  ops_->indexPut_(*this, indices, val);
}

void TensorImpl::indexPut_(
    const std::vector<std::reference_wrapper<TensorImpl>> &indices,
    const TensorImpl &val) {
  TENSOR_CHECK_EMPTY_RET(*this, );
  for (auto &t : indices) {
    if (t.get().device_ != device_) {
      TensorOperations::error(__FUNCTION__, TensorError_DeviceNotAligned);
      return;
    }
  }
  cow();
  ops_->indexPut_(*this, indices, val);
}

TensorImpl TensorImpl::tril(int32_t diagonal) const {
  TENSOR_CHECK_EMPTY_RET(*this, {});
  if (dimCount_ != 2) {
    TensorOperations::error(__FUNCTION__, TensorError_InvalidShape);
    return {};
  }
  return ops_->triangle(*this, diagonal, true);
}

TensorImpl TensorImpl::triu(int32_t diagonal) const {
  TENSOR_CHECK_EMPTY_RET(*this, {});
  if (dimCount_ != 2) {
    TensorOperations::error(__FUNCTION__, TensorError_InvalidShape);
    return {};
  }
  return ops_->triangle(*this, diagonal, false);
}

TensorImpl TensorImpl::stack(
    const std::vector<std::reference_wrapper<TensorImpl>> &tensors,
    int32_t dim) {
  TENSOR_CHECK_EMPTY_RET(tensors, {});

  // check dim
  auto &t0 = tensors[0].get();
  int32_t targetDim = dim >= 0 ? dim : dim + t0.dim() + 1;
  if (targetDim < 0 || targetDim > t0.dim()) {
    TensorOperations::error(__FUNCTION__, TensorError_InvalidAxis);
    return {};
  }

  // check device & shapes
  for (int32_t i = 1; i < tensors.size(); i++) {
    auto &t = tensors[i].get();
    if (t.device_ != t0.device_) {
      TensorOperations::error(__FUNCTION__, TensorError_DeviceNotAligned);
      return {};
    }

    if (t.shape() != t0.shape()) {
      TensorOperations::error(__FUNCTION__, TensorError_ShapeNotAligned);
      return {};
    }
  }

  // init result shape
  Shape retShape = t0.shape();
  retShape.insert(retShape.begin() + targetDim,
                  static_cast<int32_t>(tensors.size()));
  TensorImpl retTensor = shape(retShape, t0.device_);

  int32_t innerSize = 1;
  for (int32_t i = targetDim; i < t0.dimCount_; i++) {
    innerSize *= t0.shape_[i];
  }

  int32_t outerSize = 1;
  for (int32_t i = 0; i < targetDim; i++) {
    outerSize *= t0.shape_[i];
  }

  for (int32_t i = 0; i < tensors.size(); i++) {
    const auto &t = tensors[i].get();
    auto *srcPtr = t.data_;
    auto dstPtr = retTensor.data_ + i * innerSize;
    for (int32_t j = 0; j < outerSize; j++) {
      retTensor.ops_->copyOnDevice(dstPtr, srcPtr, innerSize * sizeof(float));
      srcPtr += innerSize;
      dstPtr += tensors.size() * innerSize;
    }
  }

  return retTensor;
}

TensorImpl TensorImpl::vstack(
    const std::vector<std::reference_wrapper<TensorImpl>> &tensors) {
  TENSOR_CHECK_EMPTY_RET(tensors, {});

  auto targetDim = 0;
  auto &t0 = tensors[0].get();

  // check device & shapes
  for (auto i = 1; i < tensors.size(); i++) {
    auto &t = tensors[i].get();
    if (t.device_ != t0.device_) {
      TensorOperations::error(__FUNCTION__, TensorError_DeviceNotAligned);
      return {};
    }
  }

  // check shapes
  bool shapesAligned = true;
  if (t0.dim() == 1) {
    // 1-D arrays must have the same length
    for (auto i = 1; i < tensors.size(); i++) {
      auto &t = tensors[i].get();
      if (t.shape() != t0.shape()) {
        shapesAligned = false;
        break;
      }
    }
  } else {
    shapesAligned = TensorOperations::checkShapeEqual(tensors, targetDim);
  }
  if (!shapesAligned) {
    TensorOperations::error(__FUNCTION__, TensorError_ShapeNotAligned);
    return {};
  }

  Shape retShape;
  if (t0.dim() == 1) {
    retShape = {static_cast<int32_t>(tensors.size()), t0.shape_[0]};
  } else {
    retShape = t0.shape();
    for (auto i = 1; i < tensors.size(); i++) {
      retShape[targetDim] += tensors[i].get().shape()[targetDim];
    }
  }

  auto retTensor = shape(retShape, t0.device_);

  auto dstPtr = retTensor.data_;
  for (auto &tensor : tensors) {
    const auto &t = tensor.get();
    retTensor.ops_->copyOnDevice(dstPtr, t.data_, t.elemCount_ * sizeof(float));
    dstPtr += t.elemCount_;
  }

  return retTensor;
}

TensorImpl TensorImpl::hstack(
    const std::vector<std::reference_wrapper<TensorImpl>> &tensors) {
  TENSOR_CHECK_EMPTY_RET(tensors, {});

  auto targetDim = 1;
  auto &t0 = tensors[0].get();

  // check device & shapes
  for (auto i = 1; i < tensors.size(); i++) {
    auto &t = tensors[i].get();
    if (t.device_ != t0.device_) {
      TensorOperations::error(__FUNCTION__, TensorError_DeviceNotAligned);
      return {};
    }
  }

  // check shapes
  bool shapesAligned = true;
  if (t0.dim() == 1) {
    // 1-D arrays which can be any length
    for (auto i = 1; i < tensors.size(); i++) {
      auto &t = tensors[i].get();
      if (t.dim() != t0.dim()) {
        shapesAligned = false;
        break;
      }
    }
  } else {
    shapesAligned = TensorOperations::checkShapeEqual(tensors, targetDim);
  }
  if (!shapesAligned) {
    TensorOperations::error(__FUNCTION__, TensorError_ShapeNotAligned);
    return {};
  }

  Shape retShape = t0.shape();
  if (t0.dim() == 1) {
    for (auto i = 1; i < tensors.size(); i++) {
      retShape[0] += tensors[i].get().shape()[0];
    }
  } else {
    for (auto i = 1; i < tensors.size(); i++) {
      retShape[targetDim] += tensors[i].get().shape()[targetDim];
    }
  }

  auto retTensor = shape(retShape, t0.device_);

  if (t0.dim() == 1) {
    auto dstPtr = retTensor.data_;
    for (auto &tensor : tensors) {
      const auto &t = tensor.get();
      retTensor.ops_->copyOnDevice(dstPtr, t.data_,
                                   t.elemCount_ * sizeof(float));
      dstPtr += t.elemCount_;
    }
  } else {
    int32_t innerSize = 1;
    for (int32_t i = targetDim; i < t0.dimCount_; i++) {
      innerSize *= t0.shape_[i];
    }

    int32_t outerSize = 1;
    for (int32_t i = 0; i < targetDim; i++) {
      outerSize *= t0.shape_[i];
    }

    for (int32_t i = 0; i < tensors.size(); i++) {
      const auto &t = tensors[i].get();
      auto *srcPtr = t.data_;
      auto dstPtr = retTensor.data_ + i * innerSize;
      for (int32_t j = 0; j < outerSize; j++) {
        retTensor.ops_->copyOnDevice(dstPtr, srcPtr, innerSize * sizeof(float));
        srcPtr += innerSize;
        dstPtr += tensors.size() * innerSize;
      }
    }
  }

  return retTensor;
}

std::vector<TensorImpl> TensorImpl::split(int32_t splitSize,
                                          int32_t dim) const {
  if (dim < 0) {
    dim += dimCount_;
  }
  if (dim < 0 || dim >= dimCount_) {
    TensorOperations::error(__FUNCTION__, TensorError_InvalidAxis);
    return {};
  }

  const auto dimSize = shape_[dim];
  if (splitSize <= 0 || dimSize % splitSize != 0) {
    TensorOperations::error(__FUNCTION__, TensorError_InvalidSections);
    return {};
  }

  const auto sections = dimSize / splitSize;
  std::vector<TensorImpl> retTensors(sections);

  // init shape of result tensors
  auto retShape = shape_;
  retShape[dim] = splitSize;
  for (auto i = 0; i < sections; i++) {
    retTensors[i] = shape(retShape, device_);
  }

  const auto innerStride = strides_[dim];
  const auto innerSize = innerStride * splitSize;
  const auto outerSize = elemCount_ / dimSize;

  for (auto i = 0; i < sections; i++) {
    for (auto outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      const auto srcOffset = outerIdx * dimSize * innerStride + i * innerSize;
      const auto dstOffset = outerIdx * splitSize * innerStride;
      const auto copySize = splitSize * innerStride * sizeof(float);
      ops_->copyOnDevice(retTensors[i].data_ + dstOffset, data_ + srcOffset,
                         copySize);
    }
  }
  return retTensors;
}

TensorImpl TensorImpl::dot(const TensorImpl &a, const TensorImpl &b) {
  TENSOR_CHECK_DEVICE_RET(a, b, {});

  if (a.dimCount_ != 1 || b.dimCount_ != 1) {
    TensorOperations::error(__FUNCTION__, TensorError_InvalidShape);
    return {};
  }

  if (a.elemCount_ != b.elemCount_) {
    TensorOperations::error(__FUNCTION__, TensorError_ShapeNotAligned);
    return {};
  }

  return a.ops_->dot(a, b);
}

TensorImpl TensorImpl::matmul(const TensorImpl &a, const TensorImpl &b) {
  TENSOR_CHECK_EMPTY_RET(a, {});
  TENSOR_CHECK_EMPTY_RET(b, {});
  TENSOR_CHECK_DEVICE_RET(a, b, {});

  if (a.dim() == 0 || b.dim() == 0) {
    TensorOperations::error(__FUNCTION__, TensorError_InvalidShape);
    return {};
  }

  Shape shapeA = a.shape();
  Shape shapeB = b.shape();
  bool prependA = false;
  bool appendB = false;
  if (shapeA.size() == 1) {
    shapeA.insert(shapeA.begin(), 1);
    prependA = true;
  }
  if (shapeB.size() == 1) {
    shapeB.insert(shapeB.end(), 1);
    appendB = true;
  }

  // check matrix multiplication compatible
  if (shapeA.back() != shapeB[shapeB.size() - 2]) {
    TensorOperations::error(__FUNCTION__, TensorError_ShapeNotAligned);
    return {};
  }

  // check shape broadcast compatible
  Shape retShape;
  auto compatible =
      TensorOperations::checkShapeCompatible(shapeA, shapeB, retShape, 2);
  if (compatible == ShapeCompatible_Error) {
    TensorOperations::error(__FUNCTION__, TensorError_ShapeNotAligned);
    return {};
  }

  auto retDimCnt = static_cast<int32_t>(retShape.size());
  auto m = shapeA[shapeA.size() - 2];
  auto k = shapeA.back();
  auto n = shapeB.back();

  retShape[retDimCnt - 2] = m;
  retShape[retDimCnt - 1] = n;
  TensorImpl retTensor = shape(retShape, a.device_);
  if (retDimCnt > 2) {
    // batched matrix multiply with broadcasting
    int32_t batchSize = retTensor.elemCount_ / (m * n);

    Shape aStrides = a.strides();
    Shape bStrides = b.strides();
    while (aStrides.size() < retTensor.dimCount_) {
      aStrides.insert(aStrides.begin(), 0);
    }
    while (bStrides.size() < retTensor.dimCount_) {
      bStrides.insert(bStrides.begin(), 0);
    }

    for (int32_t batch = 0; batch < batchSize; batch++) {
      int32_t aOffset = 0;
      int32_t bOffset = 0;
      int32_t tmp = batch;
      for (int i = retDimCnt - 3; i >= 0; i--) {
        int32_t index = tmp % retShape[i];
        tmp /= retShape[i];
        if (a.shape().size() > i && a.shape()[i] != 1) {
          aOffset += index * aStrides[i];
        }
        if (b.shape().size() > i && b.shape()[i] != 1) {
          bOffset += index * bStrides[i];
        }
      }

      a.ops_->gemm(retTensor.data_ + batch * m * n, a.data_ + aOffset,
                   b.data_ + bOffset, m, k, n, false, false);
    }
  } else {
    a.ops_->gemm(retTensor.data_, a.data_, b.data_, m, k, n, false, false);
    if (prependA) {
      retTensor.reshape_({n});
    }
  }

  // reduce dimension if necessary
  if (appendB) {
    if (prependA) {
      retTensor.reshape_({});
    } else {
      retTensor.reshape_({m});
    }
  }

  return retTensor;
}

TensorImpl TensorImpl::matmulTrans(const TensorImpl &a, const TensorImpl &b,
                                   bool transA, bool transB) {
  // fast path
  if (a.dim() == 2 && b.dim() == 2) {
    TENSOR_CHECK_DEVICE_RET(a, b, {});
    // a[m, k], b[k, n] -> [m, n]
    int32_t m = a.shape()[transA ? 1 : 0];
    int32_t k = a.shape()[transA ? 0 : 1];
    int32_t n = b.shape()[transB ? 0 : 1];
    if (k != b.shape()[transB ? 1 : 0]) {
      TensorOperations::error(__FUNCTION__, TensorError_ShapeNotAligned);
      return {};
    }
    TensorImpl ret = shape({m, n}, a.device_);
    a.ops_->gemm(ret.data_, a.data_, b.data_, m, k, n, transA, transB);
    return ret;
  }

  // slow path
  return matmul(transA ? a.permute() : a, transB ? b.permute() : b);
}

TensorImpl TensorImpl::im2col(Size2D kernel, Size2D stride,
                              Size2D padding) const {
  TENSOR_CHECK_EMPTY_RET(*this, {});
  return ops_->im2col(*this, kernel, stride, padding);
}

TensorImpl TensorImpl::col2im(const Shape &shape, Size2D kernelSize,
                              Size2D stride, Size2D padding) const {
  TENSOR_CHECK_EMPTY_RET(*this, {});
  return ops_->col2im(*this, shape, kernelSize, stride, padding);
}

}  // namespace tinytorch
