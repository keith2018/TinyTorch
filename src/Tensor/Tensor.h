/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <cmath>

#include "Scalar.h"
#include "TensorImpl.h"

namespace tinytorch {

struct FunctionBase;
class AutogradMeta;
class Tensor;

using TensorPair = std::pair<Tensor, Tensor>;
using TensorList = std::vector<Tensor>;
using TensorPtr = Tensor*;

class Tensor {
 public:
  Tensor() = default;
  ~Tensor() = default;

  Tensor(const Tensor&) = default;
  Tensor& operator=(const Tensor&) = default;
  Tensor(Tensor&&) noexcept = default;
  Tensor& operator=(Tensor&&) noexcept = default;

  bool equals(const Tensor& other) const { return impl_ == other.impl_ && gradMeta_ == other.gradMeta_; }

  template <typename T>
  Tensor(const Array1d<T>& values, IntArrayView shape, Options options);

  template <typename T>
  Tensor(const Array1d<T>& values, IntArrayView shape);

  template <typename T>
  explicit Tensor(const Array1d<T>& values, Options options);

  template <typename T>
  explicit Tensor(const Array1d<T>& values);

  template <typename T>
  explicit Tensor(const Array2d<T>& values, Options options);

  template <typename T>
  explicit Tensor(const Array2d<T>& values);

  template <typename T>
  explicit Tensor(const Array3d<T>& values, Options options);

  template <typename T>
  explicit Tensor(const Array3d<T>& values);

  Tensor(IntArrayView shape, Options options);
  Tensor(IntArrayView shape, Options options, const std::shared_ptr<Storage>& storage, int64_t offset);

  static Tensor empty(IntArrayView shape, Options options = {});
  static Tensor scalar(const Scalar& scalar, Options options = {});
  static Tensor ones(IntArrayView shape, Options options = {});
  static Tensor onesLike(const Tensor& t, Options options = {});
  static Tensor zeros(IntArrayView shape, Options options = {});
  static Tensor zerosLike(const Tensor& t, Options options = {});
  static Tensor rand(IntArrayView shape, Options options = {});
  static Tensor randn(IntArrayView shape, Options options = {});
  static Tensor uniform(IntArrayView shape, float min, float max, Options options = {});
  static Tensor bernoulli(IntArrayView shape, float p, Options options = {});

  template <typename T>
  static Tensor arange(T start, T stop, T step = 1, Options options = {});

  template <typename T>
  static Tensor linspace(T start, T end, int64_t steps, Options options = {});

  bool defined() const { return impl_ != nullptr; }
  TensorImpl& getImpl() const { return *impl_; }

  DType dtype() const { return impl_->dtype(); }
  Device device() const { return impl_->device(); }
  bool pinnedMemory() const { return impl_->pinnedMemory(); }
  bool requiresGrad() const { return impl_->requiresGrad(); }
  void setRequiresGrad(bool require, const std::shared_ptr<FunctionBase>& fn = nullptr);

  int64_t dim() const { return impl_->dim(); }
  int64_t numel() const { return impl_->numel(); }
  int64_t storageOffset() const { return impl_->storageOffset(); }
  bool isScalar() const { return impl_->isScalar(); }

  template <typename T = void>
  T* dataPtr() {
    return impl_->dataPtr<T>();
  }

  template <typename T = void>
  const T* dataPtr() const {
    return impl_->dataPtr<T>();
  }

  const Options& options() const { return impl_->options(); }
  IntArrayView shape() const { return impl_->shape(); }
  IntArrayView strides() const { return impl_->strides(); }
  int64_t shape(int64_t d) const { return impl_->shape(d); }
  int64_t stride(int64_t d) const { return impl_->stride(d); }
  const std::shared_ptr<Storage>& storage() const { return impl_->storage(); }

  void copyOnWrite() const { impl_->copyOnWrite(); }
  Tensor clone() const;

  std::shared_ptr<FunctionBase>& gradFn() const;
  void setGradFn(const std::shared_ptr<FunctionBase>& fn) const;

  const Tensor& grad() const;
  void setGrad(const Tensor& grad) const;
  void setGrad(Tensor&& grad) const;
  void addGrad(const Tensor& grad) const;
  void addGrad(Tensor&& grad) const;
  void zeroGrad() const;

  void backward(const Tensor& grad) const;
  void backward() const;
  bool isLeaf() const;

  template <typename T>
  std::vector<T> toList() const;

  template <typename T>
  T item() const;

  // convert
  Tensor to(DType type) const;
  Tensor to(Device device) const;

  // fill
  void fill_(const Scalar& val);
  void fillMasked_(const Tensor& mask, const Scalar& val);
  void fillZero_();
  void fillOne_();
  void fillLinSpace_(const Scalar& start, const Scalar& step, int64_t steps);
  void fillUniform_(float min, float max);
  void fillNormal_(float mean = 0.f, float stddev = 1.f);
  void fillBernoulli_(float p);

  // math
  Tensor operator+(const Tensor& other) const;
  Tensor operator-(const Tensor& other) const;
  Tensor operator*(const Tensor& other) const;
  Tensor operator/(const Tensor& other) const;

  Tensor operator+(const Scalar& other) const;
  Tensor operator-(const Scalar& other) const;
  Tensor operator*(const Scalar& other) const;
  Tensor operator/(const Scalar& other) const;

  friend Tensor operator+(const Scalar& self, const Tensor& other);
  friend Tensor operator-(const Scalar& self, const Tensor& other);
  friend Tensor operator*(const Scalar& self, const Tensor& other);
  friend Tensor operator/(const Scalar& self, const Tensor& other);

  void operator+=(const Tensor& other);
  void operator-=(const Tensor& other);
  void operator*=(const Tensor& other);
  void operator/=(const Tensor& other);

  void operator+=(const Scalar& other);
  void operator-=(const Scalar& other);
  void operator*=(const Scalar& other);
  void operator/=(const Scalar& other);

  Tensor operator<(const Tensor& other) const;
  Tensor operator<=(const Tensor& other) const;
  Tensor operator>(const Tensor& other) const;
  Tensor operator>=(const Tensor& other) const;
  Tensor operator==(const Tensor& other) const;
  Tensor operator!=(const Tensor& other) const;

  Tensor operator<(const Scalar& other) const;
  Tensor operator<=(const Scalar& other) const;
  Tensor operator>(const Scalar& other) const;
  Tensor operator>=(const Scalar& other) const;
  Tensor operator==(const Scalar& other) const;
  Tensor operator!=(const Scalar& other) const;

  Tensor sin() const;
  Tensor cos() const;
  Tensor sqrt() const;
  Tensor pow(const Scalar& exp) const;
  Tensor pow(const Tensor& exp) const;

  static Tensor maximum(const Tensor& a, const Tensor& b);
  static Tensor minimum(const Tensor& a, const Tensor& b);

  // reduce
  Tensor min() const;
  Tensor max() const;
  Tensor argmin() const;
  Tensor argmax() const;
  Tensor sum() const;
  Tensor mean() const;
  Tensor var(bool unbiased = true) const;
  TensorPair varMean(bool unbiased = true) const;

  TensorPair min(int64_t dim, bool keepDims = false) const;
  TensorPair max(int64_t dim, bool keepDims = false) const;
  Tensor argmin(int64_t dim, bool keepDims = false) const;
  Tensor argmax(int64_t dim, bool keepDims = false) const;
  Tensor sum(int64_t dim, bool keepDims = false) const;
  Tensor mean(int64_t dim, bool keepDims = false) const;
  Tensor var(int64_t dim, bool unbiased, bool keepDims = false) const;
  TensorPair varMean(int64_t dim, bool unbiased, bool keepDims = false) const;

  Tensor sum(IntArrayView dims, bool keepDims = false) const;
  Tensor mean(IntArrayView dims, bool keepDims = false) const;
  Tensor var(IntArrayView dims, bool unbiased = true, bool keepDims = false) const;
  TensorPair varMean(IntArrayView dims, bool unbiased = true, bool keepDims = false) const;

  // transform
  void reshape_(IntArrayView shape);
  void permute_(IntArrayView dims);
  void permute_();
  void flatten_(int64_t startDim = 0, int64_t endDim = -1);
  void unflatten_(int64_t dim, IntArrayView shape);
  void squeeze_(int64_t dim = -1);
  void squeeze_(IntArrayView dims);
  void unsqueeze_(int64_t dim);
  void transpose_(int64_t dim0, int64_t dim1);
  void t_();

  Tensor reshape(IntArrayView shape) const;
  Tensor view(IntArrayView shape) const;
  Tensor permute(IntArrayView dims) const;
  Tensor permute() const;
  Tensor flatten(int64_t startDim = 0, int64_t endDim = -1) const;
  Tensor unflatten(int64_t dim, IntArrayView shape) const;
  Tensor squeeze(int64_t dim = -1) const;
  Tensor squeeze(IntArrayView dims) const;
  Tensor unsqueeze(int64_t dim) const;
  Tensor transpose(int64_t dim0, int64_t dim1) const;
  Tensor t() const;

 private:
  void initAutogradMeta(const std::shared_ptr<FunctionBase>& fn = nullptr);
  void inplaceSet(Tensor&& val);
  void backwardImpl(const Tensor& grad) const;

  std::shared_ptr<TensorImpl> impl_ = nullptr;
  std::shared_ptr<AutogradMeta> gradMeta_ = nullptr;
};

template <typename T>
Tensor::Tensor(const std::vector<T>& values, const IntArrayView shape, Options options) {
  impl_ = std::make_shared<TensorImpl>(values, shape, options);
  initAutogradMeta();
}

template <typename T>
Tensor::Tensor(const std::vector<T>& values, const IntArrayView shape)
    : Tensor(values, shape, options::dtype(TypeToDType_v<T>)) {}

template <typename T>
Tensor::Tensor(const Array1d<T>& values, Options options) {
  SizeVector shape = {static_cast<int64_t>(values.size())};
  impl_ = std::make_shared<TensorImpl>(values, shape.view(), options);
  initAutogradMeta();
}

template <typename T>
Tensor::Tensor(const Array1d<T>& values) : Tensor(values, options::dtype(TypeToDType_v<T>)) {}

template <typename T>
Tensor::Tensor(const Array2d<T>& values, Options options) {
  SizeVector shape = {static_cast<int64_t>(values.size()), static_cast<int64_t>(values[0].size())};
  impl_ = std::make_shared<TensorImpl>(flatten2d(values), shape.view(), options);
  initAutogradMeta();
}

template <typename T>
Tensor::Tensor(const Array2d<T>& values) : Tensor(values, options::dtype(TypeToDType_v<T>)) {}

template <typename T>
Tensor::Tensor(const Array3d<T>& values, Options options) {
  SizeVector shape = {static_cast<int64_t>(values.size()), static_cast<int64_t>(values[0].size()),
                      static_cast<int64_t>(values[0][0].size())};
  impl_ = std::make_shared<TensorImpl>(flatten3d(values), shape.view(), options);
  initAutogradMeta();
}

template <typename T>
Tensor::Tensor(const Array3d<T>& values) : Tensor(values, options::dtype(TypeToDType_v<T>)) {}

template <typename T>
Tensor Tensor::arange(T start, T stop, T step, Options options) {
  options.dtype(TypeToDType_v<T>);
  auto steps = static_cast<int64_t>(std::ceil((stop - start) / step));
  Tensor ret({steps}, options);
  ret.fillLinSpace_(start, step, steps);
  return ret;
}

template <typename T>
Tensor Tensor::linspace(T start, T end, int64_t steps, Options options) {
  ASSERT(steps > 0);
  options.dtype(TypeToDType_v<T>);
  T step = 0;
  if (steps > 1) {
    step = (end - start) / static_cast<T>(steps - 1);
  }
  Tensor ret({steps}, options);
  ret.fillLinSpace_(start, step, steps);
  return ret;
}

template <typename T>
std::vector<T> Tensor::toList() const {
  ASSERT(defined());
  return impl_->toList<T>();
}

template <typename T>
T Tensor::item() const {
  ASSERT(defined());
  return impl_->item<T>();
}

}  // namespace tinytorch
