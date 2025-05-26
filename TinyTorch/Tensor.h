/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <memory>

#include "TensorImpl/TensorImpl.h"

namespace tinytorch {

struct AutogradMeta;
class Function;
class FuncLeaf;

class Tensor {
 public:
  Tensor() : data_(std::make_shared<TensorImpl>()) {}
  explicit Tensor(TensorImpl &&data, bool requiresGrad = false,
                  const std::shared_ptr<Function> &gradFunc = nullptr);

  explicit Tensor(const Array1d &values1d, bool requiresGrad = false);
  explicit Tensor(const Array2d &values2d, bool requiresGrad = false);
  explicit Tensor(const Array3d &values3d, bool requiresGrad = false);

  static Tensor shape(const Shape &shape, bool requiresGrad = false);
  static Tensor scalar(const float &value, bool requiresGrad = false);
  static Tensor ones(const Shape &shape, bool requiresGrad = false);
  static Tensor onesLike(const Tensor &t, bool requiresGrad = false);
  static Tensor zeros(const Shape &shape, bool requiresGrad = false);
  static Tensor randn(const Shape &shape, bool requiresGrad = false);
  static Tensor arange(float start, float stop, float steps,
                       bool requiresGrad = false);
  static Tensor linspace(float start, float end, int steps,
                         bool requiresGrad = false);

  bool empty() const { return data_->empty(); }

  int32_t dim() const { return data_->dim(); }

  int32_t numel() const { return data_->numel(); }

  const Shape &shape() const { return data_->shape(); }

  Device device() const { return data_->device(); }

  float item() const { return data_->item(); }

  Tensor reshape(const Shape &shape) const;
  static Tensor reshape(const Tensor &t, const Shape &shape);

  Tensor flatten(int32_t startDim = 0, int32_t endDim = -1) const;
  static Tensor flatten(const Tensor &t, int32_t startDim = 0,
                        int32_t endDim = -1);
  Tensor unflatten(int32_t dim, const std::vector<int32_t> &sizes) const;
  static Tensor unflatten(const Tensor &t, int32_t dim,
                          const std::vector<int32_t> &sizes);

  // functions
  Tensor operator+(const Tensor &other) const;
  Tensor operator-(const Tensor &other) const;
  Tensor operator*(const Tensor &other) const;
  Tensor operator/(const Tensor &other) const;

  Tensor operator+(const float &other) const;
  Tensor operator-(const float &other) const;
  Tensor operator*(const float &other) const;
  Tensor operator/(const float &other) const;

  friend Tensor operator+(const float &other, const Tensor &obj);
  friend Tensor operator-(const float &other, const Tensor &obj);
  friend Tensor operator*(const float &other, const Tensor &obj);
  friend Tensor operator/(const float &other, const Tensor &obj);

  void operator+=(const Tensor &other);
  void operator-=(const Tensor &other);
  void operator*=(const Tensor &other);
  void operator/=(const Tensor &other);

  void operator+=(const float &other);
  void operator-=(const float &other);
  void operator*=(const float &other);
  void operator/=(const float &other);

  Tensor sin() const;
  Tensor cos() const;
  Tensor pow(const float &exp) const;
  Tensor pow(const Tensor &exp) const;
  Tensor sum() const;

  Tensor squeeze(int32_t dim = -1) const;
  Tensor unsqueeze(int32_t dim) const;

 public:
  bool isRequiresGrad() const;

  void setRequiresGrad(bool requires);

  void backward(const Tensor &grad = {});

  const Tensor &getGrad() const;

  void setGrad(const Tensor &grad);

  void zeroGrad();

  std::shared_ptr<Function> getGradFunc() const;

  bool isLeaf() const;

  TensorImpl &data() const { return *data_; }

  Tensor &to(Device device) {
    data_->to_(device);
    return *this;
  }

  std::vector<float> toList() const { return data_->toList(); }

  static bool deviceAvailable(Device device) {
    return TensorImpl::deviceAvailable(device);
  }

 private:
  void initAutograd(bool requiresGrad,
                    const std::shared_ptr<Function> &gradFunc = nullptr);

 private:
  bool requiresGrad_ = false;
  std::shared_ptr<TensorImpl> data_;
  std::shared_ptr<AutogradMeta> gradMeta_ = nullptr;

  friend struct AutogradMeta;
  friend class Function;
  friend class FuncLeaf;
};

struct AutogradMeta : std::enable_shared_from_this<AutogradMeta> {
  void setGradFunc(const std::shared_ptr<Function> &gradFunc);
  void backward(const Tensor &grad);
  void buildBackwardGraph();

  Tensor grad_;
  std::shared_ptr<Function> gradFunc_;
  std::shared_ptr<FuncLeaf> gradLeaf_;
  std::vector<std::shared_ptr<Function>> backwardGraph_;
};

class NoGradScope {
 public:
  explicit NoGradScope(const char *tag) : tag_(tag) {
    prevGradEnabled = gradEnabled;
    gradEnabled = false;
  }

  ~NoGradScope() { gradEnabled = prevGradEnabled; }

  static bool isGradEnabled() { return gradEnabled; }

  explicit operator bool() const { return true; }

 private:
  thread_local static bool gradEnabled;
  bool prevGradEnabled;
  const char *tag_;
};

#define NoGradTag(X) if (auto _noGrad = tinytorch::NoGradScope(X))
#define withNoGrad NoGradTag("")

}  // namespace tinytorch
