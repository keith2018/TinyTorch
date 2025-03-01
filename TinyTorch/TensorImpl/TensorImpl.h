/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "TensorImpl_base.h"

namespace TinyTorch {

#ifdef DEFAULT_DEVICE_CUDA
static Device defaultDevice = Device::CUDA;
#else
static Device defaultDevice = Device::CPU;
#endif

class TensorOperations;
class TensorOpsCPU;

#ifdef USE_CUDA
class TensorOpsCUDA;
#endif

// float type elements only
class TensorImpl {
  friend class TensorOperations;
  friend class TensorOpsCPU;

#ifdef USE_CUDA
  friend class TensorOpsCUDA;
#endif

 public:
  TensorImpl() = default;
  ~TensorImpl();

  TensorImpl(const TensorImpl &other);
  TensorImpl(TensorImpl &&other) noexcept;

  TensorImpl &operator=(const TensorImpl &other);
  TensorImpl &operator=(TensorImpl &&other) noexcept;

  // create
  explicit TensorImpl(const Array1d &values1d, Device device = defaultDevice);
  explicit TensorImpl(const Array2d &values2d, Device device = defaultDevice);
  explicit TensorImpl(const Array3d &values3d, Device device = defaultDevice);

  static TensorImpl shape(const Shape &shape, Device device = defaultDevice);
  static TensorImpl scalar(const float &value, Device device = defaultDevice);

  static TensorImpl ones(const Shape &shape, Device device = defaultDevice);

  static TensorImpl onesLike(const TensorImpl &t,
                             Device device = defaultDevice);

  static TensorImpl zeros(const Shape &shape, Device device = defaultDevice);

  static TensorImpl zerosLike(const TensorImpl &t,
                              Device device = defaultDevice);

  static TensorImpl rand(const Shape &shape, Device device = defaultDevice);

  static TensorImpl uniform(const Shape &shape, float min, float max,
                            Device device = defaultDevice);

  static TensorImpl randn(const Shape &shape, Device device = defaultDevice);

  static TensorImpl bernoulli(const Shape &shape, float p,
                              Device device = defaultDevice);

  static TensorImpl arange(float start, float stop, float step = 1.f,
                           Device device = defaultDevice);

  static TensorImpl linspace(float start, float end, int32_t steps,
                             Device device = defaultDevice);

  // conversion
  void to_(Device device);
  TensorImpl to(Device device);

  std::vector<float> toList() const;
  float item() const;

  // shape
  TensorImpl reshape(const Shape &shape);
  static TensorImpl reshape(const TensorImpl &t, const Shape &shape);
  TensorImpl reshape(const Shape &shape) const;
  TensorImpl view(const Shape &shape) const { return reshape(shape); }

  void flatten_(int32_t startDim = 0, int32_t endDim = -1);
  static TensorImpl flatten(const TensorImpl &t, int32_t startDim = 0,
                            int32_t endDim = -1);
  void unflatten_(int32_t dim, const std::vector<int32_t> &sizes);
  static TensorImpl unflatten(const TensorImpl &t, int32_t dim,
                              const std::vector<int32_t> &sizes);

  void squeeze_(int32_t dim = -1);
  void squeeze_(const std::vector<int32_t> &dims);
  static TensorImpl squeeze(const TensorImpl &t, int32_t dim = -1);
  static TensorImpl squeeze(const TensorImpl &t,
                            const std::vector<int32_t> &dims);
  void unsqueeze_(int32_t dim);
  static TensorImpl unsqueeze(const TensorImpl &t, int32_t dim);

  // fill
  void fill_(float value);
  void fillUniform_(float min, float max);
  static void fill(TensorImpl &t, float value) { t.fill_(value); }
  static void fillUniform(TensorImpl &t, float min, float max) {
    t.fillUniform_(min, max);
  }

  // math
  TensorImpl operator+(const TensorImpl &other) const;
  TensorImpl operator-(const TensorImpl &other) const;
  TensorImpl operator*(const TensorImpl &other) const;
  TensorImpl operator/(const TensorImpl &other) const;

  TensorImpl operator+(const float &other) const;
  TensorImpl operator-(const float &other) const;
  TensorImpl operator*(const float &other) const;
  TensorImpl operator/(const float &other) const;

  friend TensorImpl operator+(const float &other, const TensorImpl &obj);
  friend TensorImpl operator-(const float &other, const TensorImpl &obj);
  friend TensorImpl operator*(const float &other, const TensorImpl &obj);
  friend TensorImpl operator/(const float &other, const TensorImpl &obj);

  void operator+=(const TensorImpl &other);
  void operator-=(const TensorImpl &other);
  void operator*=(const TensorImpl &other);
  void operator/=(const TensorImpl &other);

  void operator+=(const float &other);
  void operator-=(const float &other);
  void operator*=(const float &other);
  void operator/=(const float &other);

  TensorImpl pow(const TensorImpl &other) const;
  TensorImpl pow(const float &other) const;

  static TensorImpl pow(const TensorImpl &x1, const TensorImpl &x2) {
    return x1.pow(x2);
  }
  static TensorImpl pow(const TensorImpl &x1, const float &x2) {
    return x1.pow(x2);
  }

  void sin_();
  void cos_();
  void sqrt_();
  void tanh_();
  void exp_();
  void log_();

  static TensorImpl sin(const TensorImpl &t);
  static TensorImpl cos(const TensorImpl &t);
  static TensorImpl sqrt(const TensorImpl &t);
  static TensorImpl tanh(const TensorImpl &t);
  static TensorImpl exp(const TensorImpl &t);
  static TensorImpl log(const TensorImpl &t);

  TensorImpl sin() const { return sin(*this); }
  TensorImpl cos() const { return cos(*this); }
  TensorImpl sqrt() const { return sqrt(*this); }
  TensorImpl tanh() const { return tanh(*this); }
  TensorImpl exp() const { return exp(*this); }
  TensorImpl log() const { return log(*this); }

  // compare
  TensorImpl operator<(const TensorImpl &other) const;
  TensorImpl operator<=(const TensorImpl &other) const;
  TensorImpl operator>(const TensorImpl &other) const;
  TensorImpl operator>=(const TensorImpl &other) const;
  TensorImpl operator==(const TensorImpl &other) const;
  TensorImpl operator!=(const TensorImpl &other) const;

  TensorImpl operator<(const float &other) const;
  TensorImpl operator<=(const float &other) const;
  TensorImpl operator>(const float &other) const;
  TensorImpl operator>=(const float &other) const;
  TensorImpl operator==(const float &other) const;
  TensorImpl operator!=(const float &other) const;

  static TensorImpl maximum(const TensorImpl &a, const TensorImpl &b);
  static TensorImpl minimum(const TensorImpl &a, const TensorImpl &b);

  // clamp
  void clampMin_(float min);
  void clampMax_(float max);
  void clamp_(float min, float max);

  static TensorImpl clampMin(const TensorImpl &t, float min);
  static TensorImpl clampMax(const TensorImpl &t, float max);
  static TensorImpl clamp(const TensorImpl &t, float min, float max);

  TensorImpl clampMin(float min) const { return clampMin(*this, min); }
  TensorImpl clampMax(float max) const { return clampMax(*this, max); }
  TensorImpl clamp(float min, float max) const {
    return clamp(*this, min, max);
  }

  // aggregation
  static TensorImpl min(const TensorImpl &t);
  static TensorImpl max(const TensorImpl &t);
  static TensorImpl sum(const TensorImpl &t);
  static TensorImpl mean(const TensorImpl &t);
  static TensorImpl var(const TensorImpl &t, bool unbiased = true);
  static TensorImpl argmin(const TensorImpl &t);
  static TensorImpl argmax(const TensorImpl &t);

  static TensorImpl min(const TensorImpl &t, int32_t dim,
                        bool keepDims = false);
  static TensorImpl max(const TensorImpl &t, int32_t dim,
                        bool keepDims = false);
  static TensorImpl sum(const TensorImpl &t, int32_t dim,
                        bool keepDims = false);
  static TensorImpl mean(const TensorImpl &t, int32_t dim,
                         bool keepDims = false);
  static TensorImpl var(const TensorImpl &t, int32_t dim, bool unbiased = true,
                        bool keepDims = false);
  static TensorImpl argmin(const TensorImpl &t, int32_t dim,
                           bool keepDims = false);
  static TensorImpl argmax(const TensorImpl &t, int32_t dim,
                           bool keepDims = false);

  static TensorImpl sum(const TensorImpl &t, const std::vector<int32_t> &dims,
                        bool keepDims = false);
  static TensorImpl mean(const TensorImpl &t, const std::vector<int32_t> &dims,
                         bool keepDims = false);
  static TensorImpl var(const TensorImpl &t, const std::vector<int32_t> &dims,
                        bool unbiased = true, bool keepDims = false);

  TensorImpl min() const { return min(*this); }

  TensorImpl max() const { return max(*this); }

  TensorImpl sum() const { return sum(*this); }

  TensorImpl mean() const { return mean(*this); }

  TensorImpl var(bool unbiased = true) const { return var(*this, unbiased); }

  TensorImpl argmin() const { return argmin(*this); }

  TensorImpl argmax() const { return argmax(*this); }

  TensorImpl min(int32_t dim, bool keepDims = false) const {
    return min(*this, dim, keepDims);
  }

  TensorImpl max(int32_t dim, bool keepDims = false) const {
    return max(*this, dim, keepDims);
  }

  TensorImpl sum(int32_t dim, bool keepDims = false) const {
    return sum(*this, dim, keepDims);
  }

  TensorImpl mean(int32_t dim, bool keepDims = false) const {
    return mean(*this, dim, keepDims);
  }

  TensorImpl var(int32_t dim, bool unbiased, bool keepDims = false) const {
    return var(*this, dim, unbiased, keepDims);
  }

  TensorImpl argmin(int32_t dim, bool keepDims = false) const {
    return argmin(*this, dim, keepDims);
  }

  TensorImpl argmax(int32_t dim, bool keepDims = false) const {
    return argmax(*this, dim, keepDims);
  }

  TensorImpl sum(const std::vector<int32_t> &dims,
                 bool keepDims = false) const {
    return sum(*this, dims, keepDims);
  }

  TensorImpl mean(const std::vector<int32_t> &dims,
                  bool keepDims = false) const {
    return mean(*this, dims, keepDims);
  }

  TensorImpl var(const std::vector<int32_t> &dims, bool unbiased = true,
                 bool keepDims = false) const {
    return var(*this, dims, unbiased, keepDims);
  }

  // transpose
  void transpose_(int32_t dim0, int32_t dim1);
  static TensorImpl transpose(const TensorImpl &t, int32_t dim0, int32_t dim1);
  TensorImpl transpose(int32_t dim0, int32_t dim1) const {
    return transpose(*this, dim0, dim1);
  }

  void t_();
  static TensorImpl t(const TensorImpl &t);
  TensorImpl t() const { return t(*this); }

  // permute
  TensorImpl permute(const std::vector<int32_t> &dims = {}) const;

  static TensorImpl permute(const TensorImpl &t,
                            const std::vector<int32_t> &dims = {}) {
    return t.permute(dims);
  }

  // index
  template <typename... Args>
  TensorImpl index(Args... args) const {
    static_assert((std::is_same_v<Args, int32_t> && ...));
    std::vector<int32_t> indices;
    indices.reserve(sizeof...(args));
    (indices.push_back(args), ...);
    return index(indices);
  }
  TensorImpl index(const std::vector<int32_t> &indices) const;
  TensorImpl index(const std::vector<TensorImpl> &indices) const;

  void indexPut_(const std::vector<int32_t> &indices, float val);
  void indexPut_(const std::vector<int32_t> &indices, const TensorImpl &val);
  void indexPut_(const std::vector<TensorImpl> &indices, float val);
  void indexPut_(const std::vector<TensorImpl> &indices, const TensorImpl &val);

  // stack
  static TensorImpl stack(
      const std::vector<std::reference_wrapper<TensorImpl>> &tensors,
      int32_t dim = 0);

  // dot
  TensorImpl dot(const TensorImpl &t) const { return dot(*this, t); }
  static TensorImpl dot(const TensorImpl &a, const TensorImpl &b);

  // matmul
  TensorImpl matmul(const TensorImpl &t) const { return matmul(*this, t); }
  static TensorImpl matmul(const TensorImpl &a, const TensorImpl &b);

  TensorImpl matmulTrans(const TensorImpl &t, bool transA, bool transB) const {
    return matmulTrans(*this, t, transA, transB);
  }
  static TensorImpl matmulTrans(const TensorImpl &a, const TensorImpl &b,
                                bool transA, bool transB);

  // im2col
  TensorImpl im2col(Size2D kernel, Size2D stride, Size2D padding = 0) const;

  // col2im
  TensorImpl col2im(const Shape &shape, Size2D kernelSize, Size2D stride,
                    Size2D padding = 0) const;

  // properties
  Device device() const { return device_; }

  bool empty() const { return elemCount_ == 0; }

  int32_t dim() const { return dimCount_; }

  int32_t numel() const { return elemCount_; }

  const Shape &shape() const { return shape_; }

  const Shape &strides() const { return strides_; }

  float *data() { return data_; }

  const float *data() const { return data_; }

  TensorOperations *ops() const { return ops_; }

  static bool deviceAvailable(Device device) {
    return getOps(device) != nullptr;
  }

 protected:
  void initMeta();
  void initMeta(const TensorImpl &other);
  void initData(const float *ptr = nullptr, Device device = Device::CPU);
  void dispose();

  void moveFrom(TensorImpl &&other);
  void copyToDevice(void *dst, const void *src, size_t count, Device srcDevice);
  static TensorOperations *getOps(Device device);

  int32_t dimCount_ = 0;
  int32_t elemCount_ = 0;
  Shape shape_;
  Shape strides_;
  float *data_ = nullptr;

  Device device_ = Device::CPU;
  TensorOperations *ops_ = nullptr;
};

}  // namespace TinyTorch
