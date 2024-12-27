/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace TinyTorch {

#define TENSOR_MAX_DIMS 8

typedef enum TensorError_ {
  TensorError_None = 0,
  TensorError_EmptyTensor,
  TensorError_InvalidShape,
  TensorError_InvalidAxis,
  TensorError_InvalidSections,
  TensorError_ShapeNotAligned,
  TensorError_NotSupport,
} TensorError;

typedef enum ShapeCompatible_ {
  ShapeCompatible_Error = 0,
  ShapeCompatible_SameShape,
  ShapeCompatible_Broadcast,
} ShapeCompatible;

typedef std::vector<int32_t> Shape;
typedef std::vector<float> Array1d;
typedef std::vector<std::vector<float>> Array2d;
typedef std::vector<std::vector<std::vector<float>>> Array3d;

class RandomGenerator {
 public:
  static void setSeed(const unsigned int seed) {
    seed_ = seed;
    randomEngine_ = std::default_random_engine(seed_.value());
  }
  static std::default_random_engine getGenerator() {
    if (seed_.has_value()) {
      return randomEngine_;
    }
    std::random_device r;
    return std::default_random_engine(r());
  }

 private:
  static std::optional<unsigned int> seed_;
  static std::default_random_engine randomEngine_;
};

struct Size2D {
  Size2D(int32_t n) : h(n), w(n) {}
  Size2D(int32_t h, int32_t w) : h(h), w(w) {}

  int32_t h = 0;
  int32_t w = 0;
};

// one axis only
class Axis {
 public:
  Axis() = delete;

  Axis(int32_t axis) : axis_(axis) {}

  int32_t get(int32_t axisCnt) const {
    return axis_ >= 0 ? axis_ : axis_ + axisCnt;
  }

 private:
  int32_t axis_ = 0;
};

class UFunc {
 public:
  virtual ~UFunc() = default;
  virtual void op(const float &val) { idx_++; };

  virtual float result() { return tmp; };

  virtual void reset() {
    idx_ = 0;
    tmp = 0.f;
  }

 protected:
  int32_t idx_ = 0;
  float tmp = 0.f;
};

class UFuncSum : public UFunc {
 public:
  void op(const float &val) override { tmp += val; }
};

class UFuncMean : public UFunc {
 public:
  void op(const float &val) override {
    idx_++;
    tmp += val;
  }

  float result() override { return tmp / (float)idx_; }
};

class UFuncVar : public UFunc {
 public:
  void op(const float &val) override {
    idx_++;
    tmp += val;
    squareSum_ += val * val;
  }

  float result() override {
    float mean = tmp / (float)idx_;
    return squareSum_ / (float)idx_ - mean * mean;
  }

  void reset() override {
    idx_ = 0;
    tmp = 0;
    squareSum_ = 0;
  }

 private:
  float squareSum_ = 0;
};

class UFuncMin : public UFunc {
 public:
  void op(const float &val) override {
    if (val < tmp) {
      tmp = val;
    }
  }

  void reset() override { tmp = std::numeric_limits<float>::max(); }
};

class UFuncMax : public UFunc {
 public:
  void op(const float &val) override {
    if (val > tmp) {
      tmp = val;
    }
  }

  void reset() override { tmp = -std::numeric_limits<float>::max(); }
};

class UFuncArgMin : public UFunc {
 public:
  void op(const float &val) override {
    if (val < tmp) {
      tmp = val;
      minIdx_ = idx_;
    }
    idx_++;
  }

  float result() override { return (float)minIdx_; }

  void reset() override {
    tmp = std::numeric_limits<float>::max();
    idx_ = 0;
    minIdx_ = 0;
  }

 private:
  int32_t minIdx_ = 0;
};

class UFuncArgMax : public UFunc {
 public:
  void op(const float &val) override {
    if (val > tmp) {
      tmp = val;
      maxIdx_ = idx_;
    }
    idx_++;
  }

  float result() override { return (float)maxIdx_; }

  void reset() override {
    tmp = -std::numeric_limits<float>::max();
    idx_ = 0;
    maxIdx_ = 0;
  }

 private:
  int32_t maxIdx_ = 0;
};

// float type elements only
class TensorImpl {
 public:
  TensorImpl() = default;

  TensorImpl(const TensorImpl &other) {
    dispose();
    copyFrom(other);
    initData(other.data_);
  }

  TensorImpl(TensorImpl &&other) noexcept {
    copyFrom(other);
    other.data_ = nullptr;
  }

  TensorImpl &operator=(const TensorImpl &other) {
    if (this == &other) {
      return *this;
    }
    dispose();
    copyFrom(other);
    initData(other.data_);
    return *this;
  }

  TensorImpl &operator=(TensorImpl &&other) noexcept {
    if (this == &other) {
      return *this;
    }
    dispose();
    copyFrom(other);
    other.data_ = nullptr;
    return *this;
  }

  void copyFrom(const TensorImpl &other) {
    dimCount_ = other.dimCount_;
    elemCount_ = other.elemCount_;
    shape_ = other.shape_;
    strides_ = other.strides_;
    data_ = other.data_;
  }

  void dispose() {
    dimCount_ = 0;
    elemCount_ = 0;
    shape_.clear();
    strides_.clear();
    delete[] data_;
    data_ = nullptr;
  }

  ~TensorImpl() { dispose(); }

  static TensorImpl shape(const Shape &shape);

  static TensorImpl scalar(const float &value);

  static TensorImpl ones(const Shape &shape);

  static TensorImpl onesLike(const TensorImpl &t);

  static TensorImpl zeros(const Shape &shape);

  static TensorImpl rand(const Shape &shape);

  static TensorImpl randn(const Shape &shape);

  static TensorImpl bernoulli(const Shape &shape, float p);

  static TensorImpl tri(int32_t n, int32_t m = 0, int32_t k = 0);

  // 1d array
  explicit TensorImpl(const Array1d &values1d);
  // 2d array
  explicit TensorImpl(const Array2d &values2d);
  // 3d array
  explicit TensorImpl(const Array3d &values3d);

  TensorImpl reshape(const Shape &shape);
  static TensorImpl reshape(const TensorImpl &t, const Shape &shape);
  TensorImpl reshape(const Shape &shape) const;
  TensorImpl view(const Shape &shape) const { return reshape(shape); }

  void flatten(int32_t startDim = 0, int32_t endDim = -1);
  static TensorImpl flatten(const TensorImpl &t, int32_t startDim = 0,
                            int32_t endDim = -1);
  void unflatten(int32_t dim, const std::vector<int32_t> &sizes);
  static TensorImpl unflatten(const TensorImpl &t, int32_t dim,
                              const std::vector<int32_t> &sizes);

  void squeeze(int32_t dim = -1);
  void squeeze(const std::vector<int32_t> &dims);
  static TensorImpl squeeze(const TensorImpl &t, int32_t dim = -1);
  static TensorImpl squeeze(const TensorImpl &t,
                            const std::vector<int32_t> &dims);
  void unsqueeze(int32_t dim);
  static TensorImpl unsqueeze(const TensorImpl &t, int32_t dim);

  bool empty() const { return elemCount_ == 0; }

  bool isScalar() const { return dimCount_ == 0 && elemCount_ == 1; }

  int32_t dim() const { return dimCount_; }

  int32_t size() const { return elemCount_; }

  const Shape &shape() const { return shape_; }

  const Shape &strides() const { return strides_; }

  float item() const { return data_[0]; }

  float &operator[](int32_t idx) { return data_[idx]; }

  const float &operator[](int32_t idx) const { return data_[idx]; }

  template <typename T = float>
  std::vector<T> toArray() const;

  // fill
  void fill(float value);
  static TensorImpl fill(const TensorImpl &t, float value);

  // clamp
  void clampMin(float min);
  void clampMax(float max);
  void clamp(float min, float max);

  static TensorImpl clampMin(const TensorImpl &t, float min);
  static TensorImpl clampMax(const TensorImpl &t, float max);
  static TensorImpl clamp(const TensorImpl &t, float min, float max);

  // range
  static std::vector<int32_t> range(int32_t start, int32_t stop,
                                    int32_t step = 1);
  static TensorImpl arange(float start, float stop, float step = 1.f);
  static TensorImpl linspace(float start, float end, int steps);

  // indexing
  template <typename... Args>
  TensorImpl index(Args... args) const {
    std::vector<int32_t> vec;
    vec.reserve(sizeof...(args));
    (vec.push_back(args), ...);
    return indexInteger(vec);
  }
  TensorImpl indexInteger(const std::vector<int32_t> &idx,
                          float *dataPtr = nullptr) const;
  TensorImpl index(const std::vector<int32_t> &idx) const;
  TensorImpl indexAdvance(
      const std::vector<std::vector<int32_t>> &indexes) const;

  void indexIntegerSet(const std::vector<int32_t> &idx, float val);
  void indexIntegerSet(const std::vector<int32_t> &idx, const TensorImpl &val);
  void indexAdvanceSet(const std::vector<std::vector<int32_t>> &indexes,
                       float val);
  void indexAdvanceSet(const std::vector<std::vector<int32_t>> &indexes,
                       const TensorImpl &val);

  // im2col
  TensorImpl im2col(Size2D kernelSize, Size2D stride, Size2D padding = 0) const;
  // col2im
  TensorImpl col2im(const Shape &inputShape, Size2D kernelSize, Size2D stride,
                    Size2D padding = 0) const;

  // transpose
  TensorImpl transpose(const std::vector<int32_t> &axis = {}) const;

  static TensorImpl transpose(const TensorImpl &t,
                              const std::vector<int32_t> &axis = {}) {
    return t.transpose(axis);
  }

  // split
  std::vector<TensorImpl> split(int32_t sections, const Axis &axis = 0) const;

  std::vector<TensorImpl> vsplit(int32_t sections) const {
    return split(sections, 0);
  }

  std::vector<TensorImpl> hsplit(int32_t sections) const {
    return split(sections, 1);
  }

  std::vector<TensorImpl> dsplit(int32_t sections) const {
    return split(sections, 2);
  }

  std::vector<TensorImpl> split(const std::vector<int32_t> &indices,
                                const Axis &axis = 0) const;

  std::vector<TensorImpl> vsplit(const std::vector<int32_t> &indices) const {
    return split(indices, 0);
  }

  std::vector<TensorImpl> hsplit(const std::vector<int32_t> &indices) const {
    return split(indices, 1);
  }

  std::vector<TensorImpl> dsplit(const std::vector<int32_t> &indices) const {
    return split(indices, 2);
  }

  static std::vector<TensorImpl> split(const TensorImpl &t, int32_t sections,
                                       const Axis &axis = 0) {
    return t.split(sections, axis);
  }

  static std::vector<TensorImpl> vsplit(const TensorImpl &t, int32_t sections) {
    return t.split(sections, 0);
  }

  static std::vector<TensorImpl> hsplit(const TensorImpl &t, int32_t sections) {
    return t.split(sections, 1);
  }

  static std::vector<TensorImpl> dsplit(const TensorImpl &t, int32_t sections) {
    return t.split(sections, 2);
  }

  static std::vector<TensorImpl> split(const TensorImpl &t,
                                       const std::vector<int32_t> &indices,
                                       const Axis &axis = 0) {
    return t.split(indices, axis);
  }

  static std::vector<TensorImpl> vsplit(const TensorImpl &t,
                                        const std::vector<int32_t> &indices) {
    return t.split(indices, 0);
  }

  static std::vector<TensorImpl> hsplit(const TensorImpl &t,
                                        const std::vector<int32_t> &indices) {
    return t.split(indices, 1);
  }

  static std::vector<TensorImpl> dsplit(const TensorImpl &t,
                                        const std::vector<int32_t> &indices) {
    return t.split(indices, 2);
  }

  // concatenate
  static TensorImpl concatenate(
      const std::vector<std::reference_wrapper<TensorImpl>> &arrays);
  static TensorImpl concatenate(
      const std::vector<std::reference_wrapper<TensorImpl>> &arrays,
      const Axis &axis);

  // stack
  static TensorImpl stack(
      const std::vector<std::reference_wrapper<TensorImpl>> &arrays,
      const Axis &axis = 0);
  static TensorImpl vstack(
      const std::vector<std::reference_wrapper<TensorImpl>> &arrays);
  static TensorImpl hstack(
      const std::vector<std::reference_wrapper<TensorImpl>> &arrays);
  static TensorImpl dstack(
      const std::vector<std::reference_wrapper<TensorImpl>> &arrays);

  // compare
  TensorImpl operator<(const TensorImpl &other) const;
  TensorImpl operator>(const TensorImpl &other) const;
  TensorImpl operator==(const TensorImpl &other) const;
  TensorImpl operator!=(const TensorImpl &other) const;
  TensorImpl operator<(const float &other) const;
  TensorImpl operator>(const float &other) const;
  TensorImpl operator==(const float &other) const;
  TensorImpl operator!=(const float &other) const;

  static TensorImpl maximum(const TensorImpl &a, const TensorImpl &b);
  static TensorImpl minimum(const TensorImpl &a, const TensorImpl &b);

  // math
  TensorImpl operator+(const TensorImpl &other) const;
  TensorImpl operator-(const TensorImpl &other) const;
  TensorImpl operator*(const TensorImpl &other) const;
  TensorImpl operator/(const TensorImpl &other) const;

  TensorImpl operator+(const float &other) const;
  TensorImpl operator-(const float &other) const;
  TensorImpl operator*(const float &other) const;
  TensorImpl operator/(const float &other) const;

  void operator+=(const TensorImpl &other);
  void operator-=(const TensorImpl &other);
  void operator*=(const TensorImpl &other);
  void operator/=(const TensorImpl &other);

  void operator+=(const float &other);
  void operator-=(const float &other);
  void operator*=(const float &other);
  void operator/=(const float &other);

  friend TensorImpl operator+(const float &other, const TensorImpl &obj);
  friend TensorImpl operator-(const float &other, const TensorImpl &obj);
  friend TensorImpl operator*(const float &other, const TensorImpl &obj);
  friend TensorImpl operator/(const float &other, const TensorImpl &obj);

  static TensorImpl sin(const TensorImpl &t);
  static TensorImpl cos(const TensorImpl &t);
  static TensorImpl sqrt(const TensorImpl &t);
  static TensorImpl tanh(const TensorImpl &t);
  static TensorImpl exp(const TensorImpl &t);
  static TensorImpl log(const TensorImpl &t);

  TensorImpl sin() const { return TensorImpl::sin(*this); }
  TensorImpl cos() const { return TensorImpl::cos(*this); }
  TensorImpl sqrt() const { return TensorImpl::sqrt(*this); }
  TensorImpl tanh() const { return TensorImpl::tanh(*this); }
  TensorImpl exp() const { return TensorImpl::exp(*this); }
  TensorImpl log() const { return TensorImpl::log(*this); }

  TensorImpl pow(const TensorImpl &other) const;
  TensorImpl pow(const float &other) const;

  static TensorImpl pow(const TensorImpl &x1, const TensorImpl &x2) {
    return x1.pow(x2);
  }
  static TensorImpl pow(const TensorImpl &x1, const float &x2) {
    return x1.pow(x2);
  }

  // linear algebra
  static float dot(const float &a, const float &b);
  static TensorImpl dot(const TensorImpl &a, const float &b);
  static TensorImpl dot(const float &a, const TensorImpl &b);
  static TensorImpl dot(const TensorImpl &a, const TensorImpl &b);
  static TensorImpl matmul(const TensorImpl &a, const TensorImpl &b);
  static TensorImpl matmulTrans(const TensorImpl &a, const TensorImpl &b);

  // aggregation

  static float min(const TensorImpl &t);
  static float max(const TensorImpl &t);
  static float mean(const TensorImpl &t);
  static float sum(const TensorImpl &t);
  static float var(const TensorImpl &t);
  static float argmin(const TensorImpl &t);
  static float argmax(const TensorImpl &t);

  float min() const { return TensorImpl::min(*this); };
  float max() const { return TensorImpl::max(*this); };
  float mean() const { return TensorImpl::mean(*this); };
  float sum() const { return TensorImpl::sum(*this); };
  float var() const { return TensorImpl::var(*this); };
  float argmin() const { return TensorImpl::argmin(*this); };
  float argmax() const { return TensorImpl::argmax(*this); };

  static TensorImpl min(const TensorImpl &t, const Axis &axis,
                        bool keepDims = false);
  static TensorImpl max(const TensorImpl &t, const Axis &axis,
                        bool keepDims = false);
  static TensorImpl mean(const TensorImpl &t, const Axis &axis,
                         bool keepDims = false);
  static TensorImpl sum(const TensorImpl &t, const Axis &axis,
                        bool keepDims = false);
  static TensorImpl var(const TensorImpl &t, const Axis &axis,
                        bool keepDims = false);
  static TensorImpl argmin(const TensorImpl &t, const Axis &axis,
                           bool keepDims = false);
  static TensorImpl argmax(const TensorImpl &t, const Axis &axis,
                           bool keepDims = false);

  TensorImpl min(const Axis &axis, bool keepDims = false) const {
    return TensorImpl::min(*this, axis, keepDims);
  }

  TensorImpl max(const Axis &axis, bool keepDims = false) const {
    return TensorImpl::max(*this, axis, keepDims);
  }

  TensorImpl mean(const Axis &axis, bool keepDims = false) const {
    return TensorImpl::mean(*this, axis, keepDims);
  }

  TensorImpl sum(const Axis &axis, bool keepDims = false) const {
    return TensorImpl::sum(*this, axis, keepDims);
  }

  TensorImpl var(const Axis &axis, bool keepDims = false) const {
    return TensorImpl::var(*this, axis, keepDims);
  }

  TensorImpl argmin(const Axis &axis, bool keepDims = false) const {
    return TensorImpl::argmin(*this, axis, keepDims);
  }

  TensorImpl argmax(const Axis &axis, bool keepDims = false) const {
    return TensorImpl::argmax(*this, axis, keepDims);
  }

 public:
  class Iterator {
   public:
    explicit Iterator(const float *ptr) : ptr(ptr) {}
    const float &operator*() const { return *ptr; }

    Iterator &operator++() {
      ++ptr;
      return *this;
    }

    bool operator==(const Iterator &other) const { return ptr == other.ptr; }
    bool operator!=(const Iterator &other) const { return ptr != other.ptr; }

   private:
    const float *ptr;
  };

  Iterator begin() const { return Iterator(data_); }
  Iterator end() const { return Iterator(data_ + elemCount_); }

 protected:
  void initMeta();
  void initData(const float *from = nullptr);

  void traverse(UFunc &func, int32_t start, int32_t stride, int32_t cnt) const;
  TensorImpl reduce(UFunc &func, int32_t axis, bool keepDims = false) const;
  void splitAxis(std::vector<TensorImpl> &retTensors,
                 std::vector<int32_t> &splitIndices, int32_t axis) const;

  static TensorImpl arraysConcat(
      const std::vector<std::reference_wrapper<TensorImpl>> &arrays,
      const Shape &retShape, const std::vector<int32_t> &concatIndices,
      int32_t axis);
  static ShapeCompatible checkCompatible(const Shape &t0, const Shape &t1,
                                         Shape &retShape, int32_t skipLast = 0);
  static bool checkShapeEqual(
      const std::vector<std::reference_wrapper<TensorImpl>> &arrays,
      int32_t exceptAxis);
  static void error(const char *where, TensorError error);

 private:
  static float fastTanh(float x);
  void indexIntegerSet(const std::vector<int32_t> &idx, const float *valPtr);

 protected:
  int32_t dimCount_ = 0;
  int32_t elemCount_ = 0;
  Shape shape_;
  Shape strides_;
  float *data_ = nullptr;
};

template <typename T>
std::vector<T> TensorImpl::toArray() const {
  std::vector<T> ret;
  ret.reserve(elemCount_);
  for (int32_t i = 0; i < elemCount_; i++) {
    ret.push_back((T)data_[i]);
  }
  return ret;
}

class TensorIter {
 public:
  explicit TensorIter(const Shape &shape);

  // get shape
  Shape shape();

  // reshape
  void reshape(const Shape &shape);

  // get size
  int32_t size() const { return size_; }

  // get current coordinates
  const int32_t *coordinates() const { return coordinates_; };

  // return -1 if not available
  int32_t next();

  // reset to init states
  void reset();

  // broadcast to shape (no broadcast rules check)
  void broadcast(const Shape &shape);

  // transpose
  void transpose(const std::vector<int32_t> &axis);

 protected:
  // reorder array
  static void reorder(int32_t *v, const std::vector<int32_t> &order) {
    auto n = order.size();
    std::vector<int32_t> temp(n);
    for (int i = 0; i < n; ++i) {
      temp[i] = v[order[i]];
    }
    memcpy(v, temp.data(), sizeof(int32_t) * n);
  }

 protected:
  int32_t ndM1_ = 0;
  int32_t size_ = 0;
  int32_t dimsM1_[TENSOR_MAX_DIMS]{};

  int32_t strides_[TENSOR_MAX_DIMS]{};
  int32_t backStrides_[TENSOR_MAX_DIMS]{};

  int32_t coordinates_[TENSOR_MAX_DIMS]{};
  int32_t index_ = 0;
  int32_t itCnt_ = 0;
};

}  // namespace TinyTorch