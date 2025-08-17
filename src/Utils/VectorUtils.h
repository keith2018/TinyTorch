/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <initializer_list>
#include <memory>
#include <vector>

namespace tinytorch {

template <typename T>
class ArrayView;

template <typename T, size_t N = 5>
class SmallVector {
 public:
  using size_type = size_t;

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  SmallVector() : size_(0), capacity_(N), data_(inlineData()) {}

  explicit SmallVector(size_type size) : SmallVector() {
    reserve(size);
    std::uninitialized_value_construct_n(data_, size);
    size_ = size;
  }

  SmallVector(size_type size, const T& value) : SmallVector() {
    reserve(size);
    std::uninitialized_fill_n(data_, size, value);
    size_ = size;
  }

  SmallVector(std::initializer_list<T> init) : SmallVector() {
    reserve(init.size());
    std::uninitialized_copy(init.begin(), init.end(), data_);
    size_ = init.size();
  }

  // NOLINTNEXTLINE(google-explicit-constructor)
  SmallVector(const std::vector<T>& init) : SmallVector() {
    static_assert(!std::is_same_v<T, bool>,
                  "SmallVector<bool> cannot be constructed from a std::vector<bool> bitfield.");
    reserve(init.size());
    std::uninitialized_copy(init.begin(), init.end(), data_);
    size_ = init.size();
  }

  // NOLINTNEXTLINE(google-explicit-constructor)
  SmallVector(const ArrayView<T>& view) : SmallVector() {
    reserve(view.size());
    std::uninitialized_copy(view.begin(), view.end(), data_);
    size_ = view.size();
  }

  SmallVector(const SmallVector& other) : SmallVector() {
    reserve(other.size_);
    std::uninitialized_copy(other.begin(), other.end(), data_);
    size_ = other.size_;
  }

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  SmallVector(SmallVector&& other) noexcept : size_(other.size_), capacity_(other.capacity_) {
    if (other.data_ == other.inlineData()) {
      data_ = inlineData();
      std::uninitialized_move(other.data_, other.data_ + size_, data_);
      other.clear();
    } else {
      data_ = other.data_;
      other.data_ = other.inlineData();
      other.size_ = 0;
      other.capacity_ = N;
    }
  }

  SmallVector& operator=(const SmallVector& other) {
    if (this != &other) {
      clear();
      reserve(other.size_);
      std::uninitialized_copy(other.begin(), other.end(), data_);
      size_ = other.size_;
    }
    return *this;
  }

  SmallVector& operator=(SmallVector&& other) noexcept {
    if (this != &other) {
      clear();
      if (data_ != inlineData()) {
        operator delete[](data_, static_cast<std::align_val_t>(alignof(T)));
      }
      size_ = other.size_;
      capacity_ = other.capacity_;
      if (other.data_ == other.inlineData()) {
        data_ = inlineData();
        std::uninitialized_move(other.data_, other.data_ + size_, data_);
        other.clear();
      } else {
        data_ = other.data_;
        other.data_ = other.inlineData();
        other.size_ = 0;
        other.capacity_ = N;
      }
    }
    return *this;
  }

  bool operator==(const ArrayView<T>& other) const { return other == *this; }
  bool operator!=(const ArrayView<T>& other) const { return other != *this; }

  ~SmallVector() {
    clear();
    if (data_ != inlineData()) {
      operator delete[](data_, static_cast<std::align_val_t>(alignof(T)));
    }
  }

  template <typename... Args>
  void pushBack(Args&&... args) {
    if (size_ == capacity_) {
      grow();
    }
    new (data_ + size_) T(std::forward<Args>(args)...);
    size_++;
  }

  void clear() noexcept {
    std::destroy(data_, data_ + size_);
    size_ = 0;
  }

  void reserve(size_type capacity) {
    if (capacity > capacity_) {
      growTo(capacity);
    }
  }

  void resize(size_type size) {
    if (size < size_) {
      std::destroy(data_ + size, data_ + size_);
    } else if (size > size_) {
      reserve(size);
      std::uninitialized_value_construct_n(data_ + size_, size - size_);
    }
    size_ = size;
  }

  void resize(size_type size, const T& value) {
    if (size < size_) {
      std::destroy(data_ + size, data_ + size_);
    } else if (size > size_) {
      reserve(size);
      std::uninitialized_fill_n(data_ + size_, size - size_, value);
    }
    size_ = size;
  }

  T* insert(T* pos, const T& value) {
    size_type idx = pos - data_;
    if (size_ == capacity_) {
      grow();
    }
    if (idx < size_) {
      new (data_ + size_) T(std::move(data_[size_ - 1]));
      for (size_type i = size_ - 1; i > idx; i--) {
        data_[i] = std::move(data_[i - 1]);
      }
      data_[idx] = value;
    } else {
      new (data_ + size_) T(value);
    }
    size_++;
    return data_ + idx;
  }

  bool empty() const { return size_ == 0; }
  size_type size() const { return size_; }
  size_type capacity() const { return capacity_; }
  T* data() { return data_; }
  const T* data() const { return data_; }
  T& operator[](size_type idx) { return data_[idx]; }
  const T& operator[](size_type idx) const { return data_[idx]; }
  T* begin() { return data_; }
  T* end() { return data_ + size_; }
  const T* begin() const { return data_; }
  const T* end() const { return data_ + size_; }
  T& front() { return data_[0]; }
  const T& front() const { return data_[0]; }
  T& back() { return data_[size_ - 1]; }
  const T& back() const { return data_[size_ - 1]; }
  ArrayView<T> view() const { return ArrayView<T>(data_, size_); }

 private:
  T* inlineData() noexcept { return reinterpret_cast<T*>(inlineStorage_); }
  const T* inlineData() const noexcept { return reinterpret_cast<const T*>(inlineStorage_); }

  void grow() { growTo(capacity_ * 2); }
  void growTo(size_type newCap) {
    T* newData = static_cast<T*>(operator new[](sizeof(T) * newCap, static_cast<std::align_val_t>(alignof(T))));
    std::uninitialized_move(data_, data_ + size_, newData);
    std::destroy(data_, data_ + size_);
    if (data_ != inlineData()) {
      operator delete[](data_, static_cast<std::align_val_t>(alignof(T)));
    }
    data_ = newData;
    capacity_ = newCap;
  }

  size_type size_;
  size_type capacity_;
  T* data_;
  alignas(T) char inlineStorage_[sizeof(T) * N];
};

template <typename T>
class ArrayView {
 public:
  using value_type = T;
  using const_iterator = const T*;
  using size_type = size_t;

  ArrayView() : data_(nullptr), size_(0) {}
  ArrayView(const T* data, size_type size) : data_(data), size_(size) {}
  // NOLINTNEXTLINE(google-explicit-constructor)
  ArrayView(const std::vector<T>& vec) : data_(vec.data()), size_(vec.size()) {
    static_assert(!std::is_same_v<T, bool>, "ArrayView<bool> cannot be constructed from a std::vector<bool> bitfield.");
  }
  // NOLINTNEXTLINE(google-explicit-constructor)
  ArrayView(const SmallVector<T>& vec) : data_(vec.data()), size_(vec.size()) {}
  ArrayView(std::initializer_list<T> vec)
      : data_(std::begin(vec) == std::end(vec) ? static_cast<T*>(nullptr) : std::begin(vec)), size_(vec.size()) {}

  ArrayView(const ArrayView& other) = default;
  ArrayView& operator=(const ArrayView& other) = default;

  const T& operator[](size_type idx) const {
    ASSERT(idx < size_);
    return data_[idx];
  }

  const T& front() const {
    ASSERT(!empty());
    return data_[0];
  }

  const T& back() const {
    ASSERT(!empty());
    return data_[size_ - 1];
  }

  bool operator==(const ArrayView& other) const {
    if (size_ != other.size_) {
      return false;
    }
    for (size_type i = 0; i < size_; i++) {
      if (!(data_[i] == other.data_[i])) {
        return false;
      }
    }
    return true;
  }

  bool operator!=(const ArrayView& other) const { return !(*this == other); }

  size_type size() const { return size_; }
  bool empty() const { return size_ == 0; }
  const T* data() const { return data_; }

  const_iterator begin() const { return data_; }
  const_iterator end() const { return data_ + size_; }

 private:
  const T* data_;
  size_type size_;
};

using IntArrayView = ArrayView<int64_t>;

}  // namespace tinytorch
