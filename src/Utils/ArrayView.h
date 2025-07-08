/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <initializer_list>
#include <vector>

#include "Macros.h"

namespace tinytorch {

template <typename T>
class ArrayView {
 public:
  using value_type = T;
  using const_iterator = const T*;
  using size_type = size_t;

  ArrayView() : data_(nullptr), size_(0) {}
  ArrayView(const T* data, size_t size) : data_(data), size_(size) {}
  // NOLINTNEXTLINE(google-explicit-constructor)
  ArrayView(const std::vector<T>& vec) : data_(vec.data()), size_(vec.size()) {
    static_assert(!std::is_same_v<T, bool>, "ArrayView<bool> cannot be constructed from a std::vector<bool> bitfield.");
  }
  ArrayView(std::initializer_list<T> vec)
      : data_(std::begin(vec) == std::end(vec) ? static_cast<T*>(nullptr) : std::begin(vec)), size_(vec.size()) {}

  ArrayView(const ArrayView& other) = default;
  ArrayView& operator=(const ArrayView& other) = default;

  const T& operator[](size_t idx) const {
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
    for (size_t i = 0; i < size_; i++) {
      if (!(data_[i] == other.data_[i])) {
        return false;
      }
    }
    return true;
  }

  bool operator!=(const ArrayView& other) const { return !(*this == other); }

  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }
  const T* data() const { return data_; }

  const_iterator begin() const { return data_; }
  const_iterator end() const { return data_ + size_; }

 private:
  const T* data_;
  size_t size_;
};

using IntArrayView = ArrayView<int64_t>;

}  // namespace tinytorch
