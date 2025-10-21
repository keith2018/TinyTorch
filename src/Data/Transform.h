/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Tensor.h"

namespace tinytorch::data::transforms {

class Transform {
 public:
  virtual ~Transform() = default;
  virtual Tensor process(Tensor& input) const { return input; }
};

class Compose : public Transform {
 public:
  template <typename... Transforms>
  explicit Compose(Transforms&&... transforms) {
    transforms_.reserve(sizeof...(Transforms));
    pushBack(std::forward<Transforms>(transforms)...);
  }

  Compose(std::initializer_list<std::shared_ptr<Transform>> transforms) {
    transforms_.reserve(transforms.size());
    for (const auto& transform : transforms) {
      transforms_.emplace_back(transform);
    }
  }

  template <typename TransformType>
  void pushBack(TransformType&& transform) {
    transforms_.push_back(std::make_shared<TransformType>(std::forward<TransformType>(transform)));
  }

  void pushBack(const std::shared_ptr<Transform>& transform) { transforms_.emplace_back(transform); }

  Tensor process(Tensor& input) const override {
    Tensor ret = input;
    for (auto& trans : transforms_) {
      ret = trans->process(ret);
    }
    return ret;
  }

 private:
  template <typename First, typename Second, typename... Rest>
  void pushBack(First&& first, Second&& second, Rest&&... rest) {
    pushBack(std::forward<First>(first));
    pushBack(std::forward<Second>(second), std::forward<Rest>(rest)...);
  }

  void pushBack() {}

  std::vector<std::shared_ptr<Transform>> transforms_;
};

class Normalize : public Transform {
 public:
  Normalize(float mean, float std) : mean_(mean), std_(std) {}
  Tensor process(Tensor& input) const override { return (input - mean_) / std_; }

 private:
  float mean_;
  float std_;
};

}  // namespace tinytorch::data::transforms
