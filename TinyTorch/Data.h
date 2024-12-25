/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <algorithm>
#include <numeric>
#include <vector>

#include "Tensor.h"

namespace TinyTorch::data {

namespace transforms {

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
    transforms_.push_back(std::make_shared<TransformType>(
        std::forward<TransformType>(transform)));
  }

  void pushBack(const std::shared_ptr<Transform>& transform) {
    transforms_.emplace_back(transform);
  }

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
  Tensor process(Tensor& input) const override;

 private:
  float mean_;
  float std_;
};

}  // namespace transforms

class Dataset {
 public:
  virtual ~Dataset() = default;

  virtual size_t size() const = 0;
  virtual std::vector<Tensor> getItem(size_t idx) = 0;
};

class DatasetMNIST : public Dataset {
 public:
  enum MnistDataType {
    TRAIN,
    TEST,
  };
  DatasetMNIST(const std::string& dir, MnistDataType type,
               const std::shared_ptr<transforms::Transform>& transform);

  size_t size() const override { return size_; }

  std::vector<Tensor> getItem(size_t idx) override;

 private:
  static int32_t toInt32(const char* p);
  void loadImages(const std::string& path);
  void loadLabels(const std::string& path);

  std::vector<std::vector<float>> images_;
  std::vector<float> labels_;
  int32_t height_ = 0;
  int32_t width_ = 0;
  size_t size_ = 0;

  std::shared_ptr<transforms::Transform> transform_;
};

class DataLoader {
 public:
  DataLoader(const std::shared_ptr<Dataset>& dataset, size_t batchSize,
             bool shuffle = false, bool dropLast = false)
      : dataset_(dataset), batchSize_(batchSize), shuffle_(shuffle) {
    if (dropLast) {
      batchCnt_ = dataset->size() / batchSize;
    } else {
      batchCnt_ = (size_t)std::ceil((float)dataset->size() / (float)batchSize);
    }
    indices_.resize(dataset->size());
    std::iota(indices_.begin(), indices_.end(), 0);
    if (shuffle_) {
      std::shuffle(indices_.begin(), indices_.end(),
                   RandomGenerator::getGenerator());
    }
  }

  const Dataset& dataset() const { return *dataset_; }

  size_t batchSize() const { return batchSize_; }

  class Iterator {
   public:
    Iterator(const DataLoader& loader, size_t startIdx)
        : loader_(loader), batchIdx(startIdx) {}

    bool operator!=(const Iterator& other) const {
      return batchIdx != other.batchIdx;
    }

    bool operator==(const Iterator& other) const {
      return batchIdx == other.batchIdx;
    }

    Iterator& operator++() {
      batchIdx++;
      return *this;
    }

    static std::vector<std::reference_wrapper<TensorImpl>> getBatchRef(
        std::vector<TensorImpl>& batch) {
      std::vector<std::reference_wrapper<TensorImpl>> ret;
      ret.reserve(batch.size());
      for (auto& b : batch) {
        ret.emplace_back(std::ref(b));
      }
      return ret;
    }

    std::tuple<size_t, std::vector<Tensor>> operator*() const {
      std::vector<std::vector<TensorImpl>> itemList;
      for (size_t i = batchIdx * loader_.batchSize_;
           i < batchIdx * loader_.batchSize_ + loader_.batchSize_ &&
           i < loader_.indices_.size();
           i++) {
        if (i >= loader_.indices_.size()) {
          continue;
        }
        auto item = loader_.dataset_->getItem(loader_.indices_[i]);
        itemList.resize(item.size());
        for (size_t j = 0; j < item.size(); j++) {
          itemList[j].reserve(loader_.batchSize_);
          itemList[j].emplace_back(item[j].data());
        }
      }

      std::vector<Tensor> batch;
      batch.reserve(loader_.batchSize_);
      for (auto& it : itemList) {
        batch.emplace_back(TensorImpl::stack(getBatchRef(it)));
      }
      return {batchIdx, batch};
    }

   private:
    const DataLoader& loader_;
    size_t batchIdx;
  };

  Iterator begin() const { return {*this, 0}; }

  Iterator end() const { return {*this, batchCnt_}; }

  size_t size() const { return batchCnt_; }

 protected:
  std::shared_ptr<Dataset> dataset_;
  size_t batchSize_;
  size_t batchCnt_;
  bool shuffle_;
  std::vector<size_t> indices_;
};

}  // namespace TinyTorch::data
