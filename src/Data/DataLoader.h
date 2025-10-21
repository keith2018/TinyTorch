/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Dataset.h"
#include "Operations.h"
#include "Sampler.h"
#include "Tensor.h"

namespace tinytorch::data {

class DataLoader {
 public:
  DataLoader(const std::shared_ptr<Dataset>& dataset, size_t batchSize,
             const std::shared_ptr<Sampler>& sampler = nullptr, bool shuffle = true, bool dropLast = false)
      : dataset_(dataset), batchSize_(batchSize) {
    if (sampler) {
      indices_ = sampler->indices();
    } else {
      indices_.resize(dataset->size());
      std::iota(indices_.begin(), indices_.end(), 0);
      if (shuffle) {
        std::shuffle(indices_.begin(), indices_.end(), RandomGeneratorCPU::getGenerator());
      }
    }

    if (dropLast) {
      batchCnt_ = indices_.size() / batchSize;
    } else {
      batchCnt_ = std::ceil(static_cast<float>(indices_.size()) * 1.0f / static_cast<float>(batchSize));
    }
  }

  const Dataset& dataset() const { return *dataset_; }
  size_t batchSize() const { return batchSize_; }

  class Iterator {
   public:
    Iterator(const DataLoader& loader, size_t startIdx) : loader_(loader), batchIdx(startIdx) {}
    bool operator!=(const Iterator& other) const { return batchIdx != other.batchIdx; }
    Iterator& operator++() {
      batchIdx++;
      return *this;
    }

    std::tuple<size_t, std::vector<Tensor>> operator*() const {
      std::vector<std::vector<Tensor>> itemList;
      for (size_t i = batchIdx * loader_.batchSize_;
           i < batchIdx * loader_.batchSize_ + loader_.batchSize_ && i < loader_.indices_.size(); i++) {
        auto item = loader_.dataset_->getItem(loader_.indices_[i]);
        itemList.resize(item.size());
        for (size_t j = 0; j < item.size(); j++) {
          itemList[j].emplace_back(item[j]);
        }
      }
      std::vector<Tensor> batch;
      batch.reserve(itemList.size());
      for (auto& it : itemList) {
        batch.emplace_back(op::stack(ArrayView<Tensor>(it), 0));
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

 private:
  std::shared_ptr<Dataset> dataset_;
  size_t batchSize_;
  size_t batchCnt_;
  std::vector<size_t> indices_;
};

}  // namespace tinytorch::data
