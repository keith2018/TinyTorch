/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <algorithm>
#include <numeric>

#include "Utils/RandomGenerator.h"

namespace tinytorch::data {

class Sampler {
 public:
  virtual ~Sampler() = default;
  virtual std::vector<size_t> indices() const = 0;
  virtual size_t size() const = 0;
  virtual void setEpoch(size_t epoch) {}
};

class SequentialSampler : public Sampler {
 public:
  explicit SequentialSampler(size_t datasetSize) {
    indices_.resize(datasetSize);
    std::iota(indices_.begin(), indices_.end(), 0);
  }
  std::vector<size_t> indices() const override { return indices_; }
  size_t size() const override { return indices_.size(); }

 private:
  std::vector<size_t> indices_;
};

class RandomSampler : public Sampler {
 public:
  explicit RandomSampler(size_t datasetSize, unsigned long seed = 0) : datasetSize_(datasetSize), seed_(seed) {
    generateIndices();
  }

  std::vector<size_t> indices() const override { return indices_; }
  size_t size() const override { return indices_.size(); }

  void setEpoch(size_t epoch) override {
    epoch_ = epoch;
    generateIndices();
  }

 private:
  void generateIndices() {
    indices_.resize(datasetSize_);
    std::iota(indices_.begin(), indices_.end(), 0);

    auto& gen = RandomGeneratorCPU::getGenerator();
    gen.seed(static_cast<unsigned int>(seed_ + epoch_));
    std::shuffle(indices_.begin(), indices_.end(), gen);
  }

  size_t datasetSize_;
  unsigned long seed_;
  size_t epoch_ = 0;
  std::vector<size_t> indices_;
};

}  // namespace tinytorch::data
