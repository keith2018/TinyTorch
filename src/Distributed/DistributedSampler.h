/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Data/Sampler.h"

namespace tinytorch::distributed {

class DistributedSampler : public data::Sampler {
 public:
  explicit DistributedSampler(size_t datasetSize, std::optional<size_t> numReplicas = std::nullopt,
                              std::optional<size_t> rank = std::nullopt, bool shuffle = true, unsigned long seed = 0,
                              bool dropLast = false);

  std::vector<size_t> indices() const override { return indices_; }
  size_t size() const override { return indices_.size(); }

  void setEpoch(size_t epoch) override {
    epoch_ = epoch;
    generateIndices();
  }

 private:
  void generateIndices();

  size_t datasetSize_ = 0;
  size_t numReplicas_ = 0;
  size_t rank_ = 0;
  bool shuffle_;
  unsigned long seed_;
  bool dropLast_;
  size_t epoch_ = 0;
  size_t numSamples_;
  size_t totalSize_;
  std::vector<size_t> indices_;
};

}  // namespace tinytorch::distributed
