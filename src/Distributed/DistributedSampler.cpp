/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "DistributedSampler.h"

#include <algorithm>

#include "DistributedProcessGroup.h"

namespace tinytorch::distributed {

DistributedSampler::DistributedSampler(size_t datasetSize, std::optional<size_t> numReplicas,
                                       std::optional<size_t> rank, bool shuffle, unsigned long seed, bool dropLast)
    : datasetSize_(datasetSize), shuffle_(shuffle), seed_(seed), dropLast_(dropLast) {
  auto dpg = DistributedProcessGroup::getInstance();
  if (numReplicas.has_value()) {
    numReplicas_ = numReplicas.value();
  } else {
    numReplicas_ = dpg->getWorldSize();
  }
  if (rank.has_value()) {
    rank_ = rank.value();
  } else {
    rank_ = dpg->getRank();
  }

  if (dropLast_) {
    numSamples_ = datasetSize_ / numReplicas_;
  } else {
    numSamples_ = std::ceil(static_cast<float>(datasetSize_) * 1.0f / static_cast<float>(numReplicas_));
  }
  totalSize_ = numSamples_ * numReplicas_;
  generateIndices();
}

void DistributedSampler::generateIndices() {
  std::vector<size_t> allIndices(datasetSize_);
  std::iota(allIndices.begin(), allIndices.end(), 0);

  if (shuffle_) {
    auto& gen = RandomGeneratorCPU::getGenerator();
    gen.seed(static_cast<unsigned int>(seed_ + epoch_));
    std::shuffle(allIndices.begin(), allIndices.end(), gen);
  }

  if (!dropLast_) {
    if (allIndices.size() < totalSize_) {
      allIndices.insert(allIndices.end(), allIndices.begin(),
                        allIndices.begin() + static_cast<int64_t>(totalSize_ - allIndices.size()));
    }
  } else {
    allIndices.resize(totalSize_);
  }

  auto offset = static_cast<int64_t>(numSamples_ * rank_);
  indices_.assign(allIndices.begin() + offset, allIndices.begin() + offset + static_cast<int64_t>(numSamples_));
}

}  // namespace tinytorch::distributed