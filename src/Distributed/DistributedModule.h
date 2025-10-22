/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "DistributedProcessGroup.h"
#include "Modules.h"
#include "Reducer.h"

namespace tinytorch::distributed {

class DistributedModule : public nn::Module {
 public:
  explicit DistributedModule(Module& module, std::shared_ptr<DistributedProcessGroup> processGroup = nullptr,
                             int rootRank = 0, int64_t bucketBytesCap = kDefaultBucketBytesCap);

  ~DistributedModule() override = default;

  Tensor forward(const Tensor& input) override;

 private:
  Module& module_;
  std::unique_ptr<Reducer> reducer_;
  std::shared_ptr<DistributedProcessGroup> processGroup_;
  int rootRank_;
};

}  // namespace tinytorch::distributed
