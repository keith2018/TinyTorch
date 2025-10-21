/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <atomic>

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
  void zeroGrad() override;

  void broadcastParameters(int rootRank = 0);
  void synchronizeGradients() const;

  bool hasUnfinishedGradientSync() const;
  bool allGradientsReady() const;

 private:
  void ensurePreviousSyncComplete() const;
  void notifySyncComplete() const;

  Module& module_;
  std::unique_ptr<Reducer> reducer_;
  std::shared_ptr<DistributedProcessGroup> processGroup_;
  int rootRank_;

  mutable std::atomic<bool> forwardStarted_{false};
  mutable std::atomic<bool> backwardInProgress_{false};

  mutable std::mutex syncStateMutex_;
  mutable std::condition_variable syncStateCV_;
  mutable std::atomic<bool> waitingForSync_{false};
};

}  // namespace tinytorch::distributed
