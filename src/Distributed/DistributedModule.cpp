/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "DistributedModule.h"

namespace tinytorch::distributed {

DistributedModule::DistributedModule(Module& module, std::shared_ptr<DistributedProcessGroup> processGroup,
                                     int rootRank, int64_t bucketBytesCap)
    : module_(module),
      processGroup_(processGroup ? std::move(processGroup) : DistributedProcessGroup::getInstance()),
      rootRank_(rootRank) {
  registerModule("model", module_);
  auto parameters = module_.parameters();

  reducer_ = std::make_unique<Reducer>(std::move(parameters), processGroup_, bucketBytesCap);
  reducer_->broadcastParameters(rootRank_);
}

Tensor DistributedModule::forward(const Tensor& input) {
  ensurePreviousSyncComplete();

  if (!forwardStarted_.load()) {
    reducer_->prepareForBackward();
    forwardStarted_.store(true);
    backwardInProgress_.store(false);
  }
  return module_.forward(input);
}

void DistributedModule::zeroGrad() {
  ensurePreviousSyncComplete();
  Module::zeroGrad();

  forwardStarted_.store(false);
  backwardInProgress_.store(false);
}

void DistributedModule::broadcastParameters(int rootRank) {
  ensurePreviousSyncComplete();

  rootRank_ = rootRank;
  reducer_->broadcastParameters(rootRank);
}

void DistributedModule::synchronizeGradients() const {
  reducer_->synchronizeGradients();
  notifySyncComplete();
}

bool DistributedModule::hasUnfinishedGradientSync() const { return reducer_->hasUnfinishedOperations(); }

bool DistributedModule::allGradientsReady() const { return reducer_->allGradientsReady(); }

void DistributedModule::ensurePreviousSyncComplete() const {
  if (hasUnfinishedGradientSync()) {
    {
      std::lock_guard<std::mutex> lock(syncStateMutex_);
      waitingForSync_.store(true);
    }

    reducer_->waitForAllOperations();
    notifySyncComplete();
  }
}

void DistributedModule::notifySyncComplete() const {
  std::lock_guard<std::mutex> lock(syncStateMutex_);
  waitingForSync_.store(false);
  syncStateCV_.notify_all();
}

}  // namespace tinytorch::distributed
