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
  reducer_->prepareForBackward();
  return module_.forward(input);
}

}  // namespace tinytorch::distributed
