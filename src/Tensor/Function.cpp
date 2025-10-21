/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Function.h"

#include "AutogradMeta.h"

namespace tinytorch {

void FuncLeaf::backward(const Tensor& grad) {
  auto owner = weakOwner.lock();
  if (owner == nullptr) {
    ASSERT(false && "backward error: FuncLeaf no owner");
    return;
  }

  owner->setGrad(grad);
  owner->applyHooks();
}

}  // namespace tinytorch
