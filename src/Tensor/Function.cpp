/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Function.h"

#include "AutogradMeta.h"

namespace tinytorch {

TensorList FuncLeaf::backward(const Tensor& grad) {
  auto owner = weakOwner.lock();
  if (owner == nullptr) {
    ASSERT(false);
    return {};
  }

  owner->grad_ = grad;
  return {};
}

}  // namespace tinytorch
