/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "AutogradMeta.h"

#include <queue>
#include <set>

#include "Function.h"
#include "Operations.h"
#include "Tensor.h"

namespace tinytorch {

void AutogradMeta::setGradFn(const std::shared_ptr<FunctionBase> &fn) {
  ASSERT(fn != nullptr);
  gradFn_ = fn;
  fn->weakOwner = shared_from_this();
}

void AutogradMeta::setGrad(const Tensor &grad) { grad_ = grad; }

void AutogradMeta::setGrad(Tensor &&grad) { grad_ = std::move(grad); }

void AutogradMeta::addGrad(const Tensor &grad) {
  if (grad_.defined()) {
    op::addInplace(grad_, grad, 1);
  } else {
    grad_ = grad;
  }
}

void AutogradMeta::addGrad(Tensor &&grad) {
  if (grad_.defined()) {
    op::addInplace(grad_, grad, 1);
  } else {
    grad_ = std::move(grad);
  }
}

void AutogradMeta::zeroGrad(const Tensor &owner) {
  if (!grad_.defined()) {
    grad_ = Tensor::empty(owner.shape(), owner.options().noGrad());
  }
  op::fill(grad_, 0);
}

bool AutogradMeta::isLeaf() const {
  ASSERT(gradFn_ != nullptr);
  return typeid(*gradFn_) == typeid(FuncLeaf);
}

void AutogradMeta::backward(const Tensor &grad) {
  ASSERT(grad.defined());

  if (gradFn_ == nullptr) {
    LOGE("error call backward: gradFunc == nullptr");
    return;
  }

  if (backwardGraph_.empty()) {
    buildBackwardGraph();
  }

  addGrad(grad);
  for (auto &currFunc : backwardGraph_) {
    auto owner = currFunc->weakOwner.lock();
    ASSERT(owner != nullptr);
    currFunc->backward(owner->grad_);
  }
}

void AutogradMeta::buildBackwardGraph() {
  std::unordered_map<std::shared_ptr<FunctionBase>, int> deps;
  std::deque<std::shared_ptr<FunctionBase>> q;

  std::set<std::shared_ptr<FunctionBase>> traversed = {gradFn_};
  q.push_back(gradFn_);
  while (!q.empty()) {
    const auto curr = q.front();
    q.pop_front();
    for (const auto &next : curr->nextFunctions) {
      auto nextFn = next.lock();
      ASSERT(nextFn != nullptr);
      deps[nextFn] += 1;
      if (traversed.find(nextFn) == traversed.end()) {
        q.push_back(nextFn);
        traversed.insert(nextFn);
      }
    }
  }

  q.push_back(gradFn_);
  while (!q.empty()) {
    const auto currFunc = q.front();
    backwardGraph_.push_back(currFunc);
    q.pop_front();
    for (const auto &next : currFunc->nextFunctions) {
      auto nextFn = next.lock();
      ASSERT(nextFn != nullptr);
      deps[nextFn] -= 1;
      if (deps[nextFn] == 0) {
        q.push_back(nextFn);
      }
    }
  }
}

}  // namespace tinytorch
