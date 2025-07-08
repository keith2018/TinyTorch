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

void AutogradMeta::backward(const Tensor &grad) {
  ASSERT(grad.defined());

  if (gradFn_ == nullptr) {
    LOGE("error call backward: gradFunc == nullptr");
    return;
  }

  if (backwardGraph_.empty()) {
    buildBackwardGraph();
  }

  std::unordered_map<FunctionBase *, Tensor> inputs;
  inputs[gradFn_.get()] = grad;

  for (auto &fn : backwardGraph_) {
    auto outputs = fn->backward(inputs[fn.get()]);
    ASSERT(outputs.size() == fn->nextFunctions.size());

    for (auto i = 0; i < fn->nextFunctions.size(); i++) {
      auto nextFn = fn->nextFunctions[i].lock();
      ASSERT(nextFn != nullptr);
      auto it = inputs.find(nextFn.get());
      if (it != inputs.end()) {
        op::addInplace(it->second, outputs[i], 1);
      } else {
        inputs[nextFn.get()] = std::move(outputs[i]);
      }
    }
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
