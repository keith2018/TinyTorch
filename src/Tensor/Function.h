/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <deque>
#include <type_traits>
#include <vector>

#include "IValue.h"
#include "NoGradGuard.h"

namespace tinytorch {

class AutogradMeta;

struct AutogradContext {
  std::deque<IValue> savedData;
  TensorList savedInputs;

  void pushData(IValue&& val) { savedData.push_back(std::move(val)); }

  IValue popData() {
    ASSERT(!savedData.empty());
    IValue ret = std::move(savedData.front());
    savedData.pop_front();
    return ret;
  }

  explicit AutogradContext(TensorList&& inputs) : savedInputs(std::move(inputs)) {}
};

struct FunctionBase : std::enable_shared_from_this<FunctionBase> {
  virtual ~FunctionBase() = default;
  virtual void backward(const Tensor& grad) = 0;

  std::vector<std::weak_ptr<FunctionBase>> nextFunctions;
  std::shared_ptr<AutogradContext> ctx;
  std::weak_ptr<AutogradMeta> weakOwner;
};

template <typename X, typename... Args>
using forward_t = decltype(X::forward(nullptr, std::declval<Args>()...));

template <typename>
constexpr bool always_false = false;

template <class T>
struct Function {
  struct FunctionInstance : FunctionBase {
    explicit FunctionInstance(std::shared_ptr<AutogradContext> c) { ctx = std::move(c); }
    void backward(const Tensor& grad) override { T::backward(ctx.get(), grad); }
  };

  template <typename X = T, typename... Args>
  static auto apply(Args&&... args) -> std::enable_if_t<std::is_same_v<X, T>, forward_t<X, Args...>>;
};

inline void collectTensorInfo(const Tensor& t, bool& requiresGrad, TensorList& inputTensors,
                              std::vector<std::weak_ptr<FunctionBase>>& inputFns) {
  inputTensors.push_back(t);
  if (t.defined() && t.requiresGrad()) {
    requiresGrad = true;
    inputFns.push_back(t.gradFn());
  }
}

inline void collectTensorInfo(const TensorPair& tp, bool& requiresGrad, TensorList& inputTensors,
                              std::vector<std::weak_ptr<FunctionBase>>& inputFns) {
  collectTensorInfo(tp.first, requiresGrad, inputTensors, inputFns);
  collectTensorInfo(tp.second, requiresGrad, inputTensors, inputFns);
}

inline void collectTensorInfo(const TensorListView& tlv, bool& requiresGrad, TensorList& inputTensors,
                              std::vector<std::weak_ptr<FunctionBase>>& inputFns) {
  for (const auto& t : tlv) {
    collectTensorInfo(t, requiresGrad, inputTensors, inputFns);
  }
}

template <typename TensorType>
void handleOutputTensor(TensorType& t, bool shouldRequireGrad, const std::shared_ptr<FunctionBase>& fn) {
  if (shouldRequireGrad) {
    t.setRequiresGrad(true, fn);
  } else {
    ASSERT(!t.requiresGrad());
  }
}

template <>
inline void handleOutputTensor(TensorPair& tp, bool shouldRequireGrad, const std::shared_ptr<FunctionBase>& fn) {
  handleOutputTensor(tp.first, shouldRequireGrad, fn);
  handleOutputTensor(tp.second, shouldRequireGrad, fn);
}

template <>
inline void handleOutputTensor(TensorList& tl, bool shouldRequireGrad, const std::shared_ptr<FunctionBase>& fn) {
  for (auto& t : tl) {
    handleOutputTensor(t, shouldRequireGrad, fn);
  }
}

template <class T>
template <typename X, typename... Args>
auto Function<T>::apply(Args&&... args) -> std::enable_if_t<std::is_same_v<X, T>, forward_t<X, Args...>> {
  bool requiresGrad = false;
  TensorList inputTensors;
  std::vector<std::weak_ptr<FunctionBase>> inputFns;

  if (NoGradGuard::isGradEnabled()) {
    inputTensors.reserve(sizeof...(Args));
    inputFns.reserve(sizeof...(Args));

    auto processArg = [&](auto&& arg) {
      using U = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<U, Tensor> || std::is_same_v<U, TensorPair> || std::is_same_v<U, TensorListView>) {
        collectTensorInfo(arg, requiresGrad, inputTensors, inputFns);
      }
    };
    (processArg(args), ...);
  }

  using OutputType = forward_t<X, Args...>;
  auto ctx = requiresGrad ? std::make_shared<AutogradContext>(std::move(inputTensors)) : nullptr;
  auto output = T::forward(ctx.get(), std::forward<Args>(args)...);

  std::shared_ptr<FunctionInstance> fnInst = nullptr;
  if (requiresGrad) {
    fnInst = std::make_shared<FunctionInstance>(std::move(ctx));
    fnInst->nextFunctions = std::move(inputFns);
  }

  if constexpr (std::is_same_v<OutputType, Tensor> || std::is_same_v<OutputType, TensorPair> ||
                std::is_same_v<OutputType, TensorList>) {
    handleOutputTensor(output, requiresGrad, fnInst);
  } else {
    static_assert(always_false<OutputType>, "Unsupported return type for T::forward");
  }
  return output;
}

class FuncLeaf : public FunctionBase {
 public:
  void backward(const Tensor& grad) override;
};

}  // namespace tinytorch
