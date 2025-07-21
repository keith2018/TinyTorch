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

template <class T>
template <typename X, typename... Args>
auto Function<T>::apply(Args&&... args) -> std::enable_if_t<std::is_same_v<X, T>, forward_t<X, Args...>> {
  bool requiresGrad = false;
  TensorList inputTensors;
  std::vector<std::weak_ptr<FunctionBase>> inputFns;

  if (NoGradGuard::isGradEnabled()) {
    inputTensors.reserve(sizeof...(Args));
    inputFns.reserve(sizeof...(Args));

    auto collect = [&](auto&& arg) {
      using U = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<U, Tensor>) {
        inputTensors.push_back(arg);
        if (arg.requiresGrad()) {
          requiresGrad = true;
          inputFns.push_back(arg.gradFn());
        }
      }
    };
    (collect(args), ...);
  }

  using OutputType = forward_t<X, Args...>;
  if (requiresGrad) {
    auto ctx = std::make_shared<AutogradContext>(std::move(inputTensors));
    auto output = T::forward(ctx.get(), std::forward<Args>(args)...);
    auto fn = std::make_shared<FunctionInstance>(ctx);
    fn->nextFunctions = std::move(inputFns);

    if constexpr (std::is_same_v<OutputType, Tensor>) {
      output.setRequiresGrad(true, fn);
      return output;
    } else if constexpr (std::is_same_v<OutputType, TensorPair>) {
      output.first.setRequiresGrad(true, fn);
      output.second.setRequiresGrad(true, fn);
      return output;
    } else {
      static_assert(always_false<OutputType>, "Unsupported return type for T::forward");
      return output;
    }
  }

  auto output = T::forward(nullptr, std::forward<Args>(args)...);
  if constexpr (std::is_same_v<OutputType, Tensor>) {
    ASSERT(!output.requiresGrad());
    return output;
  } else if constexpr (std::is_same_v<OutputType, TensorPair>) {
    ASSERT(!output.first.requiresGrad());
    ASSERT(!output.second.requiresGrad());
    return output;
  } else {
    static_assert(always_false<OutputType>, "Unsupported return type for T::forward");
    return output;
  }
}

class FuncLeaf : public FunctionBase {
 public:
  void backward(const Tensor& grad) override;
};

}  // namespace tinytorch
