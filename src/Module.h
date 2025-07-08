/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <optional>

#include "Tensor.h"

namespace tinytorch::nn {

class Module {
 public:
  virtual ~Module() = default;
  virtual std::vector<TensorPtr> parameters();
  virtual std::vector<TensorPtr> states();
  virtual void resetParameters();
  virtual void zeroGrad();

  virtual Tensor forward(Tensor &input) { return {}; }
  virtual Tensor forward(Tensor &input1, Tensor &input2) { return {}; }

  template <typename... Args>
  Tensor operator()(Args &&...args) {
    return forward(std::forward<Args>(args)...);
  }

  void registerModules(const std::vector<std::reference_wrapper<Module>> &modules) {
    subModules_.reserve(modules.size());
    for (auto module : modules) {
      subModules_.emplace_back(module.get());
    }
  }

  void registerModule(const std::reference_wrapper<Module> &module) { subModules_.push_back(module); }

  void to(Device device);

  void eval() { train(false); }

  void train(bool mode = true) { setTraining(mode); }

 protected:
  virtual void setTraining(bool mode) { training_ = mode; }

  bool training_ = true;
  std::vector<std::reference_wrapper<Module>> subModules_;
};

class Sequential : public Module {
 public:
  template <typename... Modules>
  explicit Sequential(Modules &&...modules) {
    modules_.reserve(sizeof...(Modules));
    pushBack(std::forward<Modules>(modules)...);
  }

  Sequential(std::initializer_list<std::shared_ptr<Module>> modules) {
    modules_.reserve(modules.size());
    for (const auto &module : modules) {
      modules_.emplace_back(module);
    }
  }

  template <typename ModuleType>
  void pushBack(ModuleType &&module) {
    modules_.push_back(std::make_shared<ModuleType>(std::forward<ModuleType>(module)));
  }

  void pushBack(const std::shared_ptr<Module> &module) { modules_.emplace_back(module); }

  Tensor forward(Tensor &input) override;
  std::vector<TensorPtr> parameters() override;
  std::vector<TensorPtr> states() override;
  void resetParameters() override;
  void zeroGrad() override;

  Module &operator[](const int index) { return *modules_[index]; }

 private:
  void setTraining(bool mode) override;

  template <typename First, typename Second, typename... Rest>
  void pushBack(First &&first, Second &&second, Rest &&...rest) {
    pushBack(std::forward<First>(first));
    pushBack(std::forward<Second>(second), std::forward<Rest>(rest)...);
  }

  void pushBack() {}

  std::vector<std::shared_ptr<Module>> modules_;
};

class Linear : public Module {
 public:
  Linear(int64_t inFeatures, int64_t outFeatures, bool bias = true);

  Tensor forward(Tensor &input) override;
  std::vector<TensorPtr> parameters() override;
  std::vector<TensorPtr> states() override;
  void resetParameters() override;
  void zeroGrad() override;

  Tensor &weights() { return weights_; }
  Tensor &bias() { return bias_; }

 private:
  int64_t inFeatures_;
  int64_t outFeatures_;
  bool useBias_;
  Tensor weights_;
  Tensor bias_;
};

class Flatten : public Module {
 public:
  explicit Flatten(int64_t startDim = 0, int64_t endDim = -1) : startDim_(startDim), endDim_(endDim) {}

  Tensor forward(Tensor &input) override;

 private:
  int64_t startDim_;
  int64_t endDim_;
};

class Relu : public Module {
 public:
  Tensor forward(Tensor &input) override;
};

class Dropout : public Module {
 public:
  explicit Dropout(float p = 0.5f) : p_(p) {}

  Tensor forward(Tensor &input) override;

 private:
  float p_;
};

class Softmax : public Module {
 public:
  explicit Softmax(int64_t dim) : dim_(dim) {}

  Tensor forward(Tensor &input) override;

 private:
  int64_t dim_;
};

class LogSoftmax : public Module {
 public:
  explicit LogSoftmax(int64_t dim) : dim_(dim) {}

  Tensor forward(Tensor &input) override;

 private:
  int64_t dim_;
};

class MaxPool2D : public Module {
 public:
  explicit MaxPool2D(Dim2D kernelSize, std::optional<Dim2D> stride = std::nullopt, Dim2D padding = 0)
      : kernelSize_(kernelSize), stride_(stride.has_value() ? stride.value() : kernelSize), padding_(padding) {}

  Tensor forward(Tensor &input) override;

 private:
  Dim2D kernelSize_;
  Dim2D stride_;
  Dim2D padding_;
};

class Conv2D : public Module {
 public:
  Conv2D(int64_t inFeatures, int64_t outFeatures, Dim2D kernelSize, Dim2D stride = 1, Dim2D padding = 0,
         bool bias = true);

  Tensor forward(Tensor &input) override;
  std::vector<TensorPtr> parameters() override;
  std::vector<TensorPtr> states() override;
  void resetParameters() override;
  void zeroGrad() override;

  Tensor &weights() { return weights_; }
  Tensor &bias() { return bias_; }

 private:
  int64_t inFeatures_;
  int64_t outFeatures_;
  Dim2D kernelSize_;
  Dim2D stride_;
  Dim2D padding_;
  bool useBias_;

  Tensor weights_;
  Tensor bias_;
};

}  // namespace tinytorch::nn
