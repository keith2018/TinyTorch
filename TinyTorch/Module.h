/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Tensor.h"

namespace TinyTorch::nn {

class Module {
 public:
  virtual ~Module() = default;

  virtual Tensor forward(Tensor &x) { return {}; }
  virtual Tensor forward(Tensor &x1, Tensor &x2) { return {}; }

  virtual std::vector<Tensor *> parameters();
  virtual void resetParameters();
  virtual void zeroGrad();

  virtual Tensor operator()(Tensor &x) { return forward(x); }
  virtual Tensor operator()(Tensor &x1, Tensor &x2) { return forward(x1, x2); }

  void registerModules(const std::vector<Module *> &modules) {
    subModules_ = modules;
  }

  void eval() { train(false); }

  void train(bool mode = true) { setTraining(mode); }

 protected:
  virtual void setTraining(bool mode) { training_ = mode; }

  bool training_ = true;
  std::vector<Module *> subModules_;
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
    modules_.push_back(
        std::make_shared<ModuleType>(std::forward<ModuleType>(module)));
  }

  void pushBack(const std::shared_ptr<Module> &module) {
    modules_.emplace_back(module);
  }

  Tensor forward(Tensor &input) override;
  std::vector<Tensor *> parameters() override;
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
  Linear(int32_t inFeatures, int32_t outFeatures, bool bias = true);

  Tensor forward(Tensor &input) override;
  std::vector<Tensor *> parameters() override;
  void resetParameters() override;
  void zeroGrad() override;

  Tensor &Weights() { return weights_; }
  Tensor &Bias() { return bias_; }

 private:
  int32_t inFeatures_;
  int32_t outFeatures_;
  bool useBias_;
  Tensor weights_;
  Tensor bias_;
};

class Flatten : public Module {
 public:
  explicit Flatten(int32_t startDim = 0, int32_t endDim = -1)
      : startDim_(startDim), endDim_(endDim) {}

  Tensor forward(Tensor &input) override;

 private:
  int32_t startDim_;
  int32_t endDim_;
};

class Relu : public Module {
 public:
  Tensor forward(Tensor &input) override;
};

class Dropout : public Module {
 public:
  explicit Dropout(float p = 0.5f) : p_(p) {}

  Tensor forward(Tensor &x) override;

 private:
  float p_;
};

class Softmax : public Module {
 public:
  explicit Softmax(int32_t dim) : dim_(dim) {}

  Tensor forward(Tensor &x) override;

 private:
  int32_t dim_;
};

class LogSoftmax : public Module {
 public:
  explicit LogSoftmax(int32_t dim) : dim_(dim) {}

  Tensor forward(Tensor &x) override;

 private:
  int32_t dim_;
};

class MaxPool2D : public Module {
 public:
  explicit MaxPool2D(Size2D kernelSize,
                     std::optional<Size2D> stride = std::nullopt,
                     Size2D padding = 0)
      : kernelSize_(kernelSize),
        stride_(stride.has_value() ? stride.value() : kernelSize),
        padding_(padding) {}

  Tensor forward(Tensor &x) override;

 private:
  Size2D kernelSize_;
  Size2D stride_;
  Size2D padding_;
};

class Conv2D : public Module {
 public:
  Conv2D(int32_t inFeatures, int32_t outFeatures, Size2D kernelSize,
         Size2D stride = 1, Size2D padding = 0, bool bias = true);

  Tensor forward(Tensor &input) override;
  std::vector<Tensor *> parameters() override;
  void resetParameters() override;
  void zeroGrad() override;

  Tensor &Weights() { return weights_; }
  Tensor &Bias() { return bias_; }

 private:
  int32_t inFeatures_;
  int32_t outFeatures_;
  Size2D kernelSize_;
  Size2D stride_;
  Size2D padding_;
  bool useBias_;
  Tensor weights_;
  Tensor bias_;
};

}  // namespace TinyTorch::nn
