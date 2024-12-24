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

  virtual Tensor forward(const Tensor &x) const { return {}; }
  virtual Tensor forward(const Tensor &x1, const Tensor &x2) const {
    return {};
  }

  virtual std::vector<Tensor *> parameters();
  virtual void resetParameters();
  virtual void zeroGrad();

  virtual Tensor operator()(const Tensor &x) const {
    if (!training_) {
      withNoGrad { return forward(x); }
    }
    return forward(x);
  }
  virtual Tensor operator()(const Tensor &x1, const Tensor &x2) const {
    if (!training_) {
      withNoGrad { return forward(x1, x2); }
    }
    return forward(x1, x2);
  }

  void registerModules(const std::vector<Module *> &modules) {
    subModules_ = modules;
  }

  void eval() { train(false); }

  void train(bool mode = true) { setTraining(mode); }

 protected:
  virtual void setTraining(bool mode) { training_ = mode; }

 protected:
  bool training_ = true;
  std::vector<Module *> subModules_;
};

class Sequential : public Module {
 public:
  template <typename... Args>
  explicit Sequential(Args... args) : modules_{args...} {}

  Tensor forward(const Tensor &input) const override;
  std::vector<Tensor *> parameters() override;
  void resetParameters() override;
  void zeroGrad() override;

  Module &operator[](const int index) { return *modules_[index]; }

 protected:
  void setTraining(bool mode) override;

 private:
  std::vector<std::shared_ptr<Module>> modules_;
};

class Linear : public Module {
 public:
  Linear(int32_t inFeatures, int32_t outFeatures, bool bias = true);

  Tensor forward(const Tensor &input) const override;
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

  Tensor forward(const Tensor &input) const override;

 private:
  int32_t startDim_;
  int32_t endDim_;
};

class Relu : public Module {
 public:
  Tensor forward(const Tensor &input) const override;
};

class Dropout : public Module {
 public:
  explicit Dropout(float p = 0.5f) : p_(p) {}

  Tensor forward(const Tensor &x) const override;

 private:
  float p_;
};

class Softmax : public Module {
 public:
  explicit Softmax(int32_t dim) : dim_(dim) {}

  Tensor forward(const Tensor &x) const override;

 private:
  int32_t dim_;
};

class LogSoftmax : public Module {
 public:
  explicit LogSoftmax(int32_t dim) : dim_(dim) {}

  Tensor forward(const Tensor &x) const override;

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

  Tensor forward(const Tensor &x) const override;

 private:
  Size2D kernelSize_;
  Size2D stride_;
  Size2D padding_;
};

class Conv2D : public Module {
 public:
  Conv2D(int32_t inFeatures, int32_t outFeatures, Size2D kernelSize,
         Size2D stride = 1, Size2D padding = 0, bool bias = true);

  Tensor forward(const Tensor &input) const override;
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
