/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Module.h"

#include "Functions.h"
#include "Initializer.h"

namespace tinytorch::nn {

// NOLINTNEXTLINE(misc-no-recursion)
std::vector<TensorPtr> Module::parameters() {
  std::vector<TensorPtr> ret;
  for (auto &module : subModules_) {
    for (const auto &p : module.get().parameters()) {
      ret.push_back(p);
    }
  }
  return ret;
}

// NOLINTNEXTLINE(misc-no-recursion)
std::vector<TensorPtr> Module::states() {
  std::vector<TensorPtr> ret;
  for (auto &module : subModules_) {
    for (const auto &p : module.get().states()) {
      ret.push_back(p);
    }
  }
  return ret;
}

// NOLINTNEXTLINE(misc-no-recursion)
void Module::resetParameters() {
  for (auto &module : subModules_) {
    module.get().resetParameters();
  }
}

// NOLINTNEXTLINE(misc-no-recursion)
void Module::zeroGrad() {
  for (auto &module : subModules_) {
    module.get().zeroGrad();
  }
}

void Module::to(Device device) {
  for (auto &module : subModules_) {
    for (auto &p : module.get().states()) {
      *p = p->to(device);
    }
  }
}

Tensor Sequential::forward(Tensor &input) {
  Tensor ret = {input};
  for (auto &module : modules_) {
    ret = (*module)(ret);
  }

  return ret;
}

std::vector<TensorPtr> Sequential::parameters() {
  std::vector<TensorPtr> ret;
  for (auto &module : modules_) {
    for (const auto &p : module->parameters()) {
      ret.push_back(p);
    }
  }
  return ret;
}

std::vector<TensorPtr> Sequential::states() {
  std::vector<TensorPtr> ret;
  for (auto &module : modules_) {
    for (const auto &p : module->states()) {
      ret.push_back(p);
    }
  }
  return ret;
}

void Sequential::resetParameters() {
  for (auto &module : modules_) {
    module->resetParameters();
  }
}

void Sequential::zeroGrad() {
  for (auto &module : modules_) {
    module->zeroGrad();
  }
}

void Sequential::setTraining(bool mode) {
  Module::setTraining(mode);
  for (auto &module : modules_) {
    module->train(mode);
  }
}

Linear::Linear(int64_t inFeatures, int64_t outFeatures, bool bias) : useBias_(bias) {
  Options options = options::requiresGrad(true);
  weights_ = Tensor::empty({outFeatures, inFeatures}, options);
  if (bias) {
    bias_ = Tensor::empty({outFeatures}, options);
  }
  Linear::resetParameters();
}

Tensor Linear::forward(Tensor &input) { return function::linear(input, weights_, bias_); }

std::vector<TensorPtr> Linear::parameters() {
  if (useBias_) {
    return {&weights_, &bias_};
  }
  return {&weights_};
}

std::vector<TensorPtr> Linear::states() { return parameters(); }

void Linear::resetParameters() {
  Initializer::kaimingUniform(weights_, std::sqrt(5.f));
  if (useBias_) {
    auto fanIn = Initializer::calculateFan(weights_).first;
    const auto bound = fanIn > 0 ? 1.f / std::sqrt(static_cast<float>(fanIn)) : 0;
    Initializer::uniform(bias_, -bound, bound);
  }
}

void Linear::zeroGrad() {
  weights_.zeroGrad();
  if (useBias_) {
    bias_.zeroGrad();
  }
}

Tensor Flatten::forward(Tensor &input) { return function::flatten(input, startDim_, endDim_); }

Tensor Relu::forward(Tensor &input) { return function::relu(input); }

Tensor Dropout::forward(Tensor &input) { return function::dropout(input, p_, training_); }

Tensor Softmax::forward(Tensor &input) { return function::softmax(input, dim_); }

Tensor LogSoftmax::forward(Tensor &input) { return function::logSoftmax(input, dim_); }

Tensor MaxPool2D::forward(Tensor &input) { return function::maxPool2d(input, kernelSize_, stride_, padding_); }

Conv2D::Conv2D(int64_t inFeatures, int64_t outFeatures, Dim2D kernelSize, Dim2D stride, Dim2D padding, bool bias)
    : kernelSize_(kernelSize), stride_(stride), padding_(padding), useBias_(bias) {
  Options options = options::requiresGrad(true);
  weights_ = Tensor::empty({outFeatures, inFeatures, kernelSize_.h, kernelSize_.w}, options);
  if (bias) {
    bias_ = Tensor::empty({outFeatures}, options);
  }
  Conv2D::resetParameters();
}

Tensor Conv2D::forward(Tensor &input) { return function::conv2d(input, weights_, bias_, stride_, padding_); }

std::vector<TensorPtr> Conv2D::parameters() {
  if (useBias_) {
    return {&weights_, &bias_};
  }
  return {&weights_};
}

std::vector<TensorPtr> Conv2D::states() { return parameters(); }

void Conv2D::resetParameters() {
  Initializer::kaimingUniform(weights_, std::sqrt(5.f));
  if (useBias_) {
    auto fanIn = Initializer::calculateFan(weights_).first;
    if (fanIn != 0) {
      const auto bound = 1.f / std::sqrt(static_cast<float>(fanIn));
      Initializer::uniform(bias_, -bound, bound);
    }
  }
}

void Conv2D::zeroGrad() {
  weights_.zeroGrad();
  if (useBias_) {
    bias_.zeroGrad();
  }
}

}  // namespace tinytorch::nn
