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
  Module() = default;
  virtual ~Module() = default;

  void registerModules(const std::vector<std::pair<std::string, std::reference_wrapper<Module>>> &modules) {
    subModules_.reserve(modules.size());
    for (auto &pair : modules) {
      subModules_.emplace_back(pair);
    }
  }

  void registerModule(const std::string &name, std::reference_wrapper<Module> module) {
    subModules_.emplace_back(name, module);
  }

  // NOLINTNEXTLINE(misc-no-recursion)
  std::vector<std::pair<std::string, TensorPtr>> namedParameters(const std::string &prefix = "") {
    std::vector<std::pair<std::string, TensorPtr>> ret;
    for (const auto &[name, param] : namedParameters_()) {
      ret.emplace_back(prefix + name, param);
    }
    for (const auto &[name, module] : subModules_) {
      auto subParams = module.get().namedParameters(prefix + name + ".");
      ret.insert(ret.end(), subParams.begin(), subParams.end());
    }
    return ret;
  }

  // NOLINTNEXTLINE(misc-no-recursion)
  std::vector<std::pair<std::string, TensorPtr>> namedStates(const std::string &prefix = "") {
    std::vector<std::pair<std::string, TensorPtr>> ret;
    for (const auto &[name, state] : namedStates_()) {
      ret.emplace_back(prefix + name, state);
    }
    for (const auto &[name, module] : subModules_) {
      auto subStates = module.get().namedStates(prefix + name + ".");
      ret.insert(ret.end(), subStates.begin(), subStates.end());
    }
    return ret;
  }

  std::vector<TensorPtr> parameters() {
    auto allParameters = namedParameters();
    std::vector<TensorPtr> ret;
    ret.reserve(allParameters.size());
    for (const auto &[name, param] : allParameters) {
      ret.emplace_back(param);
    }
    return ret;
  }

  std::vector<TensorPtr> states() {
    auto allStates = namedStates();
    std::vector<TensorPtr> ret;
    ret.reserve(allStates.size());
    for (const auto &[name, state] : allStates) {
      ret.emplace_back(state);
    }
    return ret;
  }

  void initParameters() { resetParameters(); }

  // NOLINTNEXTLINE(misc-no-recursion)
  virtual void resetParameters() {
    for (auto &[name, module] : subModules_) {
      module.get().resetParameters();
    }
  }

  // NOLINTNEXTLINE(misc-no-recursion)
  virtual void zeroGrad() {
    auto allParameters = namedParameters();
    for (auto &[name, param] : allParameters) {
      param->zeroGrad();
    }
  }

  virtual Tensor forward(const Tensor &input) { return {}; }
  Tensor operator()(const Tensor &input) { return forward(input); }

  void to(Device device) {
    auto allStates = namedStates();
    for (auto &[name, state] : allStates) {
      *state = state->to(device);
    }
  }

  void eval() { train(false); }

  void train(bool mode = true) { setTraining(mode); }

 protected:
  // NOLINTNEXTLINE(misc-no-recursion)
  virtual void setTraining(bool mode) {
    training_ = mode;
    for (auto &[name, module] : subModules_) {
      module.get().setTraining(mode);
    }
  }
  virtual std::vector<std::pair<std::string, TensorPtr>> namedParameters_() { return {}; }
  virtual std::vector<std::pair<std::string, TensorPtr>> namedStates_() { return namedParameters_(); }

  bool training_ = true;
  std::vector<std::pair<std::string, std::reference_wrapper<Module>>> subModules_;
};

class ModuleList : public Module {
 public:
  ModuleList() = default;

  ModuleList(std::initializer_list<std::shared_ptr<Module>> modules) {
    modules_.reserve(modules.size());
    size_t idx = 0;
    for (const auto &module : modules) {
      registerModule(std::to_string(idx++), *module);
      modules_.emplace_back(module);
    }
  }

  template <typename ModuleType, typename... Args>
  void emplaceBack(Args &&...args) {
    auto ptr = std::make_shared<ModuleType>(std::forward<Args>(args)...);
    registerModule(std::to_string(modules_.size()), *ptr);
    modules_.push_back(ptr);
  }

  void pushBack(const std::shared_ptr<Module> &module) {
    registerModule(std::to_string(modules_.size()), *module);
    modules_.emplace_back(module);
  }

  Module &operator[](size_t idx) {
    ASSERT(idx < modules_.size());
    return *modules_[idx];
  }
  const Module &operator[](size_t idx) const {
    ASSERT(idx < modules_.size());
    return *modules_[idx];
  }
  size_t size() const { return modules_.size(); }
  bool empty() const { return modules_.empty(); }

  auto begin() { return modules_.begin(); }
  auto end() { return modules_.end(); }

  auto begin() const { return modules_.begin(); }
  auto end() const { return modules_.end(); }

 protected:
  std::vector<std::shared_ptr<Module>> modules_;
};

class Sequential : public ModuleList {
 public:
  Sequential(std::initializer_list<std::shared_ptr<Module>> modules) : ModuleList(modules) {}

  Tensor forward(const Tensor &input) override;
};

class Linear : public Module {
 public:
  Linear(int64_t inFeatures, int64_t outFeatures, bool bias = true, Options options = {});

  Tensor forward(const Tensor &input) override;
  void resetParameters() override;

  Tensor &weight() { return weight_; }
  Tensor &bias() { return bias_; }

 protected:
  std::vector<std::pair<std::string, TensorPtr>> namedParameters_() override;

 private:
  bool useBias_;
  Tensor weight_;
  Tensor bias_;
};

class Flatten : public Module {
 public:
  explicit Flatten(int64_t startDim = 0, int64_t endDim = -1) : startDim_(startDim), endDim_(endDim) {}

  Tensor forward(const Tensor &input) override;

 private:
  int64_t startDim_;
  int64_t endDim_;
};

class Relu : public Module {
 public:
  Tensor forward(const Tensor &input) override;
};

class Gelu : public Module {
 public:
  Tensor forward(const Tensor &input) override;
};

class Silu : public Module {
 public:
  Tensor forward(const Tensor &input) override;
};

class Dropout : public Module {
 public:
  explicit Dropout(float p = 0.5f) : p_(p) {}

  Tensor forward(const Tensor &input) override;

 private:
  float p_;
};

class Softmax : public Module {
 public:
  explicit Softmax(int64_t dim) : dim_(dim) {}

  Tensor forward(const Tensor &input) override;

 private:
  int64_t dim_;
};

class LogSoftmax : public Module {
 public:
  explicit LogSoftmax(int64_t dim) : dim_(dim) {}

  Tensor forward(const Tensor &input) override;

 private:
  int64_t dim_;
};

class MaxPool2D : public Module {
 public:
  explicit MaxPool2D(Dim2D kernel, std::optional<Dim2D> stride = std::nullopt, Dim2D padding = 0)
      : kernel_(kernel), stride_(stride.has_value() ? stride.value() : kernel), padding_(padding) {}

  Tensor forward(const Tensor &input) override;

 private:
  Dim2D kernel_;
  Dim2D stride_;
  Dim2D padding_;
};

class Conv2D : public Module {
 public:
  Conv2D(int64_t inFeatures, int64_t outFeatures, Dim2D kernel, Dim2D stride = 1, Dim2D padding = 0, bool bias = true,
         Options options = {});

  Tensor forward(const Tensor &input) override;
  void resetParameters() override;

  Tensor &weight() { return weight_; }
  Tensor &bias() { return bias_; }

 protected:
  std::vector<std::pair<std::string, TensorPtr>> namedParameters_() override;

 private:
  Dim2D kernel_;
  Dim2D stride_;
  Dim2D padding_;
  bool useBias_;

  Tensor weight_;
  Tensor bias_;
};

class Embedding : public Module {
 public:
  Embedding(int64_t numEmbeddings, int64_t embeddingDim, Options options = {});

  Tensor forward(const Tensor &input) override;
  void resetParameters() override;

  Tensor &weight() { return weight_; }

 protected:
  std::vector<std::pair<std::string, TensorPtr>> namedParameters_() override;

 private:
  Tensor weight_;
};

class LayerNorm : public Module {
 public:
  explicit LayerNorm(IntArrayView normalizedShape, float eps = 1e-5, bool bias = true, Options options = {});

  Tensor forward(const Tensor &input) override;
  void resetParameters() override;

  Tensor &weight() { return weight_; }
  Tensor &bias() { return bias_; }

 protected:
  std::vector<std::pair<std::string, TensorPtr>> namedParameters_() override;

 private:
  SizeVector normalizedShape_;
  float eps_;
  bool useBias_;
  Tensor weight_;
  Tensor bias_;
};

}  // namespace tinytorch::nn
