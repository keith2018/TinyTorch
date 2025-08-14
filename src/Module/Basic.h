/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <optional>

#include "Module.h"
#include "Operation/OpNNLayer.h"

namespace tinytorch::nn {

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

class RoPE : public Module {
 public:
  explicit RoPE(int64_t headDim, int64_t contextLength = 4096, float thetaBase = 10000.0f,
                std::optional<RopeScalingConfig> scaling = std::nullopt, Options options = {});

  Tensor forward(const Tensor &input) override;
  void resetParameters() override;

  TensorPair &rope() { return rope_; }

  Tensor &cos() { return rope_.first; }
  Tensor &sin() { return rope_.second; }

 private:
  int64_t headDim_;
  int64_t contextLength_;
  float thetaBase_;
  std::optional<RopeScalingConfig> scaling_;
  Options options_;

  TensorPair rope_;
};

}  // namespace tinytorch::nn
