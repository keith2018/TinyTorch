/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Operations.h"
#include "Tensor/Function.h"

namespace tinytorch::function {

class FuncLinear : public Function<FuncLinear> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& input, const Tensor& weight, const Tensor& bias) {
    auto output = op::matmul(input, weight, false, true);
    if (bias.defined()) {
      op::addInplace(output, bias, 1);
    }
    // TODO fuse
    return output;
  }

  static void backward(AutogradContext* ctx, const Tensor& grad) {
    auto& input = ctx->savedInputs[0];
    auto& weight = ctx->savedInputs[1];
    auto& bias = ctx->savedInputs[2];

    if (input.requiresGrad()) {
      input.addGrad(op::matmul(grad, weight, false, false));
    }
    if (weight.requiresGrad()) {
      weight.addGrad(op::matmul(grad, input, true, false));
    }
    if (bias.defined() && bias.requiresGrad()) {
      bias.addGrad(op::sumOnDim(grad, 0, false));
    }
  }
};

class FuncDropout : public Function<FuncDropout> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& input, float p, bool training) {
    if (ctx) {
      ctx->pushData(training);
    }
    if (!training) {
      return input.clone();
    }
    auto mask = Tensor::bernoulli(input.shape(), 1.f - p, input.options().noGrad());
    auto output = op::dropoutMasked(input, mask, 1.f - p);
    if (ctx) {
      ctx->pushData(p);
      ctx->pushData(mask);
    }
    // TODO fuse
    return output;
  }

  static void backward(AutogradContext* ctx, const Tensor& grad) {
    auto& input = ctx->savedInputs[0];

    if (input.requiresGrad()) {
      auto training = ctx->popData().toBool();
      if (training) {
        auto p = ctx->popData().toFloat();
        auto mask = ctx->popData().toTensor();
        input.addGrad(op::dropoutMasked(grad, mask, 1.f - p));
      } else {
        input.addGrad(grad);
      }
    }
  }
};

class FuncMaxPool : public Function<FuncMaxPool> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& input, Dim2D kernel, Dim2D stride, Dim2D padding) {
    auto shape = input.shape();
    ASSERT(shape.size() == 3 || shape.size() == 4);

    int64_t batch = (shape.size() == 4) ? shape[0] : 1;
    int64_t channels = (shape.size() == 4) ? shape[1] : shape[0];
    int64_t height = (shape.size() == 4) ? shape[2] : shape[1];
    int64_t width = (shape.size() == 4) ? shape[3] : shape[2];

    auto outH = (height - kernel.h + 2 * padding.h) / stride.h + 1;
    auto outW = (width - kernel.w + 2 * padding.w) / stride.w + 1;

    auto col = op::im2col(input, kernel, stride, padding);
    col.reshape_({-1, kernel.h * kernel.w});

    auto maxRet = op::maxOnDim(col, 1, false);
    auto maxIndices = maxRet.second;

    auto output = maxRet.first;
    output.reshape_({batch, channels, outH, outW});

    if (ctx) {
      ctx->pushData(kernel);
      ctx->pushData(stride);
      ctx->pushData(padding);
      ctx->pushData(maxIndices);
    }

    // TODO fuse
    return output;
  }

  static void backward(AutogradContext* ctx, const Tensor& grad) {
    auto& input = ctx->savedInputs[0];

    auto shape = input.shape();
    ASSERT(shape.size() == 3 || shape.size() == 4);

    auto kernel = ctx->popData().toDim2D();
    auto stride = ctx->popData().toDim2D();
    auto padding = ctx->popData().toDim2D();

    int64_t batch = (shape.size() == 4) ? shape[0] : 1;
    int64_t channels = (shape.size() == 4) ? shape[1] : shape[0];
    int64_t height = (shape.size() == 4) ? shape[2] : shape[1];
    int64_t width = (shape.size() == 4) ? shape[3] : shape[2];

    auto outH = (height - kernel.h + 2 * padding.h) / stride.h + 1;
    auto outW = (width - kernel.w + 2 * padding.w) / stride.w + 1;

    if (input.requiresGrad()) {
      auto maxIndices = ctx->popData().toTensor();

      auto gradCol = Tensor::zeros({grad.numel(), kernel.h * kernel.w}, input.options().noGrad());
      auto idx = Tensor::arange<int64_t>(0, grad.numel(), 1, input.options().noGrad());
      op::indexPutAdvance(gradCol, ArrayView<Tensor>{idx, maxIndices}, grad);
      gradCol.reshape_({batch * outH * outW, channels * kernel.h * kernel.w});
      auto inputGrad = op::col2im(gradCol, shape, kernel, stride, padding);
      // TODO fuse
      input.addGrad(std::move(inputGrad));
    }
  }
};

class FuncConv2D : public Function<FuncConv2D> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& input, const Tensor& weight, const Tensor& bias,
                        Dim2D stride, Dim2D padding) {
    ASSERT(input.dim() == 4);
    ASSERT(weight.dim() == 4);
    ASSERT(input.shape()[1] == weight.shape()[1]);

    int64_t batch = input.shape()[0];
    int64_t outChannels = weight.shape()[0];
    // int64_t inChannels = weight.shape()[1];
    int64_t height = input.shape()[2];
    int64_t width = input.shape()[3];
    Dim2D kernel = {weight.shape()[2], weight.shape()[3]};

    auto outH = (height - kernel.h + 2 * padding.h) / stride.h + 1;
    auto outW = (width - kernel.w + 2 * padding.w) / stride.w + 1;

    auto col = op::im2col(input, kernel, stride, padding);

    auto colW = op::reshape(weight, IntArrayView{outChannels, -1});
    auto ret = op::matmul(col, colW, false, true);
    if (bias.defined()) {
      ASSERT(bias.dim() == 1);
      ASSERT(bias.shape()[0] == outChannels);
      op::addInplace(ret, bias, 1);
    }
    ret.reshape_({batch, outChannels, outH, outW});

    if (ctx) {
      ctx->pushData(stride);
      ctx->pushData(padding);
      ctx->pushData(col);
    }

    return ret;
  }

  static void backward(AutogradContext* ctx, const Tensor& grad) {
    auto& input = ctx->savedInputs[0];
    auto& weight = ctx->savedInputs[1];
    auto& bias = ctx->savedInputs[2];

    int64_t outChannels = weight.shape()[0];
    Dim2D kernel = {weight.shape()[2], weight.shape()[3]};
    auto stride = ctx->popData().toDim2D();
    auto padding = ctx->popData().toDim2D();

    auto gradW = op::reshape(grad, IntArrayView{-1, outChannels});
    auto colW = op::reshape(weight, IntArrayView{outChannels, -1});

    if (input.requiresGrad()) {
      auto gradCol = op::matmul(gradW, colW, false, false);
      auto inputGrad = op::col2im(gradCol, input.shape(), kernel, stride, padding);
      input.addGrad(std::move(inputGrad));
    }
    if (weight.requiresGrad()) {
      auto col = ctx->popData().toTensor();
      auto gradColW = op::matmul(col, gradW, true, false);
      auto weightGrad = op::reshape(gradColW.permute(), weight.shape());
      weight.addGrad(std::move(weightGrad));
    }
    if (bias.defined() && bias.requiresGrad()) {
      auto biasGrad = op::sumOnDim(gradW, 0, false);
      bias.addGrad(std::move(biasGrad));
    }
  }
};

class FuncEmbedding : public Function<FuncEmbedding> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& input, const Tensor& weight) {
    ASSERT(input.dtype() == DType::Int64);
    return op::indexAdvance(weight, ArrayView<Tensor>{input});
  }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncLayerNorm : public Function<FuncLayerNorm> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& input, IntArrayView normalizedShape, const Tensor& weight,
                        const Tensor& bias, float eps) {
    return op::layerNorm(input, normalizedShape, weight, bias, eps);
  }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncRMSNorm : public Function<FuncRMSNorm> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& input, IntArrayView normalizedShape, const Tensor& weight,
                        float eps) {
    return op::rmsNorm(input, normalizedShape, weight, eps);
  }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncSDPAttention : public Function<FuncSDPAttention> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& query, const Tensor& key, const Tensor& value,
                        bool isCausal, const Tensor& attnMask, float dropoutP, std::optional<float> scale) {
    auto L = query.size(-2);
    auto S = key.size(-2);
    float scaleFactor = scale.has_value() ? scale.value() : (1.f / std::sqrt(static_cast<float>(query.size(-1))));

    auto attnWeight = op::matmul(query, op::transpose(key, -2, -1), false, false);
    op::mulInplace(attnWeight, Tensor::scalar(scaleFactor, attnWeight.options()));

    Tensor attnBias;
    if (isCausal) {
      ASSERT(!attnMask.defined());
      attnBias = Tensor::zeros({L, S}, query.options().noGrad());
      auto tempMask = Tensor::ones(attnBias.shape(), query.options().noGrad().dtype(DType::Bool));
      tempMask = op::tril(tempMask, 0);
      op::fillMaskedInplace(attnBias, op::logicNot(tempMask), -std::numeric_limits<float>::infinity());
    }

    if (attnMask.defined()) {
      attnBias = Tensor::zeros(attnMask.shape(), query.options().noGrad());
      if (attnMask.dtype() == DType::Bool) {
        op::fillMaskedInplace(attnBias, op::logicNot(attnMask), -std::numeric_limits<float>::infinity());
      } else {
        op::addInplace(attnBias, attnMask, 1);
      }
    }

    if (attnBias.defined()) {
      op::addInplace(attnWeight, attnBias, 1);
    }
    attnWeight = op::softmax(attnWeight, -1);
    if (dropoutP > 0.f) {
      attnWeight = op::dropout(attnWeight, dropoutP);
    }
    return op::matmul(attnWeight, value, false, false);
    // TODO fuse
  }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

class FuncRoPE : public Function<FuncRoPE> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& input, const Tensor& positions, const Tensor& rope) {
    return op::ropeApply(input, positions, rope);
  }
  static void backward(AutogradContext* ctx, const Tensor& grad) { NOT_IMPLEMENTED(); }
};

inline Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias = {}) {
  return FuncLinear::apply(input, weight, bias);
}

inline Tensor dropout(const Tensor& input, float p = 0.5f, bool training = true) {
  return FuncDropout::apply(input, p, training);
}

inline Tensor maxPool2d(const Tensor& input, Dim2D kernel, Dim2D stride, Dim2D padding = 0) {
  return FuncMaxPool::apply(input, kernel, stride, padding);
}

inline Tensor conv2d(const Tensor& input, const Tensor& weight, const Tensor& bias = {}, Dim2D stride = 1,
                     Dim2D padding = 0) {
  return FuncConv2D::apply(input, weight, bias, stride, padding);
}

inline Tensor embedding(const Tensor& input, const Tensor& weight) { return FuncEmbedding::apply(input, weight); }

inline Tensor layerNorm(const Tensor& input, IntArrayView normalizedShape, const Tensor& weight, const Tensor& bias,
                        float eps = 1e-5f) {
  return FuncLayerNorm::apply(input, normalizedShape, weight, bias, eps);
}

inline Tensor rmsNorm(const Tensor& input, IntArrayView normalizedShape, const Tensor& weight, float eps = 1e-8f) {
  return FuncRMSNorm::apply(input, normalizedShape, weight, eps);
}

inline Tensor sdpAttention(const Tensor& query, const Tensor& key, const Tensor& value, bool isCausal = false,
                           const Tensor& attnMask = {}, float dropoutP = 0.f,
                           std::optional<float> scale = std::nullopt) {
  return FuncSDPAttention::apply(query, key, value, isCausal, attnMask, dropoutP, scale);
}
inline Tensor ropeApply(const Tensor& input, const Tensor& positions, const Tensor& rope) {
  return FuncRoPE::apply(input, positions, rope);
}

}  // namespace tinytorch::function
