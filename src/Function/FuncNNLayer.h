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
    auto output = op::matmulTrans(input, weight, false, true);
    if (bias.defined()) {
      op::addInplace(output, bias, 1);
    }
    // TODO fuse
    return output;
  }

  static TensorList backward(AutogradContext* ctx, const Tensor& grad) {
    auto& input = ctx->savedInputs[0];
    auto& weight = ctx->savedInputs[1];
    auto& bias = ctx->savedInputs[2];

    TensorList ret;
    if (input.requiresGrad()) {
      ret.push_back(std::move(op::matmul(grad, weight)));
    }
    if (weight.requiresGrad()) {
      ret.push_back(std::move(op::matmulTrans(grad, input, true, false)));
    }
    if (bias.defined() && bias.requiresGrad()) {
      ret.push_back(std::move(op::sumOnDim(grad, 0, false)));
    }
    return ret;
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
    auto output = op::dropout(input, mask, 1.f - p);
    if (ctx) {
      ctx->pushData(p);
      ctx->pushData(mask);
    }
    // TODO fuse
    return output;
  }

  static TensorList backward(AutogradContext* ctx, const Tensor& grad) {
    auto& input = ctx->savedInputs[0];

    TensorList ret;
    if (input.requiresGrad()) {
      auto training = ctx->popData().toBool();
      if (training) {
        auto p = ctx->popData().toFloat();
        auto mask = ctx->popData().toTensor();
        ret.push_back(std::move(op::dropout(grad, mask, 1.f - p)));
      } else {
        ret.push_back(grad);
      }
    }
    return ret;
  }
};

class FuncMaxPool : public Function<FuncMaxPool> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& input, Dim2D kernelSize, Dim2D stride, Dim2D padding) {
    auto shape = input.shape();
    ASSERT(shape.size() == 3 || shape.size() == 4);

    int64_t batch = (shape.size() == 4) ? shape[0] : 1;
    int64_t channels = (shape.size() == 4) ? shape[1] : shape[0];
    int64_t height = (shape.size() == 4) ? shape[2] : shape[1];
    int64_t width = (shape.size() == 4) ? shape[3] : shape[2];

    auto outH = (height - kernelSize.h + 2 * padding.h) / stride.h + 1;
    auto outW = (width - kernelSize.w + 2 * padding.w) / stride.w + 1;

    auto col = op::im2col(input, kernelSize, stride, padding);
    col.reshape({-1, kernelSize.h * kernelSize.w});

    auto maxRet = op::maxOnDim(col, 1, false);
    auto maxIndices = maxRet.second;

    auto output = maxRet.first;
    output.reshape({batch, channels, outH, outW});

    if (ctx) {
      ctx->pushData(kernelSize);
      ctx->pushData(stride);
      ctx->pushData(padding);
      ctx->pushData(maxIndices);
    }

    // TODO fuse
    return output;
  }

  static TensorList backward(AutogradContext* ctx, const Tensor& grad) {
    auto& input = ctx->savedInputs[0];

    auto shape = input.shape();
    ASSERT(shape.size() == 3 || shape.size() == 4);

    auto kernelSize = ctx->popData().toDim2D();
    auto stride = ctx->popData().toDim2D();
    auto padding = ctx->popData().toDim2D();

    int64_t batch = (shape.size() == 4) ? shape[0] : 1;
    int64_t channels = (shape.size() == 4) ? shape[1] : shape[0];
    int64_t height = (shape.size() == 4) ? shape[2] : shape[1];
    int64_t width = (shape.size() == 4) ? shape[3] : shape[2];

    auto outH = (height - kernelSize.h + 2 * padding.h) / stride.h + 1;
    auto outW = (width - kernelSize.w + 2 * padding.w) / stride.w + 1;

    TensorList ret;
    if (input.requiresGrad()) {
      auto maxIndices = ctx->popData().toTensor();

      auto gradCol = Tensor::zeros({grad.numel(), kernelSize.h * kernelSize.w}, input.options().noGrad());
      auto idx = Tensor::arange<int64_t>(0, grad.numel(), 1, input.options().noGrad());
      op::indexPutAdvance(gradCol, ArrayView<Tensor>{idx, maxIndices}, grad);
      gradCol.reshape({batch * outH * outW, channels * kernelSize.h * kernelSize.w});
      auto inputGrad = op::col2im(gradCol, shape, kernelSize, stride, padding);
      // TODO fuse
      ret.push_back(std::move(inputGrad));
    }
    return ret;
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
    Dim2D kernelSize = {weight.shape()[2], weight.shape()[3]};

    auto outH = (height - kernelSize.h + 2 * padding.h) / stride.h + 1;
    auto outW = (width - kernelSize.w + 2 * padding.w) / stride.w + 1;

    auto col = op::im2col(input, kernelSize, stride, padding);

    auto colW = op::reshape(weight, IntArrayView{outChannels, -1});
    auto ret = op::matmulTrans(col, colW, false, true);
    if (bias.defined()) {
      ASSERT(bias.dim() == 1);
      ASSERT(bias.shape()[0] == outChannels);
      op::addInplace(ret, bias, 1);
    }
    ret.reshape({batch, outChannels, outH, outW});

    if (ctx) {
      ctx->pushData(stride);
      ctx->pushData(padding);
      ctx->pushData(col);
    }

    return ret;
  }

  static TensorList backward(AutogradContext* ctx, const Tensor& grad) {
    auto& input = ctx->savedInputs[0];
    auto& weight = ctx->savedInputs[1];
    auto& bias = ctx->savedInputs[2];

    int64_t outChannels = weight.shape()[0];
    Dim2D kernelSize = {weight.shape()[2], weight.shape()[3]};
    auto stride = ctx->popData().toDim2D();
    auto padding = ctx->popData().toDim2D();

    auto gradW = op::reshape(grad, IntArrayView{-1, outChannels});
    auto colW = op::reshape(weight, IntArrayView{outChannels, -1});

    TensorList ret;
    if (input.requiresGrad()) {
      auto gradCol = op::matmul(gradW, colW);
      auto inputGrad = op::col2im(gradCol, input.shape(), kernelSize, stride, padding);
      ret.push_back(std::move(inputGrad));
    }
    if (weight.requiresGrad()) {
      auto col = ctx->popData().toTensor();
      auto gradColW = op::matmulTrans(col, gradW, true, false);
      auto weightGrad = op::reshape(gradColW.permute(), weight.shape());
      ret.push_back(std::move(weightGrad));
    }
    if (bias.defined() && bias.requiresGrad()) {
      auto biasGrad = op::sumOnDim(gradW, 0, false);
      ret.push_back(std::move(biasGrad));
    }
    return ret;
  }
};

class FuncLayerNorm : public Function<FuncLayerNorm> {
 public:
  static Tensor forward(AutogradContext* ctx, const Tensor& input, const Tensor& weight, const Tensor& bias,
                        float eps) {
    return op::layerNorm(input, weight, bias, eps);
  }

  static TensorList backward(AutogradContext* ctx, const Tensor& grad) {
    auto& input = ctx->savedInputs[0];
    auto& weight = ctx->savedInputs[1];
    auto& bias = ctx->savedInputs[2];

    TensorList ret;
    // TODO
    NOT_IMPLEMENTED();
    return ret;
  }
};

inline Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias = {}) {
  return FuncLinear::apply(input, weight, bias);
}

inline Tensor dropout(const Tensor& input, float p = 0.5f, bool training = true) {
  return FuncDropout::apply(input, p, training);
}

inline Tensor maxPool2d(const Tensor& input, Dim2D kernelSize, Dim2D stride, Dim2D padding = 0) {
  return FuncMaxPool::apply(input, kernelSize, stride, padding);
}

inline Tensor conv2d(const Tensor& input, const Tensor& weight, const Tensor& bias = {}, Dim2D stride = 1,
                     Dim2D padding = 0) {
  return FuncConv2D::apply(input, weight, bias, stride, padding);
}

inline Tensor layerNorm(const Tensor& input, const Tensor& weight, const Tensor& bias, float eps = 1e-5f) {
  return FuncLayerNorm::apply(input, weight, bias, eps);
}

}  // namespace tinytorch::function
