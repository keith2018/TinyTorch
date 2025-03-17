/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Function.h"

#include <cassert>
#include <set>
#include <unordered_map>

#include "Tensor.h"

namespace TinyTorch {

#define FUNC_ENUM_TO_STRING(value) {value, #value}

std::unordered_map<FunctionType, std::string> Function::funcTypeToString_ = {
    FUNC_ENUM_TO_STRING(Function_Leaf),
    FUNC_ENUM_TO_STRING(Function_Leaf),
    FUNC_ENUM_TO_STRING(Function_Add),
    FUNC_ENUM_TO_STRING(Function_Sub),
    FUNC_ENUM_TO_STRING(Function_Mul),
    FUNC_ENUM_TO_STRING(Function_Div),
    FUNC_ENUM_TO_STRING(Function_Sin),
    FUNC_ENUM_TO_STRING(Function_Cos),
    FUNC_ENUM_TO_STRING(Function_Pow),
    FUNC_ENUM_TO_STRING(Function_PowScalar),
    FUNC_ENUM_TO_STRING(Function_Sum),
    FUNC_ENUM_TO_STRING(Function_Relu),
    FUNC_ENUM_TO_STRING(Function_Flatten),
    FUNC_ENUM_TO_STRING(Function_UnFlatten),
    FUNC_ENUM_TO_STRING(Function_Squeeze),
    FUNC_ENUM_TO_STRING(Function_Unsqueeze),
    FUNC_ENUM_TO_STRING(Function_Reshape),
    FUNC_ENUM_TO_STRING(Function_Linear),
    FUNC_ENUM_TO_STRING(Function_Dropout),
    FUNC_ENUM_TO_STRING(Function_Softmax),
    FUNC_ENUM_TO_STRING(Function_LogSoftmax),
    FUNC_ENUM_TO_STRING(Function_MaxPool2D),
    FUNC_ENUM_TO_STRING(Function_Conv2D),
    FUNC_ENUM_TO_STRING(Function_BatchNorm),
    FUNC_ENUM_TO_STRING(Function_MSELoss),
    FUNC_ENUM_TO_STRING(Function_NLLLoss),
};

Tensor Function::add(const Tensor& a, const Tensor& b) {
  return std::make_shared<FuncAdd>()->callForward({&a, &b});
}

Tensor Function::sub(const Tensor& a, const Tensor& b) {
  return std::make_shared<FuncSub>()->callForward({&a, &b});
}

Tensor Function::mul(const Tensor& a, const Tensor& b) {
  return std::make_shared<FuncMul>()->callForward({&a, &b});
}

Tensor Function::div(const Tensor& a, const Tensor& b) {
  return std::make_shared<FuncDiv>()->callForward({&a, &b});
}

Tensor Function::sin(const Tensor& a) {
  return std::make_shared<FuncSin>()->callForward({&a});
}

Tensor Function::cos(const Tensor& a) {
  return std::make_shared<FuncCos>()->callForward({&a});
}

Tensor Function::pow(const Tensor& a, const float& b) {
  return std::make_shared<FuncPowScalar>(b)->callForward({&a});
}

Tensor Function::pow(const Tensor& a, const Tensor& b) {
  return std::make_shared<FuncPow>()->callForward({&a, &b});
}

Tensor Function::sum(const Tensor& a) {
  return std::make_shared<FuncSum>()->callForward({&a});
}

Tensor Function::relu(const Tensor& input) {
  return std::make_shared<FuncRelu>()->callForward({&input});
}

Tensor Function::flatten(const Tensor& input, int32_t startDim,
                         int32_t endDim) {
  return std::make_shared<FuncFlatten>(startDim, endDim)->callForward({&input});
}

Tensor Function::unflatten(const Tensor& input, int32_t dim,
                           const std::vector<int32_t>& sizes) {
  return std::make_shared<FuncUnFlatten>(dim, sizes)->callForward({&input});
}

Tensor Function::squeeze(const Tensor& input, int32_t dim) {
  return std::make_shared<FuncSqueeze>(dim)->callForward({&input});
}

Tensor Function::unsqueeze(const Tensor& input, int32_t dim) {
  return std::make_shared<FuncUnsqueeze>(dim)->callForward({&input});
}

Tensor Function::reshape(const Tensor& input, const Shape& shape) {
  return std::make_shared<FuncReshape>(shape)->callForward({&input});
}

Tensor Function::linear(const Tensor& input, const Tensor& weight,
                        const Tensor& bias) {
  return std::make_shared<FuncLinear>()->callForward({&input, &weight, &bias});
}

Tensor Function::dropout(const Tensor& input, float p, bool training) {
  return std::make_shared<FuncDropout>(p, training)->callForward({&input});
}

Tensor Function::softmax(const Tensor& input, int32_t dim) {
  return std::make_shared<FuncSoftmax>(dim)->callForward({&input});
}

Tensor Function::logSoftmax(const Tensor& input, int32_t dim) {
  return std::make_shared<FuncLogSoftmax>(dim)->callForward({&input});
}

Tensor Function::maxPool2d(const Tensor& input, Size2D kernelSize,
                           std::optional<Size2D> stride, Size2D padding) {
  return std::make_shared<FuncMaxPool2D>(
             kernelSize, stride.has_value() ? stride.value() : kernelSize,
             padding)
      ->callForward({&input});
}

Tensor Function::conv2d(const Tensor& input, const Tensor& weight,
                        const Tensor& bias, Size2D stride, Size2D padding) {
  return std::make_shared<FuncConv2D>(stride, padding)
      ->callForward({&input, &weight, &bias});
}

Tensor Function::batchNorm(const Tensor& input, Tensor& runningMean,
                           Tensor& runningVar, const Tensor& weight,
                           const Tensor& bias, bool training, float momentum,
                           float eps) {
  return std::make_shared<FuncBatchNorm>(runningMean, runningVar, momentum, eps,
                                         training)
      ->callForward({&input, &weight, &bias});
}

Tensor Function::mseLoss(const Tensor& input, const Tensor& target,
                         LossReduction reduction) {
  return std::make_shared<FuncMSELoss>(reduction)->callForward(
      {&input, &target});
}

Tensor Function::nllloss(const Tensor& input, const Tensor& target,
                         LossReduction reduction) {
  return std::make_shared<FuncNLLLoss>(reduction)->callForward(
      {&input, &target});
}

Tensor Function::callForward(const std::vector<const Tensor*>& inputs) {
  auto output = forward(inputs);

  auto requiresGrad = false;
  for (const auto input : inputs) {
    if (input->isRequiresGrad()) {
      requiresGrad = true;
      break;
    }
  }

  std::shared_ptr<Function> gradFunc = nullptr;
  if (NoGradScope::isGradEnabled() && requiresGrad) {
    for (const auto input : inputs) {
      if (input->isRequiresGrad()) {
        nextFuncs_.push_back(input->getGradFunc());
      }
    }
    gradFunc = shared_from_this();
  }

  return Tensor(std::move(output), requiresGrad, gradFunc);
}

std::vector<TensorImpl> Function::callBackward(const TensorImpl& grad) {
  auto ret = backward(grad);
  savedTensors_.clear();
  return ret;
}

TensorImpl FuncLeaf::forward(const std::vector<const Tensor*>& inputs) {
  return {};
}

std::vector<TensorImpl> FuncLeaf::backward(const TensorImpl& grad) {
  auto owner = owner_.lock();
  if (owner == nullptr) {
    return {grad};
  }

  // for broadcast
  if (owner->grad_.data_->shape() != grad.shape()) {
    TensorImpl retGrad =
        TensorImpl::sum(grad, 0, owner->grad_.data_->dim() != 0);
    assert(retGrad.shape() == owner->grad_.data_->shape());
    *owner->grad_.data_ = std::move(retGrad);
    return {*owner->grad_.data_};
  }

  assert(grad.shape() == owner->grad_.data_->shape());
  *owner->grad_.data_ = grad;
  return {grad};
}

TensorImpl FuncAdd::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  return inputs[0]->data() + inputs[1]->data();
}

std::vector<TensorImpl> FuncAdd::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    ret.push_back(grad);
  }
  if (savedTensors[1].isRequiresGrad()) {
    ret.push_back(grad);
  }
  return ret;
}

TensorImpl FuncSub::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  return inputs[0]->data() - inputs[1]->data();
}

std::vector<TensorImpl> FuncSub::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    ret.push_back(grad);
  }
  if (savedTensors[1].isRequiresGrad()) {
    ret.push_back(-1 * grad);
  }
  return ret;
}

TensorImpl FuncMul::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  return inputs[0]->data() * inputs[1]->data();
}

std::vector<TensorImpl> FuncMul::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    ret.push_back(savedTensors[1].data() * grad);
  }
  if (savedTensors[1].isRequiresGrad()) {
    ret.push_back(savedTensors[0].data() * grad);
  }
  return ret;
}

TensorImpl FuncDiv::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  return inputs[0]->data() / inputs[1]->data();
}

std::vector<TensorImpl> FuncDiv::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    ret.push_back(grad / savedTensors[1].data());
  }
  if (savedTensors[1].isRequiresGrad()) {
    ret.push_back(-1.f * grad * savedTensors[0].data() /
                  (savedTensors[1].data() * savedTensors[1].data()));
  }
  return ret;
}

TensorImpl FuncSin::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  return TensorImpl::sin(inputs[0]->data());
}

std::vector<TensorImpl> FuncSin::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    ret.push_back(TensorImpl::cos(savedTensors[0].data()) * grad);
  }
  return ret;
}

TensorImpl FuncCos::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  return TensorImpl::cos(inputs[0]->data());
}

std::vector<TensorImpl> FuncCos::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    ret.push_back(-1.f * TensorImpl::sin(savedTensors[0].data()) * grad);
  }
  return ret;
}

TensorImpl FuncPow::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  return TensorImpl::pow(inputs[0]->data(), inputs[1]->data());
}

std::vector<TensorImpl> FuncPow::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    ret.push_back(
        savedTensors[1].data() *
        TensorImpl::pow(savedTensors[0].data(), savedTensors[1].data() - 1) *
        grad);
  }
  if (savedTensors[1].isRequiresGrad()) {
    ret.push_back(savedTensors[0].data() * grad);
  }
  return ret;
}

TensorImpl FuncPowScalar::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  return TensorImpl::pow(inputs[0]->data(), exp_);
}

std::vector<TensorImpl> FuncPowScalar::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    ret.push_back(exp_ * TensorImpl::pow(savedTensors[0].data(), exp_ - 1.f) *
                  grad);
  }
  return ret;
}

TensorImpl FuncSum::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  return TensorImpl::sum(inputs[0]->data());
}

std::vector<TensorImpl> FuncSum::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  auto& input = savedTensors[0];
  if (input.isRequiresGrad()) {
    ret.push_back(grad * TensorImpl::onesLike(input.data(), input.device()));
  }
  return ret;
}

TensorImpl FuncRelu::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  return TensorImpl::clampMin(inputs[0]->data(), 0);
}

std::vector<TensorImpl> FuncRelu::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    ret.push_back(grad * (savedTensors[0].data() > 0));
  }
  return ret;
}

TensorImpl FuncFlatten::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  return TensorImpl::flatten(inputs[0]->data(), startDim_, endDim_);
}

std::vector<TensorImpl> FuncFlatten::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    assert(grad.numel() == savedTensors[0].numel());
    ret.push_back(TensorImpl::reshape(grad, savedTensors[0].shape()));
  }
  return ret;
}

TensorImpl FuncUnFlatten::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  return TensorImpl::unflatten(inputs[0]->data(), dim_, sizes_);
}

std::vector<TensorImpl> FuncUnFlatten::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    assert(grad.numel() == savedTensors[0].numel());
    ret.push_back(TensorImpl::reshape(grad, savedTensors[0].shape()));
  }
  return ret;
}

TensorImpl FuncSqueeze::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  return TensorImpl::squeeze(inputs[0]->data(), dim_);
}

std::vector<TensorImpl> FuncSqueeze::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    assert(grad.numel() == savedTensors[0].numel());
    ret.push_back(TensorImpl::reshape(grad, savedTensors[0].shape()));
  }
  return ret;
}

TensorImpl FuncUnsqueeze::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  return TensorImpl::unsqueeze(inputs[0]->data(), dim_);
}

std::vector<TensorImpl> FuncUnsqueeze::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    assert(grad.numel() == savedTensors[0].numel());
    ret.push_back(TensorImpl::reshape(grad, savedTensors[0].shape()));
  }
  return ret;
}

TensorImpl FuncReshape::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  return TensorImpl::reshape(inputs[0]->data(), shape_);
}

std::vector<TensorImpl> FuncReshape::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    assert(grad.numel() == savedTensors[0].numel());
    ret.push_back(TensorImpl::reshape(grad, savedTensors[0].shape()));
  }
  return ret;
}

TensorImpl FuncLinear::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  auto output = TensorImpl::matmulTrans(inputs[0]->data(), inputs[1]->data(),
                                        false, true);
  if (!inputs[2]->empty()) {
    output += inputs[2]->data();
  }
  return output;
}

std::vector<TensorImpl> FuncLinear::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    ret.push_back(TensorImpl::matmul(grad, savedTensors[1].data()));
  }
  if (savedTensors[1].isRequiresGrad()) {
    ret.push_back(
        TensorImpl::matmulTrans(grad, savedTensors[0].data(), true, false));
  }
  if (!savedTensors[2].empty() && savedTensors[2].isRequiresGrad()) {
    ret.push_back(TensorImpl::sum(grad, 0));
  }
  return ret;
}

TensorImpl FuncDropout::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  if (training_) {
    mask_ = TensorImpl::bernoulli(inputs[0]->shape(), 1.f - p_,
                                  inputs[0]->device());
    return mask_ * inputs[0]->data() / (1.f - p_);
  } else {
    return inputs[0]->data();
  }
}

std::vector<TensorImpl> FuncDropout::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    ret.push_back(mask_ * grad / (1.f - p_));
  }
  return ret;
}

TensorImpl FuncSoftmax::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  auto max = TensorImpl::max(inputs[0]->data(), dim_, true).first;
  auto shifted = inputs[0]->data() - max;
  auto exp = TensorImpl::exp(shifted);
  auto sumExp = TensorImpl::sum(exp, dim_, true);
  forwardOutput_ = exp / sumExp;
  return forwardOutput_;
}

std::vector<TensorImpl> FuncSoftmax::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    auto retGrad = forwardOutput_ *
                   (grad - TensorImpl::sum(grad * forwardOutput_, dim_, true));
    ret.push_back(std::move(retGrad));
  }
  return ret;
}

TensorImpl FuncLogSoftmax::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  auto max = TensorImpl::max(inputs[0]->data(), dim_, true).first;
  auto logSumExp = TensorImpl::log(
      TensorImpl::sum(TensorImpl::exp(inputs[0]->data() - max), dim_, true));
  forwardOutput_ = inputs[0]->data() - max - logSumExp;
  return forwardOutput_;
}

std::vector<TensorImpl> FuncLogSoftmax::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    auto softmax = TensorImpl::exp(forwardOutput_);
    auto retGrad = grad - TensorImpl::sum(grad, dim_, true) * softmax;
    ret.push_back(std::move(retGrad));
  }
  return ret;
}

TensorImpl FuncMaxPool2D::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  auto& shape = inputs[0]->shape();
  assert(shape.size() == 3 || shape.size() == 4);
  int32_t batch = (shape.size() == 4) ? shape[0] : 1;
  int32_t channels = (shape.size() == 4) ? shape[1] : shape[0];
  int32_t height = (shape.size() == 4) ? shape[2] : shape[1];
  int32_t width = (shape.size() == 4) ? shape[3] : shape[2];

  auto outH = (height - kernelSize_.h + 2 * padding_.h) / stride_.h + 1;
  auto outW = (width - kernelSize_.w + 2 * padding_.w) / stride_.w + 1;

  auto col = inputs[0]->data().im2col(kernelSize_, stride_, padding_);
  col.reshape_({-1, kernelSize_.h * kernelSize_.w});

  auto maxRet = TensorImpl::max(col, 1);
  maxIndices_ = maxRet.second;
  auto ret = maxRet.first;
  ret.reshape_({batch, channels, outH, outW});
  return ret;
}

std::vector<TensorImpl> FuncMaxPool2D::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  auto& shape = savedTensors[0].shape();
  assert(shape.size() == 3 || shape.size() == 4);
  int32_t batch = (shape.size() == 4) ? shape[0] : 1;
  int32_t channels = (shape.size() == 4) ? shape[1] : shape[0];
  int32_t height = (shape.size() == 4) ? shape[2] : shape[1];
  int32_t width = (shape.size() == 4) ? shape[3] : shape[2];

  auto outH = (height - kernelSize_.h + 2 * padding_.h) / stride_.h + 1;
  auto outW = (width - kernelSize_.w + 2 * padding_.w) / stride_.w + 1;

  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    auto gradCol = TensorImpl::zeros(
        {grad.numel(), kernelSize_.h * kernelSize_.w}, grad.device());
    auto gradIdx =
        TensorImpl::arange(0, (float)grad.numel(), 1.f, grad.device());
    gradCol.indexPut_({gradIdx, maxIndices_}, grad);
    gradCol.reshape_(
        {batch * outH * outW, channels * kernelSize_.h * kernelSize_.w});
    auto retGrad = gradCol.col2im(shape, kernelSize_, stride_, padding_);
    ret.push_back(std::move(retGrad));
  }
  return ret;
}

TensorImpl FuncConv2D::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  auto& input = inputs[0]->data();
  auto& weight = inputs[1]->data();
  auto& bias = inputs[2]->data();
  assert(input.dim() == 4);
  assert(weight.dim() == 4);
  assert(input.shape()[1] == weight.shape()[1]);
  int32_t batch = input.shape()[0];
  int32_t outChannels = weight.shape()[0];
  // int32_t inChannels = weight.shape()[1];
  int32_t height = input.shape()[2];
  int32_t width = input.shape()[3];
  Size2D kernelSize = {weight.shape()[2], weight.shape()[3]};

  auto outH = (height - kernelSize.h + 2 * padding_.h) / stride_.h + 1;
  auto outW = (width - kernelSize.w + 2 * padding_.w) / stride_.w + 1;

  col_ = input.im2col(kernelSize, stride_, padding_);
  auto colW = TensorImpl::reshape(weight, {outChannels, -1});
  auto ret = TensorImpl::matmulTrans(col_, colW, false, true);
  if (!bias.empty()) {
    assert(bias.dim() == 1);
    assert(bias.shape()[0] == outChannels);
    ret += bias;
  }
  ret.reshape_({batch, outChannels, outH, outW});
  return ret;
}

std::vector<TensorImpl> FuncConv2D::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  auto& input = savedTensors[0].data();
  auto& weight = savedTensors[1].data();
  // auto& bias = savedTensors[2].data();

  int32_t outChannels = weight.shape()[0];
  Size2D kernelSize = {weight.shape()[2], weight.shape()[3]};

  auto gradW = TensorImpl::reshape(grad, {-1, outChannels});
  auto colW = TensorImpl::reshape(weight, {outChannels, -1});

  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    auto gradCol = TensorImpl::matmul(gradW, colW);
    auto dx = gradCol.col2im(input.shape(), kernelSize, stride_, padding_);
    ret.push_back(std::move(dx));
  }
  if (savedTensors[1].isRequiresGrad()) {
    auto gradColW = TensorImpl::matmulTrans(col_, gradW, true, false);
    auto dw = TensorImpl::reshape(gradColW.permute(), weight.shape());
    ret.push_back(std::move(dw));
  }
  if (!savedTensors[2].empty() && savedTensors[2].isRequiresGrad()) {
    auto db = TensorImpl::sum(gradW, 0);
    ret.push_back(std::move(db));
  }
  return ret;
}

TensorImpl FuncBatchNorm::forward(const std::vector<const Tensor*>& inputs) {
  auto& input = inputs[0]->data();
  auto& weight = inputs[1]->data();
  auto& bias = inputs[2]->data();

  auto& shape = input.shape();
  assert(shape.size() == 3 || shape.size() == 4);

  if (shape.size() == 3) {
    dims_ = {0, 2};
    viewShape_ = {1, shape[1], 1};
  } else {
    dims_ = {0, 2, 3};
    viewShape_ = {1, shape[1], 1, 1};
  }

  Tensor mean;
  Tensor var;
  if (training_) {
    auto varMean = input.varMean(dims_, false, true);
    mean.data() = varMean.second;
    var.data() = varMean.first;
    auto varUnbiased = input.var(dims_, true, true);

    if (!runningMean_.empty() && !runningVar_.empty()) {
      runningMean_.data() *= 1.f - momentum_;
      runningMean_.data() += TensorImpl::squeeze(mean.data()) * momentum_;
      runningVar_.data() *= 1.f - momentum_;
      runningVar_.data() += TensorImpl::squeeze(varUnbiased) * momentum_;
    }
  } else {
    if (!runningMean_.empty() && !runningVar_.empty()) {
      mean = runningMean_;
      var = runningVar_;
    } else {
      auto varMean = input.varMean(dims_, true, true);
      mean.data() = varMean.second;
      var.data() = varMean.first;
    }
  }

  auto inputCentered = Tensor(input - mean.data());
  auto std = Tensor((var.data() + eps_).sqrt());
  auto inputNorm = Tensor(inputCentered / std);

  saveForBackward(inputs);
  saveForBackward({&inputNorm, &inputCentered, &std});

  if (!weight.empty()) {
    inputNorm.data() *= weight.view(viewShape_);
  }
  if (!bias.empty()) {
    inputNorm.data() += bias.view(viewShape_);
  }
  return inputNorm.data();
}

std::vector<TensorImpl> FuncBatchNorm::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  auto& input = savedTensors[0].data();
  auto& weight = savedTensors[1].data();
  // auto& bias = savedTensors[2].data();
  auto& inputNorm = savedTensors[3].data();
  auto& inputCentered = savedTensors[4].data();
  auto& std = savedTensors[5].data();

  std::vector<TensorImpl> ret;
  // grad of input
  if (savedTensors[0].isRequiresGrad()) {
    auto dInputNorm = grad;
    if (!weight.empty()) {
      dInputNorm *= weight.view(viewShape_);
    }
    int32_t N = 1;
    for (int dim : dims_) {
      N *= input.shape()[dim];
    }
    auto dVar =
        (dInputNorm * inputCentered * -0.5f * std.pow(-3.f)).sum(dims_, true);
    auto dMean = (dInputNorm * -1.f / std).sum(dims_, true) +
                 dVar * (inputCentered * -2.f / (float)N).sum(dims_, true);
    auto dInput = dInputNorm / std + dVar * 2.f * inputCentered / (float)N +
                  dMean / (float)N;
    ret.push_back(std::move(dInput));
  }
  // grad of weight
  if (savedTensors[1].isRequiresGrad()) {
    auto dWeight = (grad * inputNorm).sum(dims_);
    ret.push_back(std::move(dWeight));
  }

  // grad of bias
  if (savedTensors[2].isRequiresGrad()) {
    auto dBias = grad.sum(dims_);
    ret.push_back(std::move(dBias));
  }
  return ret;
}

TensorImpl FuncMSELoss::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  auto ret = TensorImpl::pow(inputs[0]->data() - inputs[1]->data(), 2);
  switch (reduction_) {
    case MEAN:
      return ret.mean();
    case SUM:
      return ret.sum();
    default:
      break;
  }
  return ret;
}

std::vector<TensorImpl> FuncMSELoss::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  auto retGrad = grad * 2 * (savedTensors[0].data() - savedTensors[1].data());
  switch (reduction_) {
    case MEAN:
      retGrad /= (float)savedTensors[0].numel();
    default:
      break;
  }
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    ret.push_back(retGrad);
  }
  if (savedTensors[1].isRequiresGrad()) {
    ret.push_back(-1 * retGrad);
  }
  return ret;
}

TensorImpl FuncNLLLoss::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  assert(inputs[1]->dim() == 1);
  auto batchSize = (int32_t)inputs[0]->shape()[0];
  auto idx = TensorImpl::arange(0, (float)batchSize, 1.f, inputs[0]->device());
  auto ret = -1 * inputs[0]->data().index({idx, inputs[1]->data()});
  switch (reduction_) {
    case MEAN:
      return ret.mean();
    case SUM:
      return ret.sum();
    default:
      break;
  }
  return ret;
}

std::vector<TensorImpl> FuncNLLLoss::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  auto batchSize = (int32_t)savedTensors[0].shape()[0];
  auto retGrad = TensorImpl::zeros(savedTensors[0].shape(), grad.device());
  auto idx =
      TensorImpl::arange(0, (float)batchSize, 1.f, savedTensors[0].device());
  retGrad.indexPut_({idx, savedTensors[1].data()}, -1.f);
  switch (reduction_) {
    case MEAN:
      retGrad /= (float)batchSize;
    default:
      break;
  }
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    ret.push_back(std::move(retGrad));
  }
  return ret;
}

}  // namespace TinyTorch
