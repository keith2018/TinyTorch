/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "OpLoss.h"
#include "Operations.h"
#include "Tensor/TensorIterator.h"

namespace tinytorch::op {

template <typename T>
Tensor mseLossOpCpuImpl(const Tensor& input, const Tensor& target, LossReduction reduction = LossReduction::MEAN) {
  TensorIteratorCpu iterator(input, target);
  auto outShape = iterator.setupBroadcast();
  ASSERT(iterator.isBroadcastOk());
  Tensor out(outShape, input.options().noGrad());
  iterator.template forEach<T>(out, [](const T& a, const T& b) -> T { return (a - b) * (a - b); });
  switch (reduction) {
    case LossReduction::MEAN:
      out = op::mean(out);
    case LossReduction::SUM:
      out = op::sum(out);
    default:
      break;
  }
  // TODO fuse
  return out;
}

template <typename T>
Tensor mseLossOpBackwardCpuImpl(const Tensor& grad, const Tensor& input, const Tensor& target,
                                LossReduction reduction = LossReduction::MEAN) {
  TensorIteratorCpu iterator(input, target, grad);
  auto outShape = iterator.setupBroadcast();
  ASSERT(iterator.isBroadcastOk());
  Tensor out(outShape, input.options().noGrad());
  iterator.template forEach<T>(out, [](const T& a, const T& b, const T& g) -> T { return 2 * (a - b) * g; });

  switch (reduction) {
    case LossReduction::MEAN:
      op::divInplace(out, Tensor::scalar(input.numel(), out.options().noGrad()));
      break;
    default:
      break;
  }
  // TODO fuse
  return out;
}

template <typename T>
Tensor nllLossOpCpuImpl(const Tensor& input, const Tensor& target, LossReduction reduction = LossReduction::MEAN) {
  ASSERT(target.dim() == 1);
  ASSERT(target.dtype() == DType::Int64);
  auto batchSize = input.shape(0);
  auto idx = Tensor::arange<int64_t>(0, batchSize, 1, input.options().noGrad());
  Tensor out = op::indexAdvance(input, ArrayView<Tensor>{idx, target});
  op::mulInplace(out, Tensor::scalar(-1, out.options().noGrad()));
  switch (reduction) {
    case LossReduction::MEAN:
      out = op::mean(out);
    case LossReduction::SUM:
      out = op::sum(out);
    default:
      break;
  }
  // TODO fuse
  return out;
}

template <typename T>
Tensor nllLossOpBackwardCpuImpl(const Tensor& grad, const Tensor& input, const Tensor& target,
                                LossReduction reduction = LossReduction::MEAN) {
  ASSERT(target.dim() == 1);
  ASSERT(target.dtype() == DType::Int64);
  auto batchSize = input.shape(0);
  auto idx = Tensor::arange<int64_t>(0, batchSize, 1, input.options().noGrad());
  Tensor out = Tensor::zeros(input.shape(), input.options().noGrad());
  op::indexPutAdvance(out, ArrayView<Tensor>{idx, target}, Tensor::scalar(-1, out.options().noGrad()));

  switch (reduction) {
    case LossReduction::MEAN:
      op::divInplace(out, Tensor::scalar(batchSize, out.options().noGrad()));
      break;
    default:
      break;
  }
  // TODO fuse
  return out;
}

}  // namespace tinytorch::op
