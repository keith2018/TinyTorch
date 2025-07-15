/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "OpNNLayerCuda.cuh"

namespace tinytorch::op {

void registerNNLayerCuda() {
  // softmax
  REGISTER_OP_IMPL_DTYPE_TPL(softmax, CUDA, softmaxOpCudaImpl);
  REGISTER_OP_IMPL_DTYPE_TPL(softmaxOut, CUDA, softmaxOpOutCudaImpl);
  REGISTER_OP_IMPL_DTYPE_TPL(softmaxBackward, CUDA, softmaxOpBackwardCudaImpl);

  // logSoftmax
  REGISTER_OP_IMPL_DTYPE_TPL(logSoftmax, CUDA, logSoftmaxOpCudaImpl);
  REGISTER_OP_IMPL_DTYPE_TPL(logSoftmaxOut, CUDA, logSoftmaxOpOutCudaImpl);
  REGISTER_OP_IMPL_DTYPE_TPL(logSoftmaxBackward, CUDA, logSoftmaxOpBackwardCudaImpl);

  // dropout
  REGISTER_OP_IMPL_DTYPE_TPL(dropout, CUDA, dropoutOpCudaImpl);

  // layerNorm
  REGISTER_OP_IMPL_DTYPE_TPL(layerNorm, CUDA, layerNormOpCudaImpl);
}

}  // namespace tinytorch::op
