/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "OpNNLayerCuda.cuh"

namespace tinytorch::op {

#define REG_NN_LAYER_CUDA_FLT(NAME, FUNC)                                       \
  REGISTER_OP_IMPL(NAME, CUDA, Float32, &(FUNC<DTypeToType_t<DType::Float32>>)) \
  REGISTER_OP_IMPL(NAME, CUDA, Float16, &(FUNC<DTypeToType_t<DType::Float16>>)) \
  REGISTER_OP_IMPL(NAME, CUDA, BFloat16, &(FUNC<DTypeToType_t<DType::BFloat16>>))

void registerNNLayerCuda() {
  // softmax
  REG_NN_LAYER_CUDA_FLT(softmax, softmaxOpCudaImpl);
  REG_NN_LAYER_CUDA_FLT(softmaxOut, softmaxOpOutCudaImpl);
  REG_NN_LAYER_CUDA_FLT(softmaxBackward, softmaxOpBackwardCudaImpl);

  // logSoftmax
  REG_NN_LAYER_CUDA_FLT(logSoftmax, logSoftmaxOpCudaImpl);
  REG_NN_LAYER_CUDA_FLT(logSoftmaxOut, logSoftmaxOpOutCudaImpl);
  REG_NN_LAYER_CUDA_FLT(logSoftmaxBackward, logSoftmaxOpBackwardCudaImpl);

  // dropout
  REG_NN_LAYER_CUDA_FLT(dropout, dropoutOpCudaImpl);
  REG_NN_LAYER_CUDA_FLT(dropoutMasked, dropoutMaskedOpCudaImpl);

  // layerNorm
  REG_NN_LAYER_CUDA_FLT(layerNorm, layerNormOpCudaImpl);

  // rmsNorm
  REG_NN_LAYER_CUDA_FLT(rmsNorm, rmsNormOpCudaImpl);

  // rope
  REG_NN_LAYER_CUDA_FLT(ropeInit, ropeInitOpCudaImpl);
  REG_NN_LAYER_CUDA_FLT(ropeApply, ropeApplyOpCudaImpl);
}

}  // namespace tinytorch::op
