/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "OpNNLayerCuda.cuh"

namespace tinytorch::op {

#define REG_NN_LAYER_CUDA_F32(NAME, FUNC) REGISTER_OP_IMPL(NAME, CUDA, Float32, &(FUNC<DTypeToType_t<DType::Float32>>))

void registerNNLayerCuda() {
  // softmax
  REG_NN_LAYER_CUDA_F32(softmax, softmaxOpCudaImpl);
  REG_NN_LAYER_CUDA_F32(softmaxOut, softmaxOpOutCudaImpl);
  REG_NN_LAYER_CUDA_F32(softmaxBackward, softmaxOpBackwardCudaImpl);

  // logSoftmax
  REG_NN_LAYER_CUDA_F32(logSoftmax, logSoftmaxOpCudaImpl);
  REG_NN_LAYER_CUDA_F32(logSoftmaxOut, logSoftmaxOpOutCudaImpl);
  REG_NN_LAYER_CUDA_F32(logSoftmaxBackward, logSoftmaxOpBackwardCudaImpl);

  // dropout
  REG_NN_LAYER_CUDA_F32(dropout, dropoutOpCudaImpl);
  REG_NN_LAYER_CUDA_F32(dropoutMasked, dropoutMaskedOpCudaImpl);

  // layerNorm
  REG_NN_LAYER_CUDA_F32(layerNorm, layerNormOpCudaImpl);
}

}  // namespace tinytorch::op
