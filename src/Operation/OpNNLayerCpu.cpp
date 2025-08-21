/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "OpNNLayerCpu.h"

namespace tinytorch::op {

#define REG_NN_LAYER_CPU_FLT(NAME, FUNC)                                       \
  REGISTER_OP_IMPL(NAME, CPU, Float32, &(FUNC<DTypeToType_t<DType::Float32>>)) \
  REGISTER_OP_IMPL(NAME, CPU, Float16, &(FUNC<DTypeToType_t<DType::Float16>>)) \
  REGISTER_OP_IMPL(NAME, CPU, BFloat16, &(FUNC<DTypeToType_t<DType::BFloat16>>))

void registerNNLayerCpu() {
  // softmax
  REG_NN_LAYER_CPU_FLT(softmax, softmaxOpCpuImpl);
  REG_NN_LAYER_CPU_FLT(softmaxOut, softmaxOpOutCpuImpl);
  REG_NN_LAYER_CPU_FLT(softmaxBackward, softmaxOpBackwardCpuImpl);

  // logSoftmax
  REG_NN_LAYER_CPU_FLT(logSoftmax, logSoftmaxOpCpuImpl);
  REG_NN_LAYER_CPU_FLT(logSoftmaxOut, logSoftmaxOpOutCpuImpl);
  REG_NN_LAYER_CPU_FLT(logSoftmaxBackward, logSoftmaxOpBackwardCpuImpl);

  // dropout
  REG_NN_LAYER_CPU_FLT(dropout, dropoutOpCpuImpl);
  REG_NN_LAYER_CPU_FLT(dropoutMasked, dropoutMaskedOpCpuImpl);

  // layerNorm
  REG_NN_LAYER_CPU_FLT(layerNorm, layerNormOpCpuImpl);

  // rmsNorm
  REG_NN_LAYER_CPU_FLT(rmsNorm, rmsNormOpCpuImpl);

  // rope
  REG_NN_LAYER_CPU_FLT(ropeInit, ropeInitOpCpuImpl);
  REG_NN_LAYER_CPU_FLT(ropeApply, ropeApplyOpCpuImpl);
}

}  // namespace tinytorch::op
