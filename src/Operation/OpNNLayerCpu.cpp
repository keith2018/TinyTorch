/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "OpNNLayerCpu.h"

namespace tinytorch::op {

#define REG_NN_LAYER_CPU_F32(NAME, FUNC) REGISTER_OP_IMPL(NAME, CPU, Float32, &(FUNC<DTypeToType_t<DType::Float32>>))

void registerNNLayerCpu() {
  // softmax
  REG_NN_LAYER_CPU_F32(softmax, softmaxOpCpuImpl);
  REG_NN_LAYER_CPU_F32(softmaxOut, softmaxOpOutCpuImpl);
  REG_NN_LAYER_CPU_F32(softmaxBackward, softmaxOpBackwardCpuImpl);

  // logSoftmax
  REG_NN_LAYER_CPU_F32(logSoftmax, logSoftmaxOpCpuImpl);
  REG_NN_LAYER_CPU_F32(logSoftmaxOut, logSoftmaxOpOutCpuImpl);
  REG_NN_LAYER_CPU_F32(logSoftmaxBackward, logSoftmaxOpBackwardCpuImpl);

  // dropout
  REG_NN_LAYER_CPU_F32(dropout, dropoutOpCpuImpl);
  REG_NN_LAYER_CPU_F32(dropoutMasked, dropoutMaskedOpCpuImpl);

  // layerNorm
  REG_NN_LAYER_CPU_F32(layerNorm, layerNormOpCpuImpl);

  // rmsNorm
  REG_NN_LAYER_CPU_F32(rmsNorm, rmsNormOpCpuImpl);

  // rope
  REG_NN_LAYER_CPU_F32(ropeInit, ropeInitOpCpuImpl);
  REG_NN_LAYER_CPU_F32(ropeApply, ropeApplyOpCpuImpl);
}

}  // namespace tinytorch::op
