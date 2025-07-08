/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "OpNNLayerCpu.h"

namespace tinytorch::op {

void registerNNLayerCpu() {
  // softmax
  REGISTER_OP_IMPL_DTYPE_TPL(softmax, CPU, softmaxOpCpuImpl);
  REGISTER_OP_IMPL_DTYPE_TPL(softmaxOut, CPU, softmaxOpOutCpuImpl);
  REGISTER_OP_IMPL_DTYPE_TPL(softmaxBackward, CPU, softmaxOpBackwardCpuImpl);

  // logSoftmax
  REGISTER_OP_IMPL_DTYPE_TPL(logSoftmax, CPU, logSoftmaxOpCpuImpl);
  REGISTER_OP_IMPL_DTYPE_TPL(logSoftmaxOut, CPU, logSoftmaxOpOutCpuImpl);
  REGISTER_OP_IMPL_DTYPE_TPL(logSoftmaxBackward, CPU, logSoftmaxOpBackwardCpuImpl);

  // dropout
  REGISTER_OP_IMPL_DTYPE_TPL(dropout, CPU, dropoutOpCpuImpl);
}

}  // namespace tinytorch::op
