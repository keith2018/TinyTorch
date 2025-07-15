/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "OpLossCpu.h"

namespace tinytorch::op {

void registerLossCpu() {
  // mseLoss
  REGISTER_OP_IMPL_DTYPE_TPL(mseLoss, CPU, mseLossOpCpuImpl);
  REGISTER_OP_IMPL_DTYPE_TPL(mseLossBackward, CPU, mseLossOpBackwardCpuImpl);

  // nllLoss
  REGISTER_OP_IMPL_DTYPE_TPL(nllLoss, CPU, nllLossOpCpuImpl);
  REGISTER_OP_IMPL_DTYPE_TPL(nllLossBackward, CPU, nllLossOpBackwardCpuImpl);
}

}  // namespace tinytorch::op