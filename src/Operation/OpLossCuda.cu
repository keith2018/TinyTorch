/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "OpLossCuda.cuh"

namespace tinytorch::op {

void registerLossCuda() {
  // mseLoss
  REGISTER_OP_IMPL_DTYPE_TPL(mseLoss, CUDA, mseLossOpCudaImpl);
  REGISTER_OP_IMPL_DTYPE_TPL(mseLossBackward, CUDA, mseLossOpBackwardCudaImpl);

  // nllLoss
  REGISTER_OP_IMPL_DTYPE_TPL(nllLoss, CUDA, nllLossOpCudaImpl);
  REGISTER_OP_IMPL_DTYPE_TPL(nllLossBackward, CUDA, nllLossOpBackwardCudaImpl);
}

}  // namespace tinytorch::op