/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "OpSamplingCuda.cuh"

namespace tinytorch::op {

void registerSamplingCuda() {
  // topk
  REGISTER_OP_IMPL_DTYPE_TPL(topk, CUDA, topkOpCudaImpl);

  // multinomial
  REGISTER_OP_IMPL_DTYPE_TPL(multinomial, CUDA, multinomialOpCudaImpl);

  // sort
  REGISTER_OP_IMPL_DTYPE_TPL(sort, CUDA, sortOpCudaImpl)

  // cumsum
  REGISTER_OP_IMPL_DTYPE_TPL(cumsum, CUDA, cumsumOpCudaImpl)
}

}  // namespace tinytorch::op