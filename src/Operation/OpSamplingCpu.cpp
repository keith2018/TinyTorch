/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "OpSamplingCpu.h"

namespace tinytorch::op {

void registerSamplingCpu() {
  // topk
  REGISTER_OP_IMPL_DTYPE_TPL(topk, CPU, topkOpCpuImpl);

  // multinomial
  REGISTER_OP_IMPL_DTYPE_TPL(multinomial, CPU, multinomialOpCpuImpl);

  // sort
  REGISTER_OP_IMPL_DTYPE_TPL(sort, CPU, sortOpCpuImpl)

  // cumsum
  REGISTER_OP_IMPL_DTYPE_TPL(cumsum, CPU, cumsumOpCpuImpl)
}

}  // namespace tinytorch::op