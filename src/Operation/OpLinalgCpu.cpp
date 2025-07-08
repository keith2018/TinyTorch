/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "OpLinalgCpu.h"

namespace tinytorch::op {

void registerLinalgCpu() {
  // dot
  REGISTER_OP_IMPL_DTYPE_TPL(dot, CPU, dotOpCpuImpl);

  // matmul
  REGISTER_OP_IMPL_DTYPE_TPL(im2col, CPU, im2colOpCpuImpl);
  REGISTER_OP_IMPL_DTYPE_TPL(col2im, CPU, col2imOpCpuImpl);
}

}  // namespace tinytorch::op
