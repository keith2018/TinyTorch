/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "OpLinalgCpu.h"

namespace tinytorch::op {

#define REG_LINALG_CPU_FLT(NAME, FUNC)                                         \
  REGISTER_OP_IMPL(NAME, CPU, Float32, &(FUNC<DTypeToType_t<DType::Float32>>)) \
  REGISTER_OP_IMPL(NAME, CPU, Float16, &(FUNC<DTypeToType_t<DType::Float16>>)) \
  REGISTER_OP_IMPL(NAME, CPU, BFloat16, &(FUNC<DTypeToType_t<DType::BFloat16>>))

void registerLinalgCpuFloat() {
  // dot
  REG_LINALG_CPU_FLT(dot, dotOpCpuImpl);

  // matmul
  REG_LINALG_CPU_FLT(im2col, im2colOpCpuImpl);
  REG_LINALG_CPU_FLT(col2im, col2imOpCpuImpl);
}

void registerLinalgCpu() { registerLinalgCpuFloat(); }

}  // namespace tinytorch::op
