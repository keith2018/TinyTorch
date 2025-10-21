/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "OpFusedCpu.h"

namespace tinytorch::op {

#define REG_FUSED_CPU_FLT(NAME, FUNC)                                          \
  REGISTER_OP_IMPL(NAME, CPU, Float32, &(FUNC<DTypeToType_t<DType::Float32>>)) \
  REGISTER_OP_IMPL(NAME, CPU, Float16, &(FUNC<DTypeToType_t<DType::Float16>>)) \
  REGISTER_OP_IMPL(NAME, CPU, BFloat16, &(FUNC<DTypeToType_t<DType::BFloat16>>))

void registerFusedCpu() {
  // siluMul
  REG_FUSED_CPU_FLT(siluMul, siluMulOpCpuImpl);
}

}  // namespace tinytorch::op