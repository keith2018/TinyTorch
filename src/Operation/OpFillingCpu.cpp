/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "OpFillingCpu.h"

namespace tinytorch::op {

#define REG_FILL_CPU_FLT(NAME, FUNC)                                           \
  REGISTER_OP_IMPL(NAME, CPU, Float32, &(FUNC<DTypeToType_t<DType::Float32>>)) \
  REGISTER_OP_IMPL(NAME, CPU, Float16, &(FUNC<DTypeToType_t<DType::Float16>>)) \
  REGISTER_OP_IMPL(NAME, CPU, BFloat16, &(FUNC<DTypeToType_t<DType::BFloat16>>))

void registerFillCpuFloat() {
  REG_FILL_CPU_FLT(fillRandUniform, fillOpRandUniformCpuImpl);
  REG_FILL_CPU_FLT(fillRandNormal, fillOpRandNormalCpuImpl);
  REG_FILL_CPU_FLT(fillRandBernoulli, fillOpRandBernoulliCpuImpl);
}

void registerFillCpu() {
  REGISTER_OP_IMPL_DTYPE_TPL(fill, CPU, fillOpCpuImpl);
  REGISTER_OP_IMPL_DTYPE_TPL(fillOffset, CPU, fillOpOffsetCpuImpl);

  REGISTER_OP_IMPL_DTYPE_TPL(fillMasked, CPU, fillOpMaskedCpuImpl);
  REGISTER_OP_IMPL_DTYPE_TPL(fillMaskedOut, CPU, fillOpMaskedOutCpuImpl);
  REGISTER_OP_IMPL_DTYPE_TPL(fillMaskedInplace, CPU, fillOpMaskedInplaceCpuImpl);

  REGISTER_OP_IMPL_DTYPE_TPL(fillLinSpace, CPU, fillOpLinSpaceCpuImpl);

  registerFillCpuFloat();
}

}  // namespace tinytorch::op