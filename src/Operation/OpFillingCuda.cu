/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "OpFillingCuda.cuh"

namespace tinytorch::op {

#define REG_FILL_CUDA_FLT(NAME, FUNC)                                           \
  REGISTER_OP_IMPL(NAME, CUDA, Float32, &(FUNC<DTypeToType_t<DType::Float32>>)) \
  REGISTER_OP_IMPL(NAME, CUDA, Float16, &(FUNC<DTypeToType_t<DType::Float16>>)) \
  REGISTER_OP_IMPL(NAME, CUDA, BFloat16, &(FUNC<DTypeToType_t<DType::BFloat16>>))

void registerFillCudaFloat() {
  REG_FILL_CUDA_FLT(fillRandUniform, fillOpRandUniformCudaImpl);
  REG_FILL_CUDA_FLT(fillRandNormal, fillOpRandNormalCudaImpl);
  REG_FILL_CUDA_FLT(fillRandBernoulli, fillOpRandBernoulliCudaImpl);
}

void registerFillCuda() {
  REGISTER_OP_IMPL_DTYPE_TPL(fill, CUDA, fillOpCudaImpl);
  REGISTER_OP_IMPL_DTYPE_TPL(fillOffset, CUDA, fillOpOffsetCudaImpl);

  REGISTER_OP_IMPL_DTYPE_TPL(fillMasked, CUDA, fillOpMaskedCudaImpl);
  REGISTER_OP_IMPL_DTYPE_TPL(fillMaskedOut, CUDA, fillOpMaskedOutCudaImpl);
  REGISTER_OP_IMPL_DTYPE_TPL(fillMaskedInplace, CUDA, fillOpMaskedInplaceCudaImpl);

  REGISTER_OP_IMPL_DTYPE_TPL(fillLinSpace, CUDA, fillOpLinSpaceCudaImpl);

  registerFillCudaFloat();
}

}  // namespace tinytorch::op