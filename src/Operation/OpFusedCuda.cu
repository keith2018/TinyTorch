/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "OpFusedCuda.cuh"

namespace tinytorch::op {

#define REG_FUSED_CUDA_FLT(NAME, FUNC)                                          \
  REGISTER_OP_IMPL(NAME, CUDA, Float32, &(FUNC<DTypeToType_t<DType::Float32>>)) \
  REGISTER_OP_IMPL(NAME, CUDA, Float16, &(FUNC<DTypeToType_t<DType::Float16>>)) \
  REGISTER_OP_IMPL(NAME, CUDA, BFloat16, &(FUNC<DTypeToType_t<DType::BFloat16>>))

void registerFusedCuda() {
  // siluMul
  REG_FUSED_CUDA_FLT(siluMul, siluMulOpCudaImpl);
}

}  // namespace tinytorch::op