/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "OpLinalgCuda.cuh"

namespace tinytorch::op {

#define REG_LINALG_CUDA_FLT(NAME, FUNC)                                         \
  REGISTER_OP_IMPL(NAME, CUDA, Float32, &(FUNC<DTypeToType_t<DType::Float32>>)) \
  REGISTER_OP_IMPL(NAME, CUDA, Float16, &(FUNC<DTypeToType_t<DType::Float16>>)) \
  REGISTER_OP_IMPL(NAME, CUDA, BFloat16, &(FUNC<DTypeToType_t<DType::BFloat16>>))

void registerLinalgCudaFloat() {
  // dot
  REG_LINALG_CUDA_FLT(dot, dotOpCudaImpl);

  // matmul
  REG_LINALG_CUDA_FLT(im2col, im2colOpCudaImpl);
  REG_LINALG_CUDA_FLT(col2im, col2imOpCudaImpl);
}

void registerLinalgCuda() { registerLinalgCudaFloat(); }

}  // namespace tinytorch::op