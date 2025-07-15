/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "OpLinalgCuda.cuh"

namespace tinytorch::op {

#define REG_LINALG_CUDA_F32(NAME, FUNC) REGISTER_OP_IMPL(NAME, CUDA, Float32, &(FUNC<DTypeToType_t<DType::Float32>>))

void registerLinalgCudaFloat32() {
  // dot
  REG_LINALG_CUDA_F32(dot, dotOpCudaImpl);

  // matmul
  REG_LINALG_CUDA_F32(im2col, im2colOpCudaImpl);
  REG_LINALG_CUDA_F32(col2im, col2imOpCudaImpl);
}

void registerLinalgCuda() { registerLinalgCudaFloat32(); }

}  // namespace tinytorch::op