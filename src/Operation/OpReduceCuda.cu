/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "OpReduceCuda.cuh"

namespace tinytorch::op {

#define REG_REDUCE_CUDA_F32(NAME, FUNC, OP) \
  REGISTER_OP_IMPL(NAME, CUDA, Float32, &(FUNC<DTypeToType_t<DType::Float32>, OP>))

void registerReduceCudaFloat32() {
  // min/argmin
  REG_REDUCE_CUDA_F32(min, reduceOpAllCudaImpl, OpCudaReduceMin);
  REG_REDUCE_CUDA_F32(argmin, reduceOpArgMinMaxCudaImpl, OpCudaReduceMin);
  REG_REDUCE_CUDA_F32(minOnDim, reduceOpMinMaxDimCudaImpl, OpCudaReduceMin);

  // max/argmax
  REG_REDUCE_CUDA_F32(max, reduceOpAllCudaImpl, OpCudaReduceMax);
  REG_REDUCE_CUDA_F32(argmax, reduceOpArgMinMaxCudaImpl, OpCudaReduceMax);
  REG_REDUCE_CUDA_F32(maxOnDim, reduceOpMinMaxDimCudaImpl, OpCudaReduceMax);

  // sum
  REGISTER_OP_IMPL(sum, CUDA, Float32, &reduceOpSumCudaImpl<DTypeToType_t<DType::Float32>>);
  REGISTER_OP_IMPL(sumOnDim, CUDA, Float32, &reduceOpSumDimCudaImpl<DTypeToType_t<DType::Float32>>);
  REGISTER_OP_IMPL(sumOnDims, CUDA, Float32, &reduceOpSumDimsCudaImpl<DTypeToType_t<DType::Float32>>);

  // mean
  REGISTER_OP_IMPL(mean, CUDA, Float32, &reduceOpMeanCudaImpl<DTypeToType_t<DType::Float32>>);
  REGISTER_OP_IMPL(meanOnDim, CUDA, Float32, &reduceOpMeanDimCudaImpl<DTypeToType_t<DType::Float32>>);
  REGISTER_OP_IMPL(meanOnDims, CUDA, Float32, &reduceOpMeanDimsCudaImpl<DTypeToType_t<DType::Float32>>);

  // varMean
  REGISTER_OP_IMPL(varMean, CUDA, Float32, &reduceOpVarMeanCudaImpl<DTypeToType_t<DType::Float32>>);
  REGISTER_OP_IMPL(varMeanOnDim, CUDA, Float32, &reduceOpVarMeanDimCudaImpl<DTypeToType_t<DType::Float32>>);
  REGISTER_OP_IMPL(varMeanOnDims, CUDA, Float32, &reduceOpVarMeanDimsCudaImpl<DTypeToType_t<DType::Float32>>);
}

void registerReduceCuda() { registerReduceCudaFloat32(); }

}  // namespace tinytorch::op