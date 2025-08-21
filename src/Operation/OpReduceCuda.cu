/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "OpReduceCuda.cuh"

namespace tinytorch::op {

#define REG_REDUCE_CUDA_FLT(NAME, FUNC, OP)                                         \
  REGISTER_OP_IMPL(NAME, CUDA, Float32, &(FUNC<DTypeToType_t<DType::Float32>, OP>)) \
  REGISTER_OP_IMPL(NAME, CUDA, Float16, &(FUNC<DTypeToType_t<DType::Float16>, OP>)) \
  REGISTER_OP_IMPL(NAME, CUDA, BFloat16, &(FUNC<DTypeToType_t<DType::BFloat16>, OP>))

#define REG_REDUCE_CUDA_FLT_NO_OP(NAME, FUNC)                                   \
  REGISTER_OP_IMPL(NAME, CUDA, Float32, &(FUNC<DTypeToType_t<DType::Float32>>)) \
  REGISTER_OP_IMPL(NAME, CUDA, Float16, &(FUNC<DTypeToType_t<DType::Float16>>)) \
  REGISTER_OP_IMPL(NAME, CUDA, BFloat16, &(FUNC<DTypeToType_t<DType::BFloat16>>))

void registerReduceCudaFloat() {
  // min/argmin
  REG_REDUCE_CUDA_FLT(min, reduceOpAllCudaImpl, OpCudaReduceMin);
  REG_REDUCE_CUDA_FLT(argmin, reduceOpArgMinMaxCudaImpl, OpCudaReduceMin);
  REG_REDUCE_CUDA_FLT(minOnDim, reduceOpMinMaxDimCudaImpl, OpCudaReduceMin);

  // max/argmax
  REG_REDUCE_CUDA_FLT(max, reduceOpAllCudaImpl, OpCudaReduceMax);
  REG_REDUCE_CUDA_FLT(argmax, reduceOpArgMinMaxCudaImpl, OpCudaReduceMax);
  REG_REDUCE_CUDA_FLT(maxOnDim, reduceOpMinMaxDimCudaImpl, OpCudaReduceMax);

  // sum
  REG_REDUCE_CUDA_FLT_NO_OP(sum, reduceOpSumCudaImpl);
  REG_REDUCE_CUDA_FLT_NO_OP(sumOnDim, reduceOpSumDimCudaImpl);
  REG_REDUCE_CUDA_FLT_NO_OP(sumOnDims, reduceOpSumDimsCudaImpl);

  // mean
  REG_REDUCE_CUDA_FLT_NO_OP(mean, reduceOpMeanCudaImpl);
  REG_REDUCE_CUDA_FLT_NO_OP(meanOnDim, reduceOpMeanDimCudaImpl);
  REG_REDUCE_CUDA_FLT_NO_OP(meanOnDims, reduceOpMeanDimsCudaImpl);

  // varMean
  REG_REDUCE_CUDA_FLT_NO_OP(varMean, reduceOpVarMeanCudaImpl);
  REG_REDUCE_CUDA_FLT_NO_OP(varMeanOnDim, reduceOpVarMeanDimCudaImpl);
  REG_REDUCE_CUDA_FLT_NO_OP(varMeanOnDims, reduceOpVarMeanDimsCudaImpl);
}

void registerReduceCuda() { registerReduceCudaFloat(); }

}  // namespace tinytorch::op