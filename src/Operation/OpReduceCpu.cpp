/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "OpReduceCpu.h"

namespace tinytorch::op {

int64_t ReducerCpu::getReduceSrcIndex(const Tensor& ret, const Tensor& t, int64_t idx, int64_t dim, bool keepDims) {
  int64_t outIndex = idx;
  int64_t inIndex = 0;
  for (int64_t d = ret.dim() - 1; d >= 0; d--) {
    int64_t coord = outIndex % ret.shape(d);
    outIndex /= ret.shape(d);
    if (keepDims || d < dim) {
      inIndex += coord * t.stride(d);
    } else {
      inIndex += coord * t.stride(d + 1);
    }
  }
  return inIndex;
}

int64_t ReducerCpu::getReduceDstIndex(const Tensor& t, int64_t idx, int64_t dim) {
  int64_t retIdx = 0;
  int64_t stride = 1;
  for (int64_t d = t.dim() - 1; d >= 0; d--) {
    if (d != dim) {
      retIdx += (idx / t.stride(d) % t.shape(d)) * stride;
      stride *= t.shape(d);
    }
  }
  return retIdx;
}

int64_t ReducerCpu::getReduceDstIndex(const Tensor& t, int64_t idx, const DimArray<int64_t>& inAxis) {
  int64_t retIdx = 0;
  int64_t stride = 1;
  for (int64_t d = t.dim() - 1; d >= 0; d--) {
    if (0 == inAxis.data[d]) {
      retIdx += (idx / t.stride(d) % t.shape(d)) * stride;
      stride *= t.shape(d);
    }
  }
  return retIdx;
}

#define REG_REDUCE_CPU_FLT(NAME, FUNC, OP)                                         \
  REGISTER_OP_IMPL(NAME, CPU, Float32, &(FUNC<DTypeToType_t<DType::Float32>, OP>)) \
  REGISTER_OP_IMPL(NAME, CPU, Float16, &(FUNC<DTypeToType_t<DType::Float16>, OP>)) \
  REGISTER_OP_IMPL(NAME, CPU, BFloat16, &(FUNC<DTypeToType_t<DType::BFloat16>, OP>))

#define REG_REDUCE_CPU_FLT_NO_OP(NAME, FUNC)                                   \
  REGISTER_OP_IMPL(NAME, CPU, Float32, &(FUNC<DTypeToType_t<DType::Float32>>)) \
  REGISTER_OP_IMPL(NAME, CPU, Float16, &(FUNC<DTypeToType_t<DType::Float16>>)) \
  REGISTER_OP_IMPL(NAME, CPU, BFloat16, &(FUNC<DTypeToType_t<DType::BFloat16>>))

void registerReduceCpuFloat() {
  // min/argmin
  REG_REDUCE_CPU_FLT(min, reduceOpAllCpuImpl, OpCpuReduceMin);
  REG_REDUCE_CPU_FLT(argmin, reduceOpArgMinMaxCpuImpl, OpCpuReduceMin);
  REG_REDUCE_CPU_FLT(minOnDim, reduceOpMinMaxDimCpuImpl, OpCpuReduceMin);

  // max/argmax
  REG_REDUCE_CPU_FLT(max, reduceOpAllCpuImpl, OpCpuReduceMax);
  REG_REDUCE_CPU_FLT(argmax, reduceOpArgMinMaxCpuImpl, OpCpuReduceMax);
  REG_REDUCE_CPU_FLT(maxOnDim, reduceOpMinMaxDimCpuImpl, OpCpuReduceMax);

  // sum
  REG_REDUCE_CPU_FLT_NO_OP(sum, reduceOpSumCpuImpl);
  REG_REDUCE_CPU_FLT_NO_OP(sumOnDim, reduceOpSumDimCpuImpl);
  REG_REDUCE_CPU_FLT_NO_OP(sumOnDims, reduceOpSumDimsCpuImpl);

  // mean
  REG_REDUCE_CPU_FLT_NO_OP(mean, reduceOpMeanCpuImpl);
  REG_REDUCE_CPU_FLT_NO_OP(meanOnDim, reduceOpMeanDimCpuImpl);
  REG_REDUCE_CPU_FLT_NO_OP(meanOnDims, reduceOpMeanDimsCpuImpl);

  // varMean
  REG_REDUCE_CPU_FLT_NO_OP(varMean, reduceOpVarMeanCpuImpl);
  REG_REDUCE_CPU_FLT_NO_OP(varMeanOnDim, reduceOpVarMeanDimCpuImpl);
  REG_REDUCE_CPU_FLT_NO_OP(varMeanOnDims, reduceOpVarMeanDimsCpuImpl);
}

void registerReduceCpu() { registerReduceCpuFloat(); }

}  // namespace tinytorch::op
