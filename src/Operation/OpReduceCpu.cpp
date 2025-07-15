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

#define REG_REDUCE_CPU_F32(NAME, FUNC, OP) \
  REGISTER_OP_IMPL(NAME, CPU, Float32, &(FUNC<DTypeToType_t<DType::Float32>, OP>))

void registerReduceCpuFloat32() {
  // min/argmin
  REG_REDUCE_CPU_F32(min, reduceOpAllCpuImpl, OpCpuReduceMin);
  REG_REDUCE_CPU_F32(argmin, reduceOpArgMinMaxCpuImpl, OpCpuReduceMin);
  REG_REDUCE_CPU_F32(minOnDim, reduceOpMinMaxDimCpuImpl, OpCpuReduceMin);

  // max/argmax
  REG_REDUCE_CPU_F32(max, reduceOpAllCpuImpl, OpCpuReduceMax);
  REG_REDUCE_CPU_F32(argmax, reduceOpArgMinMaxCpuImpl, OpCpuReduceMax);
  REG_REDUCE_CPU_F32(maxOnDim, reduceOpMinMaxDimCpuImpl, OpCpuReduceMax);

  // sum
  REGISTER_OP_IMPL(sum, CPU, Float32, &reduceOpSumCpuImpl<DTypeToType_t<DType::Float32>>);
  REGISTER_OP_IMPL(sumOnDim, CPU, Float32, &reduceOpSumDimCpuImpl<DTypeToType_t<DType::Float32>>);
  REGISTER_OP_IMPL(sumOnDims, CPU, Float32, &reduceOpSumDimsCpuImpl<DTypeToType_t<DType::Float32>>);

  // mean
  REGISTER_OP_IMPL(mean, CPU, Float32, &reduceOpMeanCpuImpl<DTypeToType_t<DType::Float32>>);
  REGISTER_OP_IMPL(meanOnDim, CPU, Float32, &reduceOpMeanDimCpuImpl<DTypeToType_t<DType::Float32>>);
  REGISTER_OP_IMPL(meanOnDims, CPU, Float32, &reduceOpMeanDimsCpuImpl<DTypeToType_t<DType::Float32>>);

  // varMean
  REGISTER_OP_IMPL(varMean, CPU, Float32, &reduceOpVarMeanCpuImpl<DTypeToType_t<DType::Float32>>);
  REGISTER_OP_IMPL(varMeanOnDim, CPU, Float32, &reduceOpVarMeanDimCpuImpl<DTypeToType_t<DType::Float32>>);
  REGISTER_OP_IMPL(varMeanOnDims, CPU, Float32, &reduceOpVarMeanDimsCpuImpl<DTypeToType_t<DType::Float32>>);
}

void registerReduceCpu() { registerReduceCpuFloat32(); }

}  // namespace tinytorch::op
