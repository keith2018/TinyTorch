/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Tensor/Dispatch.h"

namespace tinytorch::op {

SizeVector getReduceShape(const Tensor& t, int64_t dim, bool keepDims);
SizeVector getReduceShape(const Tensor& t, const DimArray<int64_t>& inAxis, bool keepDims);
Options getIndicesOptions(const Tensor& t);

using ReduceOpFn = Tensor (*)(const Tensor& self);
using ReduceOpDimFn = Tensor (*)(const Tensor& self, int64_t dim, bool keepDim);
using ReduceOpDimIndicesFn = TensorPair (*)(const Tensor& self, int64_t dim, bool keepDim);
using ReduceOpMultiDimsFn = Tensor (*)(const Tensor& self, IntArrayView dims, bool keepDim);

using VarMeanOpFn = TensorPair (*)(const Tensor& self, bool unbiased);
using VarMeanOpDimFn = TensorPair (*)(const Tensor& self, int64_t dim, bool unbiased, bool keepDim);
using VarMeanOpMultiDimsFn = TensorPair (*)(const Tensor& self, IntArrayView dims, bool unbiased, bool keepDim);

// min/argmin
DEFINE_OP(min, ReduceOpFn)
DEFINE_OP(argmin, ReduceOpFn)
DEFINE_OP(minOnDim, ReduceOpDimIndicesFn)

// max/argmax
DEFINE_OP(max, ReduceOpFn)
DEFINE_OP(argmax, ReduceOpFn)
DEFINE_OP(maxOnDim, ReduceOpDimIndicesFn)

// sum
DEFINE_OP(sum, ReduceOpFn)
DEFINE_OP(sumOnDim, ReduceOpDimFn)
DEFINE_OP(sumOnDims, ReduceOpMultiDimsFn)

// mean
DEFINE_OP(mean, ReduceOpFn)
DEFINE_OP(meanOnDim, ReduceOpDimFn)
DEFINE_OP(meanOnDims, ReduceOpMultiDimsFn)

// varMean
DEFINE_OP(varMean, VarMeanOpFn)
DEFINE_OP(varMeanOnDim, VarMeanOpDimFn)
DEFINE_OP(varMeanOnDims, VarMeanOpMultiDimsFn)

void registerReduceCpu();
STATIC_CALL(registerReduceCpu);

#ifdef USE_CUDA
void registerReduceCuda();
STATIC_CALL(registerReduceCuda);
#endif

}  // namespace tinytorch::op
