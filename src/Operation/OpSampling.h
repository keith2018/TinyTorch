/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Tensor/Dispatch.h"

namespace tinytorch::op {

using TopkOpFn = TensorPair (*)(const Tensor& self, int64_t k, int64_t dim, bool largest, bool sorted);

using MultinomialOpFn = Tensor (*)(const Tensor& self, int64_t numSamples, bool replacement);

using SortOpFn = TensorPair (*)(const Tensor& self, int64_t dim, bool descending);

using CumsumOpFn = Tensor (*)(const Tensor& self, int64_t dim);

// topk
DEFINE_OP(topk, TopkOpFn)

// multinomial
DEFINE_OP(multinomial, MultinomialOpFn)

// sort
DEFINE_OP(sort, SortOpFn)

// cumsum
DEFINE_OP(cumsum, CumsumOpFn)

void registerSamplingCpu();
STATIC_CALL(registerSamplingCpu);

#ifdef USE_CUDA
void registerSamplingCuda();
STATIC_CALL(registerSamplingCuda);
#endif

}  // namespace tinytorch::op
