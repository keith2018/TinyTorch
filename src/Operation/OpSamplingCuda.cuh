/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "OpSampling.h"

namespace tinytorch::op {

template <typename T>
TensorPair topkOpCudaImpl(const Tensor& self, int64_t k, int64_t dim, bool largest, bool sorted);

template <typename T>
Tensor multinomialOpCudaImpl(const Tensor& self, int64_t nSamples, bool replacement);

template <typename T>
TensorPair sortOpCudaImpl(const Tensor& self, int64_t dim, bool descending);

template <typename T>
Tensor cumsumOpCudaImpl(const Tensor& self, int64_t dim);

}  // namespace tinytorch::op
