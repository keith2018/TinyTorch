/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Tensor/Dispatch.h"
#include "Tensor/Scalar.h"

namespace tinytorch::op {

using FillOpFn = void (*)(Tensor& self, const Scalar& val);
using FillOpOffsetFn = void (*)(Tensor& self, const Tensor& val, int64_t offset, int64_t count);

using FillOpMaskedFn = Tensor (*)(const Tensor& self, const Tensor& mask, const Scalar& val);
using FillOpMaskedOutFn = void (*)(Tensor& out, const Tensor& self, const Tensor& mask, const Scalar& val);
using FillOpMaskedInplaceFn = void (*)(Tensor& self, const Tensor& mask, const Scalar& val);

using FillOpLinSpaceFn = void (*)(Tensor& self, const Scalar& start, const Scalar& step, int64_t steps);

using FillOpRandUniformFn = void (*)(Tensor& self, float min, float max);
using FillOpRandNormalFn = void (*)(Tensor& self);
using FillOpRandBernoulliFn = void (*)(Tensor& self, float p);

DEFINE_OP(fill, FillOpFn)
DEFINE_OP(fillOffset, FillOpOffsetFn)

DEFINE_OP(fillMasked, FillOpMaskedFn)
DEFINE_OP(fillMaskedOut, FillOpMaskedOutFn)
DEFINE_OP(fillMaskedInplace, FillOpMaskedInplaceFn)

DEFINE_OP(fillLinSpace, FillOpLinSpaceFn)

DEFINE_OP(fillRandUniform, FillOpRandUniformFn)
DEFINE_OP(fillRandNormal, FillOpRandNormalFn)
DEFINE_OP(fillRandBernoulli, FillOpRandBernoulliFn)

void registerFillCpu();
void registerFillCuda();

STATIC_CALL(registerFillCpu);
STATIC_CALL(registerFillCuda);

}  // namespace tinytorch::op
