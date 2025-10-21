/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Tensor/Dispatch.h"
#include "Tensor/Scalar.h"

namespace tinytorch::op {

using SiluMulOpFn = Tensor (*)(const Tensor& self);

// siluMul
DEFINE_OP(siluMul, SiluMulOpFn)

void registerFusedCpu();
STATIC_CALL(registerFusedCpu);

#ifdef USE_CUDA
void registerFusedCuda();
STATIC_CALL(registerFusedCuda);
#endif

}  // namespace tinytorch::op
