/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Tensor/Dispatch.h"

namespace tinytorch::op {

// unary
using UnaryOpFn = Tensor (*)(const Tensor& self);
using UnaryOpOutFn = void (*)(Tensor& out, const Tensor& self);
using UnaryOpInplaceFn = void (*)(Tensor& self);

using UnaryOpBackwardFn = Tensor (*)(const Tensor& grad, const Tensor& self);

// binary
using BinaryOpAlphaFn = Tensor (*)(const Tensor& self, const Tensor& other, const Scalar& alpha);
using BinaryOpAlphaOutFn = void (*)(Tensor& out, const Tensor& self, const Tensor& other, const Scalar& alpha);
using BinaryOpAlphaInplaceFn = void (*)(Tensor& self, const Tensor& other, const Scalar& alpha);

using BinaryOpFn = Tensor (*)(const Tensor& self, const Tensor& other);
using BinaryOpOutFn = void (*)(Tensor& out, const Tensor& self, const Tensor& other);
using BinaryOpInplaceFn = void (*)(Tensor& self, const Tensor& other);

using BinaryOpBackwardFn = Tensor (*)(const Tensor& grad, const Tensor& self, const Tensor& other);

// ternary
using TernaryOpFn = Tensor (*)(const Tensor& self, const Tensor& p1, const Tensor& p2);
using TernaryOpOutFn = void (*)(Tensor& out, const Tensor& self, const Tensor& p1, const Tensor& p2);
using TernaryOpInplaceFn = void (*)(Tensor& self, const Tensor& p1, const Tensor& p2);

// addcmul
using AddcmulOpFn = Tensor (*)(const Tensor& self, const Tensor& t1, const Tensor& t2, const Scalar& value);
using AddcmulOpOutFn = void (*)(Tensor& out, const Tensor& self, const Tensor& t1, const Tensor& t2,
                                const Scalar& value);
using AddcmulOpInplaceFn = void (*)(Tensor& self, const Tensor& t1, const Tensor& t2, const Scalar& value);

// abs
DEFINE_OP(abs, UnaryOpFn)
DEFINE_OP(absOut, UnaryOpOutFn)
DEFINE_OP(absInplace, UnaryOpInplaceFn)

// neg
DEFINE_OP(neg, UnaryOpFn)
DEFINE_OP(negOut, UnaryOpOutFn)
DEFINE_OP(negInplace, UnaryOpInplaceFn)

// sign
DEFINE_OP(sign, UnaryOpFn)
DEFINE_OP(signOut, UnaryOpOutFn)
DEFINE_OP(signInplace, UnaryOpInplaceFn)

// logicNot
DEFINE_OP(logicNot, UnaryOpFn)
DEFINE_OP(logicNotOut, UnaryOpOutFn)
DEFINE_OP(logicNotInplace, UnaryOpInplaceFn)

// sqrt
DEFINE_OP(sqrt, UnaryOpFn)
DEFINE_OP(sqrtOut, UnaryOpOutFn)
DEFINE_OP(sqrtInplace, UnaryOpInplaceFn)

// square
DEFINE_OP(square, UnaryOpFn)
DEFINE_OP(squareOut, UnaryOpOutFn)
DEFINE_OP(squareInplace, UnaryOpInplaceFn)

// exp
DEFINE_OP(exp, UnaryOpFn)
DEFINE_OP(expOut, UnaryOpOutFn)
DEFINE_OP(expInplace, UnaryOpInplaceFn)

// log
DEFINE_OP(log, UnaryOpFn)
DEFINE_OP(logOut, UnaryOpOutFn)
DEFINE_OP(logInplace, UnaryOpInplaceFn)

// sin
DEFINE_OP(sin, UnaryOpFn)
DEFINE_OP(sinOut, UnaryOpOutFn)
DEFINE_OP(sinInplace, UnaryOpInplaceFn)
DEFINE_OP(sinBackwardP1, UnaryOpBackwardFn)

// cos
DEFINE_OP(cos, UnaryOpFn)
DEFINE_OP(cosOut, UnaryOpOutFn)
DEFINE_OP(cosInplace, UnaryOpInplaceFn)
DEFINE_OP(cosBackwardP1, UnaryOpBackwardFn)

// sigmoid
DEFINE_OP(sigmoid, UnaryOpFn)
DEFINE_OP(sigmoidOut, UnaryOpOutFn)
DEFINE_OP(sigmoidInplace, UnaryOpInplaceFn)

// tanh
DEFINE_OP(tanh, UnaryOpFn)
DEFINE_OP(tanhOut, UnaryOpOutFn)
DEFINE_OP(tanhInplace, UnaryOpInplaceFn)

// relu
DEFINE_OP(relu, UnaryOpFn)
DEFINE_OP(reluOut, UnaryOpOutFn)
DEFINE_OP(reluInplace, UnaryOpInplaceFn)

// gelu
DEFINE_OP(gelu, UnaryOpFn)
DEFINE_OP(geluOut, UnaryOpOutFn)
DEFINE_OP(geluInplace, UnaryOpInplaceFn)

// silu
DEFINE_OP(silu, UnaryOpFn)
DEFINE_OP(siluOut, UnaryOpOutFn)
DEFINE_OP(siluInplace, UnaryOpInplaceFn)

// add
DEFINE_OP(add, BinaryOpAlphaFn)
DEFINE_OP(addOut, BinaryOpAlphaOutFn)
DEFINE_OP(addInplace, BinaryOpAlphaInplaceFn)

// sub
DEFINE_OP(sub, BinaryOpAlphaFn)
DEFINE_OP(subOut, BinaryOpAlphaOutFn)
DEFINE_OP(subInplace, BinaryOpAlphaInplaceFn)

// mul
DEFINE_OP(mul, BinaryOpFn)
DEFINE_OP(mulOut, BinaryOpOutFn)
DEFINE_OP(mulInplace, BinaryOpInplaceFn)

// div
DEFINE_OP(div, BinaryOpFn)
DEFINE_OP(divOut, BinaryOpOutFn)
DEFINE_OP(divInplace, BinaryOpInplaceFn)
DEFINE_OP(divBackwardP2, BinaryOpBackwardFn)

// pow
DEFINE_OP(pow, BinaryOpFn)
DEFINE_OP(powOut, BinaryOpOutFn)
DEFINE_OP(powInplace, BinaryOpInplaceFn)
DEFINE_OP(powBackwardP1, BinaryOpBackwardFn)
DEFINE_OP(powBackwardP2, BinaryOpBackwardFn)

// maximum
DEFINE_OP(maximum, BinaryOpFn)
DEFINE_OP(maximumOut, BinaryOpOutFn)

// minimum
DEFINE_OP(minimum, BinaryOpFn)
DEFINE_OP(minimumOut, BinaryOpOutFn)

// equal
DEFINE_OP(eq, BinaryOpFn)
DEFINE_OP(eqOut, BinaryOpOutFn)

// not equal
DEFINE_OP(ne, BinaryOpFn)
DEFINE_OP(neOut, BinaryOpOutFn)

// less than
DEFINE_OP(lt, BinaryOpFn)
DEFINE_OP(ltOut, BinaryOpOutFn)

// less equal
DEFINE_OP(le, BinaryOpFn)
DEFINE_OP(leOut, BinaryOpOutFn)

// greater than
DEFINE_OP(gt, BinaryOpFn)
DEFINE_OP(gtOut, BinaryOpOutFn)

// greater equal
DEFINE_OP(ge, BinaryOpFn)
DEFINE_OP(geOut, BinaryOpOutFn)

// clampMin
DEFINE_OP(clampMin, BinaryOpFn)
DEFINE_OP(clampMinOut, BinaryOpOutFn)
DEFINE_OP(clampMinInplace, BinaryOpInplaceFn)

// clampMax
DEFINE_OP(clampMax, BinaryOpFn)
DEFINE_OP(clampMaxOut, BinaryOpOutFn)
DEFINE_OP(clampMaxInplace, BinaryOpInplaceFn)

// clamp
DEFINE_OP(clamp, TernaryOpFn)
DEFINE_OP(clampOut, TernaryOpOutFn)
DEFINE_OP(clampInplace, TernaryOpInplaceFn)

// addcmul
DEFINE_OP(addcmul, AddcmulOpFn)
DEFINE_OP(addcmulOut, AddcmulOpOutFn)
DEFINE_OP(addcmulInplace, AddcmulOpInplaceFn)

void registerElemWiseCpu();
STATIC_CALL(registerElemWiseCpu);

#ifdef USE_CUDA
void registerElemWiseCuda();
STATIC_CALL(registerElemWiseCuda);
#endif

}  // namespace tinytorch::op
