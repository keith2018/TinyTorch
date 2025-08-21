/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "OpElemWiseCpu.h"

namespace tinytorch::op {

#define REG_ELEM_WISE_CPU_BOOL(NAME, FUNC, OP) \
  REGISTER_OP_IMPL(NAME, CPU, Bool, &(FUNC<DTypeToType_t<DType::Bool>, OP>))

#define REG_ELEM_WISE_CPU_FLT(NAME, FUNC, OP)                                      \
  REGISTER_OP_IMPL(NAME, CPU, Float32, &(FUNC<DTypeToType_t<DType::Float32>, OP>)) \
  REGISTER_OP_IMPL(NAME, CPU, Float16, &(FUNC<DTypeToType_t<DType::Float16>, OP>)) \
  REGISTER_OP_IMPL(NAME, CPU, BFloat16, &(FUNC<DTypeToType_t<DType::BFloat16>, OP>))

#define REG_ELEM_WISE_CPU_FLT_NO_OP(NAME, FUNC)                                \
  REGISTER_OP_IMPL(NAME, CPU, Float32, &(FUNC<DTypeToType_t<DType::Float32>>)) \
  REGISTER_OP_IMPL(NAME, CPU, Float16, &(FUNC<DTypeToType_t<DType::Float16>>)) \
  REGISTER_OP_IMPL(NAME, CPU, BFloat16, &(FUNC<DTypeToType_t<DType::BFloat16>>))

#define REG_ELEM_WISE_CPU_I64(NAME, FUNC, OP) \
  REGISTER_OP_IMPL(NAME, CPU, Int64, &(FUNC<DTypeToType_t<DType::Int64>, OP>))

void registerUnaryCpuBool() {
  // logicNot
  REG_ELEM_WISE_CPU_BOOL(logicNot, unaryOpCpuImpl, OpCpuLogicNot);
  REG_ELEM_WISE_CPU_BOOL(logicNotOut, unaryOpOutCpuImpl, OpCpuLogicNot);
  REG_ELEM_WISE_CPU_BOOL(logicNotInplace, unaryOpInplaceCpuImpl, OpCpuLogicNot);

  // logicAnd
  REG_ELEM_WISE_CPU_BOOL(logicAnd, binaryOpCpuImpl, OpCpuLogicAnd);
  REG_ELEM_WISE_CPU_BOOL(logicAndOut, binaryOpOutCpuImpl, OpCpuLogicAnd);
  REG_ELEM_WISE_CPU_BOOL(logicAndInplace, binaryOpInplaceCpuImpl, OpCpuLogicAnd);

  // logicOr
  REG_ELEM_WISE_CPU_BOOL(logicOr, binaryOpCpuImpl, OpCpuLogicOr);
  REG_ELEM_WISE_CPU_BOOL(logicOrOut, binaryOpOutCpuImpl, OpCpuLogicOr);
  REG_ELEM_WISE_CPU_BOOL(logicOrInplace, binaryOpInplaceCpuImpl, OpCpuLogicOr);
}

void registerUnaryCpuFloat() {
  // abs
  REG_ELEM_WISE_CPU_FLT(abs, unaryOpCpuImpl, OpCpuAbs);
  REG_ELEM_WISE_CPU_FLT(absOut, unaryOpOutCpuImpl, OpCpuAbs);
  REG_ELEM_WISE_CPU_FLT(absInplace, unaryOpInplaceCpuImpl, OpCpuAbs);

  // neg
  REG_ELEM_WISE_CPU_FLT(neg, unaryOpCpuImpl, OpCpuNeg);
  REG_ELEM_WISE_CPU_FLT(negOut, unaryOpOutCpuImpl, OpCpuNeg);
  REG_ELEM_WISE_CPU_FLT(negInplace, unaryOpInplaceCpuImpl, OpCpuNeg);

  // sign
  REG_ELEM_WISE_CPU_FLT(sign, unaryOpCpuImpl, OpCpuSign);
  REG_ELEM_WISE_CPU_FLT(signOut, unaryOpOutCpuImpl, OpCpuSign);
  REG_ELEM_WISE_CPU_FLT(signInplace, unaryOpInplaceCpuImpl, OpCpuSign);

  // sqrt
  REG_ELEM_WISE_CPU_FLT(sqrt, unaryOpCpuImpl, OpCpuSqrt);
  REG_ELEM_WISE_CPU_FLT(sqrtOut, unaryOpOutCpuImpl, OpCpuSqrt);
  REG_ELEM_WISE_CPU_FLT(sqrtInplace, unaryOpInplaceCpuImpl, OpCpuSqrt);

  // square
  REG_ELEM_WISE_CPU_FLT(square, unaryOpCpuImpl, OpCpuSquare);
  REG_ELEM_WISE_CPU_FLT(squareOut, unaryOpOutCpuImpl, OpCpuSquare);
  REG_ELEM_WISE_CPU_FLT(squareInplace, unaryOpInplaceCpuImpl, OpCpuSquare);

  // exp
  REG_ELEM_WISE_CPU_FLT(exp, unaryOpCpuImpl, OpCpuExp);
  REG_ELEM_WISE_CPU_FLT(expOut, unaryOpOutCpuImpl, OpCpuExp);
  REG_ELEM_WISE_CPU_FLT(expInplace, unaryOpInplaceCpuImpl, OpCpuExp);

  // log
  REG_ELEM_WISE_CPU_FLT(log, unaryOpCpuImpl, OpCpuLog);
  REG_ELEM_WISE_CPU_FLT(logOut, unaryOpOutCpuImpl, OpCpuLog);
  REG_ELEM_WISE_CPU_FLT(logInplace, unaryOpInplaceCpuImpl, OpCpuLog);

  // sin
  REG_ELEM_WISE_CPU_FLT(sin, unaryOpCpuImpl, OpCpuSin);
  REG_ELEM_WISE_CPU_FLT(sinOut, unaryOpOutCpuImpl, OpCpuSin);
  REG_ELEM_WISE_CPU_FLT(sinInplace, unaryOpInplaceCpuImpl, OpCpuSin);
  REG_ELEM_WISE_CPU_FLT(sinBackwardP1, unaryOpBackwardCpuImpl, OpCpuSinBackwardP1);

  // cos
  REG_ELEM_WISE_CPU_FLT(cos, unaryOpCpuImpl, OpCpuCos);
  REG_ELEM_WISE_CPU_FLT(cosOut, unaryOpOutCpuImpl, OpCpuCos);
  REG_ELEM_WISE_CPU_FLT(cosInplace, unaryOpInplaceCpuImpl, OpCpuCos);
  REG_ELEM_WISE_CPU_FLT(cosBackwardP1, unaryOpBackwardCpuImpl, OpCpuCosBackwardP1);

  // sigmoid
  REG_ELEM_WISE_CPU_FLT(sigmoid, unaryOpCpuImpl, OpCpuSigmoid);
  REG_ELEM_WISE_CPU_FLT(sigmoidOut, unaryOpOutCpuImpl, OpCpuSigmoid);
  REG_ELEM_WISE_CPU_FLT(sigmoidInplace, unaryOpInplaceCpuImpl, OpCpuSigmoid);

  // tanh
  REG_ELEM_WISE_CPU_FLT(tanh, unaryOpCpuImpl, OpCpuTanh);
  REG_ELEM_WISE_CPU_FLT(tanhOut, unaryOpOutCpuImpl, OpCpuTanh);
  REG_ELEM_WISE_CPU_FLT(tanhInplace, unaryOpInplaceCpuImpl, OpCpuTanh);

  // relu
  REG_ELEM_WISE_CPU_FLT(relu, unaryOpCpuImpl, OpCpuRelu);
  REG_ELEM_WISE_CPU_FLT(reluOut, unaryOpOutCpuImpl, OpCpuRelu);
  REG_ELEM_WISE_CPU_FLT(reluInplace, unaryOpInplaceCpuImpl, OpCpuRelu);

  // gelu
  REG_ELEM_WISE_CPU_FLT(gelu, unaryOpCpuImpl, OpCpuGelu);
  REG_ELEM_WISE_CPU_FLT(geluOut, unaryOpOutCpuImpl, OpCpuGelu);
  REG_ELEM_WISE_CPU_FLT(geluInplace, unaryOpInplaceCpuImpl, OpCpuGelu);

  // silu
  REG_ELEM_WISE_CPU_FLT(silu, unaryOpCpuImpl, OpCpuSilu);
  REG_ELEM_WISE_CPU_FLT(siluOut, unaryOpOutCpuImpl, OpCpuSilu);
  REG_ELEM_WISE_CPU_FLT(siluInplace, unaryOpInplaceCpuImpl, OpCpuSilu);
}

void registerBinaryCpuFloat() {
  // add
  REG_ELEM_WISE_CPU_FLT(add, binaryOpAlphaCpuImpl, OpCpuAdd);
  REG_ELEM_WISE_CPU_FLT(addOut, binaryOpAlphaOutCpuImpl, OpCpuAdd);
  REG_ELEM_WISE_CPU_FLT(addInplace, binaryOpAlphaInplaceCpuImpl, OpCpuAdd);

  // sub
  REG_ELEM_WISE_CPU_FLT(sub, binaryOpAlphaCpuImpl, OpCpuSub);
  REG_ELEM_WISE_CPU_FLT(subOut, binaryOpAlphaOutCpuImpl, OpCpuSub);
  REG_ELEM_WISE_CPU_FLT(subInplace, binaryOpAlphaInplaceCpuImpl, OpCpuSub);

  // mul
  REG_ELEM_WISE_CPU_FLT(mul, binaryOpCpuImpl, OpCpuMul);
  REG_ELEM_WISE_CPU_FLT(mulOut, binaryOpOutCpuImpl, OpCpuMul);
  REG_ELEM_WISE_CPU_FLT(mulInplace, binaryOpInplaceCpuImpl, OpCpuMul);

  // div
  REG_ELEM_WISE_CPU_FLT(div, binaryOpCpuImpl, OpCpuDiv);
  REG_ELEM_WISE_CPU_FLT(divOut, binaryOpOutCpuImpl, OpCpuDiv);
  REG_ELEM_WISE_CPU_FLT(divInplace, binaryOpInplaceCpuImpl, OpCpuDiv);
  REG_ELEM_WISE_CPU_FLT(divBackwardP2, binaryOpBackwardCpuImpl, OpCpuDivBackwardP2);

  // pow
  REG_ELEM_WISE_CPU_FLT(pow, binaryOpCpuImpl, OpCpuPow);
  REG_ELEM_WISE_CPU_FLT(powOut, binaryOpOutCpuImpl, OpCpuPow);
  REG_ELEM_WISE_CPU_FLT(powInplace, binaryOpInplaceCpuImpl, OpCpuPow);
  REG_ELEM_WISE_CPU_FLT(powBackwardP1, binaryOpBackwardCpuImpl, OpCpuPowBackwardP1);
  REG_ELEM_WISE_CPU_FLT(powBackwardP2, binaryOpBackwardCpuImpl, OpCpuPowBackwardP2);

  // maximum
  REG_ELEM_WISE_CPU_FLT(maximum, binaryOpCpuImpl, OpCpuMaximum);
  REG_ELEM_WISE_CPU_FLT(maximumOut, binaryOpOutCpuImpl, OpCpuMaximum);

  // minimum
  REG_ELEM_WISE_CPU_FLT(minimum, binaryOpCpuImpl, OpCpuMinimum);
  REG_ELEM_WISE_CPU_FLT(minimumOut, binaryOpOutCpuImpl, OpCpuMinimum);

  // equal
  REG_ELEM_WISE_CPU_FLT(eq, binaryOpCompareCpuImpl, OpCpuEq);
  REG_ELEM_WISE_CPU_FLT(eqOut, binaryOpCompareOutCpuImpl, OpCpuEq);

  // not equal
  REG_ELEM_WISE_CPU_FLT(ne, binaryOpCompareCpuImpl, OpCpuNe);
  REG_ELEM_WISE_CPU_FLT(neOut, binaryOpCompareOutCpuImpl, OpCpuNe);

  // less than
  REG_ELEM_WISE_CPU_FLT(lt, binaryOpCompareCpuImpl, OpCpuLt);
  REG_ELEM_WISE_CPU_FLT(ltOut, binaryOpCompareOutCpuImpl, OpCpuLt);

  // less equal
  REG_ELEM_WISE_CPU_FLT(le, binaryOpCompareCpuImpl, OpCpuLe);
  REG_ELEM_WISE_CPU_FLT(leOut, binaryOpCompareOutCpuImpl, OpCpuLe);

  // greater than
  REG_ELEM_WISE_CPU_FLT(gt, binaryOpCompareCpuImpl, OpCpuGt);
  REG_ELEM_WISE_CPU_FLT(gtOut, binaryOpCompareOutCpuImpl, OpCpuGt);

  // greater equal
  REG_ELEM_WISE_CPU_FLT(ge, binaryOpCompareCpuImpl, OpCpuGe);
  REG_ELEM_WISE_CPU_FLT(geOut, binaryOpCompareOutCpuImpl, OpCpuGe);

  // clampMin
  REG_ELEM_WISE_CPU_FLT(clampMin, binaryOpCpuImpl, OpCpuClampMin);
  REG_ELEM_WISE_CPU_FLT(clampMinOut, binaryOpOutCpuImpl, OpCpuClampMin);
  REG_ELEM_WISE_CPU_FLT(clampMinInplace, binaryOpInplaceCpuImpl, OpCpuClampMin);

  // clampMax
  REG_ELEM_WISE_CPU_FLT(clampMax, binaryOpCpuImpl, OpCpuClampMax);
  REG_ELEM_WISE_CPU_FLT(clampMaxOut, binaryOpOutCpuImpl, OpCpuClampMax);
  REG_ELEM_WISE_CPU_FLT(clampMaxInplace, binaryOpInplaceCpuImpl, OpCpuClampMax);
}

void registerBinaryCpuInt64() {
  // maximum
  REG_ELEM_WISE_CPU_I64(maximum, binaryOpCpuImpl, OpCpuMaximum);
  REG_ELEM_WISE_CPU_I64(maximumOut, binaryOpOutCpuImpl, OpCpuMaximum);

  // minimum
  REG_ELEM_WISE_CPU_I64(minimum, binaryOpCpuImpl, OpCpuMinimum);
  REG_ELEM_WISE_CPU_I64(minimumOut, binaryOpOutCpuImpl, OpCpuMinimum);

  // equal
  REG_ELEM_WISE_CPU_I64(eq, binaryOpCompareCpuImpl, OpCpuEq);
  REG_ELEM_WISE_CPU_I64(eqOut, binaryOpCompareOutCpuImpl, OpCpuEq);

  // not equal
  REG_ELEM_WISE_CPU_I64(ne, binaryOpCompareCpuImpl, OpCpuNe);
  REG_ELEM_WISE_CPU_I64(neOut, binaryOpCompareOutCpuImpl, OpCpuNe);

  // less than
  REG_ELEM_WISE_CPU_I64(lt, binaryOpCompareCpuImpl, OpCpuLt);
  REG_ELEM_WISE_CPU_I64(ltOut, binaryOpCompareOutCpuImpl, OpCpuLt);

  // less equal
  REG_ELEM_WISE_CPU_I64(le, binaryOpCompareCpuImpl, OpCpuLe);
  REG_ELEM_WISE_CPU_I64(leOut, binaryOpCompareOutCpuImpl, OpCpuLe);

  // greater than
  REG_ELEM_WISE_CPU_I64(gt, binaryOpCompareCpuImpl, OpCpuGt);
  REG_ELEM_WISE_CPU_I64(gtOut, binaryOpCompareOutCpuImpl, OpCpuGt);

  // greater equal
  REG_ELEM_WISE_CPU_I64(ge, binaryOpCompareCpuImpl, OpCpuGe);
  REG_ELEM_WISE_CPU_I64(geOut, binaryOpCompareOutCpuImpl, OpCpuGe);

  // clampMin
  REG_ELEM_WISE_CPU_I64(clampMin, binaryOpCpuImpl, OpCpuClampMin);
  REG_ELEM_WISE_CPU_I64(clampMinOut, binaryOpOutCpuImpl, OpCpuClampMin);
  REG_ELEM_WISE_CPU_I64(clampMinInplace, binaryOpInplaceCpuImpl, OpCpuClampMin);

  // clampMax
  REG_ELEM_WISE_CPU_I64(clampMax, binaryOpCpuImpl, OpCpuClampMax);
  REG_ELEM_WISE_CPU_I64(clampMaxOut, binaryOpOutCpuImpl, OpCpuClampMax);
  REG_ELEM_WISE_CPU_I64(clampMaxInplace, binaryOpInplaceCpuImpl, OpCpuClampMax);
}

void registerTernaryCpuFloat() {
  // clamp
  REG_ELEM_WISE_CPU_FLT(clamp, ternaryOpCpuImpl, OpCpuClamp);
  REG_ELEM_WISE_CPU_FLT(clampOut, ternaryOpOutCpuImpl, OpCpuClamp);
  REG_ELEM_WISE_CPU_FLT(clampInplace, ternaryOpInplaceCpuImpl, OpCpuClamp);

  // addcmul
  REG_ELEM_WISE_CPU_FLT_NO_OP(addcmul, addcmulOpCpuImpl);
  REG_ELEM_WISE_CPU_FLT_NO_OP(addcmulOut, addcmulOpOutCpuImpl);
  REG_ELEM_WISE_CPU_FLT_NO_OP(addcmulInplace, addcmulOpInplaceCpuImpl);
}

void registerElemWiseCpu() {
  registerUnaryCpuBool();
  registerUnaryCpuFloat();

  registerBinaryCpuFloat();
  registerBinaryCpuInt64();

  registerTernaryCpuFloat();
}

}  // namespace tinytorch::op