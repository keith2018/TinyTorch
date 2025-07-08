/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "OpElemWiseCpu.h"

namespace tinytorch::op {

#define REG_ELEM_WISE_CPU_F32(NAME, FUNC, OP) \
  REGISTER_OP_IMPL(NAME, CPU, Float32, &(FUNC<DTypeToType_t<DType::Float32>, OP>))

#define REG_ELEM_WISE_CPU_I64(NAME, FUNC, OP) \
  REGISTER_OP_IMPL(NAME, CPU, Int64, &(FUNC<DTypeToType_t<DType::Int64>, OP>))

void registerUnaryCpuFloat32() {
  // abs
  REG_ELEM_WISE_CPU_F32(abs, unaryOpCpuImpl, OpCpuAbs);
  REG_ELEM_WISE_CPU_F32(absOut, unaryOpOutCpuImpl, OpCpuAbs);
  REG_ELEM_WISE_CPU_F32(absInplace, unaryOpInplaceCpuImpl, OpCpuAbs);

  // neg
  REG_ELEM_WISE_CPU_F32(neg, unaryOpCpuImpl, OpCpuNeg);
  REG_ELEM_WISE_CPU_F32(negOut, unaryOpOutCpuImpl, OpCpuNeg);
  REG_ELEM_WISE_CPU_F32(negInplace, unaryOpInplaceCpuImpl, OpCpuNeg);

  // sign
  REG_ELEM_WISE_CPU_F32(sign, unaryOpCpuImpl, OpCpuSign);
  REG_ELEM_WISE_CPU_F32(signOut, unaryOpOutCpuImpl, OpCpuSign);
  REG_ELEM_WISE_CPU_F32(signInplace, unaryOpInplaceCpuImpl, OpCpuSign);

  // sqrt
  REG_ELEM_WISE_CPU_F32(sqrt, unaryOpCpuImpl, OpCpuSqrt);
  REG_ELEM_WISE_CPU_F32(sqrtOut, unaryOpOutCpuImpl, OpCpuSqrt);
  REG_ELEM_WISE_CPU_F32(sqrtInplace, unaryOpInplaceCpuImpl, OpCpuSqrt);

  // square
  REG_ELEM_WISE_CPU_F32(square, unaryOpCpuImpl, OpCpuSquare);
  REG_ELEM_WISE_CPU_F32(squareOut, unaryOpOutCpuImpl, OpCpuSquare);
  REG_ELEM_WISE_CPU_F32(squareInplace, unaryOpInplaceCpuImpl, OpCpuSquare);

  // exp
  REG_ELEM_WISE_CPU_F32(exp, unaryOpCpuImpl, OpCpuExp);
  REG_ELEM_WISE_CPU_F32(expOut, unaryOpOutCpuImpl, OpCpuExp);
  REG_ELEM_WISE_CPU_F32(expInplace, unaryOpInplaceCpuImpl, OpCpuExp);

  // log
  REG_ELEM_WISE_CPU_F32(log, unaryOpCpuImpl, OpCpuLog);
  REG_ELEM_WISE_CPU_F32(logOut, unaryOpOutCpuImpl, OpCpuLog);
  REG_ELEM_WISE_CPU_F32(logInplace, unaryOpInplaceCpuImpl, OpCpuLog);

  // sin
  REG_ELEM_WISE_CPU_F32(sin, unaryOpCpuImpl, OpCpuSin);
  REG_ELEM_WISE_CPU_F32(sinOut, unaryOpOutCpuImpl, OpCpuSin);
  REG_ELEM_WISE_CPU_F32(sinInplace, unaryOpInplaceCpuImpl, OpCpuSin);
  REG_ELEM_WISE_CPU_F32(sinBackwardP1, unaryOpBackwardCpuImpl, OpCpuSinBackwardP1);

  // cos
  REG_ELEM_WISE_CPU_F32(cos, unaryOpCpuImpl, OpCpuCos);
  REG_ELEM_WISE_CPU_F32(cosOut, unaryOpOutCpuImpl, OpCpuCos);
  REG_ELEM_WISE_CPU_F32(cosInplace, unaryOpInplaceCpuImpl, OpCpuCos);
  REG_ELEM_WISE_CPU_F32(cosBackwardP1, unaryOpBackwardCpuImpl, OpCpuCosBackwardP1);

  // sigmoid
  REG_ELEM_WISE_CPU_F32(sigmoid, unaryOpCpuImpl, OpCpuSigmoid);
  REG_ELEM_WISE_CPU_F32(sigmoidOut, unaryOpOutCpuImpl, OpCpuSigmoid);
  REG_ELEM_WISE_CPU_F32(sigmoidInplace, unaryOpInplaceCpuImpl, OpCpuSigmoid);

  // tanh
  REG_ELEM_WISE_CPU_F32(tanh, unaryOpCpuImpl, OpCpuTanh);
  REG_ELEM_WISE_CPU_F32(tanhOut, unaryOpOutCpuImpl, OpCpuTanh);
  REG_ELEM_WISE_CPU_F32(tanhInplace, unaryOpInplaceCpuImpl, OpCpuTanh);

  // relu
  REG_ELEM_WISE_CPU_F32(relu, unaryOpCpuImpl, OpCpuRelu);
  REG_ELEM_WISE_CPU_F32(reluOut, unaryOpOutCpuImpl, OpCpuRelu);
  REG_ELEM_WISE_CPU_F32(reluInplace, unaryOpInplaceCpuImpl, OpCpuRelu);

  // gelu
  REG_ELEM_WISE_CPU_F32(gelu, unaryOpCpuImpl, OpCpuGelu);
  REG_ELEM_WISE_CPU_F32(geluOut, unaryOpOutCpuImpl, OpCpuGelu);
  REG_ELEM_WISE_CPU_F32(geluInplace, unaryOpInplaceCpuImpl, OpCpuGelu);
}

void registerBinaryCpuFloat32() {
  // add
  REG_ELEM_WISE_CPU_F32(add, binaryOpAlphaCpuImpl, OpCpuAdd);
  REG_ELEM_WISE_CPU_F32(addOut, binaryOpAlphaOutCpuImpl, OpCpuAdd);
  REG_ELEM_WISE_CPU_F32(addInplace, binaryOpAlphaInplaceCpuImpl, OpCpuAdd);

  // sub
  REG_ELEM_WISE_CPU_F32(sub, binaryOpAlphaCpuImpl, OpCpuSub);
  REG_ELEM_WISE_CPU_F32(subOut, binaryOpAlphaOutCpuImpl, OpCpuSub);
  REG_ELEM_WISE_CPU_F32(subInplace, binaryOpAlphaInplaceCpuImpl, OpCpuSub);

  // mul
  REG_ELEM_WISE_CPU_F32(mul, binaryOpCpuImpl, OpCpuMul);
  REG_ELEM_WISE_CPU_F32(mulOut, binaryOpOutCpuImpl, OpCpuMul);
  REG_ELEM_WISE_CPU_F32(mulInplace, binaryOpInplaceCpuImpl, OpCpuMul);

  // div
  REG_ELEM_WISE_CPU_F32(div, binaryOpCpuImpl, OpCpuDiv);
  REG_ELEM_WISE_CPU_F32(divOut, binaryOpOutCpuImpl, OpCpuDiv);
  REG_ELEM_WISE_CPU_F32(divInplace, binaryOpInplaceCpuImpl, OpCpuDiv);
  REG_ELEM_WISE_CPU_F32(divBackwardP2, binaryOpBackwardCpuImpl, OpCpuDivBackwardP2);

  // pow
  REG_ELEM_WISE_CPU_F32(pow, binaryOpCpuImpl, OpCpuPow);
  REG_ELEM_WISE_CPU_F32(powOut, binaryOpOutCpuImpl, OpCpuPow);
  REG_ELEM_WISE_CPU_F32(powInplace, binaryOpInplaceCpuImpl, OpCpuPow);
  REG_ELEM_WISE_CPU_F32(powBackwardP1, binaryOpBackwardCpuImpl, OpCpuPowBackwardP1);
  REG_ELEM_WISE_CPU_F32(powBackwardP2, binaryOpBackwardCpuImpl, OpCpuPowBackwardP2);

  // maximum
  REG_ELEM_WISE_CPU_F32(maximum, binaryOpCpuImpl, OpCpuMaximum);
  REG_ELEM_WISE_CPU_F32(maximumOut, binaryOpOutCpuImpl, OpCpuMaximum);

  // minimum
  REG_ELEM_WISE_CPU_F32(minimum, binaryOpCpuImpl, OpCpuMinimum);
  REG_ELEM_WISE_CPU_F32(minimumOut, binaryOpOutCpuImpl, OpCpuMinimum);

  // equal
  REG_ELEM_WISE_CPU_F32(eq, binaryOpCompareCpuImpl, OpCpuEq);
  REG_ELEM_WISE_CPU_F32(eqOut, binaryOpCompareOutCpuImpl, OpCpuEq);

  // not equal
  REG_ELEM_WISE_CPU_F32(ne, binaryOpCompareCpuImpl, OpCpuNe);
  REG_ELEM_WISE_CPU_F32(neOut, binaryOpCompareOutCpuImpl, OpCpuNe);

  // less than
  REG_ELEM_WISE_CPU_F32(lt, binaryOpCompareCpuImpl, OpCpuLt);
  REG_ELEM_WISE_CPU_F32(ltOut, binaryOpCompareOutCpuImpl, OpCpuLt);

  // less equal
  REG_ELEM_WISE_CPU_F32(le, binaryOpCompareCpuImpl, OpCpuLe);
  REG_ELEM_WISE_CPU_F32(leOut, binaryOpCompareOutCpuImpl, OpCpuLe);

  // greater than
  REG_ELEM_WISE_CPU_F32(gt, binaryOpCompareCpuImpl, OpCpuGt);
  REG_ELEM_WISE_CPU_F32(gtOut, binaryOpCompareOutCpuImpl, OpCpuGt);

  // greater equal
  REG_ELEM_WISE_CPU_F32(ge, binaryOpCompareCpuImpl, OpCpuGe);
  REG_ELEM_WISE_CPU_F32(geOut, binaryOpCompareOutCpuImpl, OpCpuGe);

  // clampMin
  REG_ELEM_WISE_CPU_F32(clampMin, binaryOpCpuImpl, OpCpuClampMin);
  REG_ELEM_WISE_CPU_F32(clampMinOut, binaryOpOutCpuImpl, OpCpuClampMin);
  REG_ELEM_WISE_CPU_F32(clampMinInplace, binaryOpInplaceCpuImpl, OpCpuClampMin);

  // clampMax
  REG_ELEM_WISE_CPU_F32(clampMax, binaryOpCpuImpl, OpCpuClampMax);
  REG_ELEM_WISE_CPU_F32(clampMaxOut, binaryOpOutCpuImpl, OpCpuClampMax);
  REG_ELEM_WISE_CPU_F32(clampMaxInplace, binaryOpInplaceCpuImpl, OpCpuClampMax);
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

void registerTernaryCpuFloat32() {
  // clamp
  REG_ELEM_WISE_CPU_F32(clamp, ternaryOpCpuImpl, OpCpuClamp);
  REG_ELEM_WISE_CPU_F32(clampOut, ternaryOpOutCpuImpl, OpCpuClamp);
  REG_ELEM_WISE_CPU_F32(clampInplace, ternaryOpInplaceCpuImpl, OpCpuClamp);
}

void registerElemWiseCpu() {
  registerUnaryCpuFloat32();

  registerBinaryCpuFloat32();
  registerBinaryCpuInt64();

  registerTernaryCpuFloat32();
}

}  // namespace tinytorch::op