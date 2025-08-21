/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "OpElemWiseCuda.cuh"

namespace tinytorch::op {

#define REG_ELEM_WISE_CUDA_BOOL(NAME, FUNC, OP) \
  REGISTER_OP_IMPL(NAME, CUDA, Bool, &(FUNC<DTypeToType_t<DType::Bool>, OP>))

#define REG_ELEM_WISE_CUDA_FLT(NAME, FUNC, OP)                                      \
  REGISTER_OP_IMPL(NAME, CUDA, Float32, &(FUNC<DTypeToType_t<DType::Float32>, OP>)) \
  REGISTER_OP_IMPL(NAME, CUDA, Float16, &(FUNC<DTypeToType_t<DType::Float16>, OP>)) \
  REGISTER_OP_IMPL(NAME, CUDA, BFloat16, &(FUNC<DTypeToType_t<DType::BFloat16>, OP>))

#define REG_ELEM_WISE_CUDA_FLT_NO_OP(NAME, FUNC)                                \
  REGISTER_OP_IMPL(NAME, CUDA, Float32, &(FUNC<DTypeToType_t<DType::Float32>>)) \
  REGISTER_OP_IMPL(NAME, CUDA, Float16, &(FUNC<DTypeToType_t<DType::Float16>>)) \
  REGISTER_OP_IMPL(NAME, CUDA, BFloat16, &(FUNC<DTypeToType_t<DType::BFloat16>>))

#define REG_ELEM_WISE_CUDA_I64(NAME, FUNC, OP) \
  REGISTER_OP_IMPL(NAME, CUDA, Int64, &(FUNC<DTypeToType_t<DType::Int64>, OP>))

void registerUnaryCudaBool() {
  // logicNot
  REG_ELEM_WISE_CUDA_BOOL(logicNot, unaryOpCudaImpl, OpCudaLogicNot);
  REG_ELEM_WISE_CUDA_BOOL(logicNotOut, unaryOpOutCudaImpl, OpCudaLogicNot);
  REG_ELEM_WISE_CUDA_BOOL(logicNotInplace, unaryOpInplaceCudaImpl, OpCudaLogicNot);

  // logicAnd
  REG_ELEM_WISE_CUDA_BOOL(logicAnd, binaryOpCudaImpl, OpCudaLogicAnd);
  REG_ELEM_WISE_CUDA_BOOL(logicAndOut, binaryOpOutCudaImpl, OpCudaLogicAnd);
  REG_ELEM_WISE_CUDA_BOOL(logicAndInplace, binaryOpInplaceCudaImpl, OpCudaLogicAnd);

  // logicOr
  REG_ELEM_WISE_CUDA_BOOL(logicOr, binaryOpCudaImpl, OpCudaLogicOr);
  REG_ELEM_WISE_CUDA_BOOL(logicOrOut, binaryOpOutCudaImpl, OpCudaLogicOr);
  REG_ELEM_WISE_CUDA_BOOL(logicOrInplace, binaryOpInplaceCudaImpl, OpCudaLogicOr);
}

void registerUnaryCudaFloat() {
  // abs
  REG_ELEM_WISE_CUDA_FLT(abs, unaryOpCudaImpl, OpCudaAbs);
  REG_ELEM_WISE_CUDA_FLT(absOut, unaryOpOutCudaImpl, OpCudaAbs);
  REG_ELEM_WISE_CUDA_FLT(absInplace, unaryOpInplaceCudaImpl, OpCudaAbs);

  // neg
  REG_ELEM_WISE_CUDA_FLT(neg, unaryOpCudaImpl, OpCudaNeg);
  REG_ELEM_WISE_CUDA_FLT(negOut, unaryOpOutCudaImpl, OpCudaNeg);
  REG_ELEM_WISE_CUDA_FLT(negInplace, unaryOpInplaceCudaImpl, OpCudaNeg);

  // sign
  REG_ELEM_WISE_CUDA_FLT(sign, unaryOpCudaImpl, OpCudaSign);
  REG_ELEM_WISE_CUDA_FLT(signOut, unaryOpOutCudaImpl, OpCudaSign);
  REG_ELEM_WISE_CUDA_FLT(signInplace, unaryOpInplaceCudaImpl, OpCudaSign);

  // sqrt
  REG_ELEM_WISE_CUDA_FLT(sqrt, unaryOpCudaImpl, OpCudaSqrt);
  REG_ELEM_WISE_CUDA_FLT(sqrtOut, unaryOpOutCudaImpl, OpCudaSqrt);
  REG_ELEM_WISE_CUDA_FLT(sqrtInplace, unaryOpInplaceCudaImpl, OpCudaSqrt);

  // square
  REG_ELEM_WISE_CUDA_FLT(square, unaryOpCudaImpl, OpCudaSquare);
  REG_ELEM_WISE_CUDA_FLT(squareOut, unaryOpOutCudaImpl, OpCudaSquare);
  REG_ELEM_WISE_CUDA_FLT(squareInplace, unaryOpInplaceCudaImpl, OpCudaSquare);

  // exp
  REG_ELEM_WISE_CUDA_FLT(exp, unaryOpCudaImpl, OpCudaExp);
  REG_ELEM_WISE_CUDA_FLT(expOut, unaryOpOutCudaImpl, OpCudaExp);
  REG_ELEM_WISE_CUDA_FLT(expInplace, unaryOpInplaceCudaImpl, OpCudaExp);

  // log
  REG_ELEM_WISE_CUDA_FLT(log, unaryOpCudaImpl, OpCudaLog);
  REG_ELEM_WISE_CUDA_FLT(logOut, unaryOpOutCudaImpl, OpCudaLog);
  REG_ELEM_WISE_CUDA_FLT(logInplace, unaryOpInplaceCudaImpl, OpCudaLog);

  // sin
  REG_ELEM_WISE_CUDA_FLT(sin, unaryOpCudaImpl, OpCudaSin);
  REG_ELEM_WISE_CUDA_FLT(sinOut, unaryOpOutCudaImpl, OpCudaSin);
  REG_ELEM_WISE_CUDA_FLT(sinInplace, unaryOpInplaceCudaImpl, OpCudaSin);
  REG_ELEM_WISE_CUDA_FLT(sinBackwardP1, unaryOpBackwardCudaImpl, OpCudaSinBackwardP1);

  // cos
  REG_ELEM_WISE_CUDA_FLT(cos, unaryOpCudaImpl, OpCudaCos);
  REG_ELEM_WISE_CUDA_FLT(cosOut, unaryOpOutCudaImpl, OpCudaCos);
  REG_ELEM_WISE_CUDA_FLT(cosInplace, unaryOpInplaceCudaImpl, OpCudaCos);
  REG_ELEM_WISE_CUDA_FLT(cosBackwardP1, unaryOpBackwardCudaImpl, OpCudaCosBackwardP1);

  // sigmoid
  REG_ELEM_WISE_CUDA_FLT(sigmoid, unaryOpCudaImpl, OpCudaSigmoid);
  REG_ELEM_WISE_CUDA_FLT(sigmoidOut, unaryOpOutCudaImpl, OpCudaSigmoid);
  REG_ELEM_WISE_CUDA_FLT(sigmoidInplace, unaryOpInplaceCudaImpl, OpCudaSigmoid);

  // tanh
  REG_ELEM_WISE_CUDA_FLT(tanh, unaryOpCudaImpl, OpCudaTanh);
  REG_ELEM_WISE_CUDA_FLT(tanhOut, unaryOpOutCudaImpl, OpCudaTanh);
  REG_ELEM_WISE_CUDA_FLT(tanhInplace, unaryOpInplaceCudaImpl, OpCudaTanh);

  // relu
  REG_ELEM_WISE_CUDA_FLT(relu, unaryOpCudaImpl, OpCudaRelu);
  REG_ELEM_WISE_CUDA_FLT(reluOut, unaryOpOutCudaImpl, OpCudaRelu);
  REG_ELEM_WISE_CUDA_FLT(reluInplace, unaryOpInplaceCudaImpl, OpCudaRelu);

  // gelu
  REG_ELEM_WISE_CUDA_FLT(gelu, unaryOpCudaImpl, OpCudaGelu);
  REG_ELEM_WISE_CUDA_FLT(geluOut, unaryOpOutCudaImpl, OpCudaGelu);
  REG_ELEM_WISE_CUDA_FLT(geluInplace, unaryOpInplaceCudaImpl, OpCudaGelu);

  // silu
  REG_ELEM_WISE_CUDA_FLT(silu, unaryOpCudaImpl, OpCudaSilu);
  REG_ELEM_WISE_CUDA_FLT(siluOut, unaryOpOutCudaImpl, OpCudaSilu);
  REG_ELEM_WISE_CUDA_FLT(siluInplace, unaryOpInplaceCudaImpl, OpCudaSilu);
}

void registerBinaryCudaFloat() {
  // add
  REG_ELEM_WISE_CUDA_FLT(add, binaryOpAlphaCudaImpl, OpCudaAdd);
  REG_ELEM_WISE_CUDA_FLT(addOut, binaryOpAlphaOutCudaImpl, OpCudaAdd);
  REG_ELEM_WISE_CUDA_FLT(addInplace, binaryOpAlphaInplaceCudaImpl, OpCudaAdd);

  // sub
  REG_ELEM_WISE_CUDA_FLT(sub, binaryOpAlphaCudaImpl, OpCudaSub);
  REG_ELEM_WISE_CUDA_FLT(subOut, binaryOpAlphaOutCudaImpl, OpCudaSub);
  REG_ELEM_WISE_CUDA_FLT(subInplace, binaryOpAlphaInplaceCudaImpl, OpCudaSub);

  // mul
  REG_ELEM_WISE_CUDA_FLT(mul, binaryOpCudaImpl, OpCudaMul);
  REG_ELEM_WISE_CUDA_FLT(mulOut, binaryOpOutCudaImpl, OpCudaMul);
  REG_ELEM_WISE_CUDA_FLT(mulInplace, binaryOpInplaceCudaImpl, OpCudaMul);

  // div
  REG_ELEM_WISE_CUDA_FLT(div, binaryOpCudaImpl, OpCudaDiv);
  REG_ELEM_WISE_CUDA_FLT(divOut, binaryOpOutCudaImpl, OpCudaDiv);
  REG_ELEM_WISE_CUDA_FLT(divInplace, binaryOpInplaceCudaImpl, OpCudaDiv);
  REG_ELEM_WISE_CUDA_FLT(divBackwardP2, binaryOpBackwardCudaImpl, OpCudaDivBackwardP2);

  // pow
  REG_ELEM_WISE_CUDA_FLT(pow, binaryOpCudaImpl, OpCudaPow);
  REG_ELEM_WISE_CUDA_FLT(powOut, binaryOpOutCudaImpl, OpCudaPow);
  REG_ELEM_WISE_CUDA_FLT(powInplace, binaryOpInplaceCudaImpl, OpCudaPow);
  REG_ELEM_WISE_CUDA_FLT(powBackwardP1, binaryOpBackwardCudaImpl, OpCudaPowBackwardP1);
  REG_ELEM_WISE_CUDA_FLT(powBackwardP2, binaryOpBackwardCudaImpl, OpCudaPowBackwardP2);

  // maximum
  REG_ELEM_WISE_CUDA_FLT(maximum, binaryOpCudaImpl, OpCudaMaximum);
  REG_ELEM_WISE_CUDA_FLT(maximumOut, binaryOpOutCudaImpl, OpCudaMaximum);

  // minimum
  REG_ELEM_WISE_CUDA_FLT(minimum, binaryOpCudaImpl, OpCudaMinimum);
  REG_ELEM_WISE_CUDA_FLT(minimumOut, binaryOpOutCudaImpl, OpCudaMinimum);

  // equal
  REG_ELEM_WISE_CUDA_FLT(eq, binaryOpCompareCudaImpl, OpCudaEq);
  REG_ELEM_WISE_CUDA_FLT(eqOut, binaryOpCompareOutCudaImpl, OpCudaEq);

  // not equal
  REG_ELEM_WISE_CUDA_FLT(ne, binaryOpCompareCudaImpl, OpCudaNe);
  REG_ELEM_WISE_CUDA_FLT(neOut, binaryOpCompareOutCudaImpl, OpCudaNe);

  // less than
  REG_ELEM_WISE_CUDA_FLT(lt, binaryOpCompareCudaImpl, OpCudaLt);
  REG_ELEM_WISE_CUDA_FLT(ltOut, binaryOpCompareOutCudaImpl, OpCudaLt);

  // less equal
  REG_ELEM_WISE_CUDA_FLT(le, binaryOpCompareCudaImpl, OpCudaLe);
  REG_ELEM_WISE_CUDA_FLT(leOut, binaryOpCompareOutCudaImpl, OpCudaLe);

  // greater than
  REG_ELEM_WISE_CUDA_FLT(gt, binaryOpCompareCudaImpl, OpCudaGt);
  REG_ELEM_WISE_CUDA_FLT(gtOut, binaryOpCompareOutCudaImpl, OpCudaGt);

  // greater equal
  REG_ELEM_WISE_CUDA_FLT(ge, binaryOpCompareCudaImpl, OpCudaGe);
  REG_ELEM_WISE_CUDA_FLT(geOut, binaryOpCompareOutCudaImpl, OpCudaGe);

  // clampMin
  REG_ELEM_WISE_CUDA_FLT(clampMin, binaryOpCudaImpl, OpCudaClampMin);
  REG_ELEM_WISE_CUDA_FLT(clampMinOut, binaryOpOutCudaImpl, OpCudaClampMin);
  REG_ELEM_WISE_CUDA_FLT(clampMinInplace, binaryOpInplaceCudaImpl, OpCudaClampMin);

  // clampMax
  REG_ELEM_WISE_CUDA_FLT(clampMax, binaryOpCudaImpl, OpCudaClampMax);
  REG_ELEM_WISE_CUDA_FLT(clampMaxOut, binaryOpOutCudaImpl, OpCudaClampMax);
  REG_ELEM_WISE_CUDA_FLT(clampMaxInplace, binaryOpInplaceCudaImpl, OpCudaClampMax);
}

void registerBinaryCudaInt64() {
  // maximum
  REG_ELEM_WISE_CUDA_I64(maximum, binaryOpCudaImpl, OpCudaMaximum);
  REG_ELEM_WISE_CUDA_I64(maximumOut, binaryOpOutCudaImpl, OpCudaMaximum);

  // minimum
  REG_ELEM_WISE_CUDA_I64(minimum, binaryOpCudaImpl, OpCudaMinimum);
  REG_ELEM_WISE_CUDA_I64(minimumOut, binaryOpOutCudaImpl, OpCudaMinimum);

  // equal
  REG_ELEM_WISE_CUDA_I64(eq, binaryOpCompareCudaImpl, OpCudaEq);
  REG_ELEM_WISE_CUDA_I64(eqOut, binaryOpCompareOutCudaImpl, OpCudaEq);

  // not equal
  REG_ELEM_WISE_CUDA_I64(ne, binaryOpCompareCudaImpl, OpCudaNe);
  REG_ELEM_WISE_CUDA_I64(neOut, binaryOpCompareOutCudaImpl, OpCudaNe);

  // less than
  REG_ELEM_WISE_CUDA_I64(lt, binaryOpCompareCudaImpl, OpCudaLt);
  REG_ELEM_WISE_CUDA_I64(ltOut, binaryOpCompareOutCudaImpl, OpCudaLt);

  // less equal
  REG_ELEM_WISE_CUDA_I64(le, binaryOpCompareCudaImpl, OpCudaLe);
  REG_ELEM_WISE_CUDA_I64(leOut, binaryOpCompareOutCudaImpl, OpCudaLe);

  // greater than
  REG_ELEM_WISE_CUDA_I64(gt, binaryOpCompareCudaImpl, OpCudaGt);
  REG_ELEM_WISE_CUDA_I64(gtOut, binaryOpCompareOutCudaImpl, OpCudaGt);

  // greater equal
  REG_ELEM_WISE_CUDA_I64(ge, binaryOpCompareCudaImpl, OpCudaGe);
  REG_ELEM_WISE_CUDA_I64(geOut, binaryOpCompareOutCudaImpl, OpCudaGe);

  // clampMin
  REG_ELEM_WISE_CUDA_I64(clampMin, binaryOpCudaImpl, OpCudaClampMin);
  REG_ELEM_WISE_CUDA_I64(clampMinOut, binaryOpOutCudaImpl, OpCudaClampMin);
  REG_ELEM_WISE_CUDA_I64(clampMinInplace, binaryOpInplaceCudaImpl, OpCudaClampMin);

  // clampMax
  REG_ELEM_WISE_CUDA_I64(clampMax, binaryOpCudaImpl, OpCudaClampMax);
  REG_ELEM_WISE_CUDA_I64(clampMaxOut, binaryOpOutCudaImpl, OpCudaClampMax);
  REG_ELEM_WISE_CUDA_I64(clampMaxInplace, binaryOpInplaceCudaImpl, OpCudaClampMax);
}

void registerTernaryCudaFloat() {
  // clamp
  REG_ELEM_WISE_CUDA_FLT(clamp, ternaryOpCudaImpl, OpCudaClamp);
  REG_ELEM_WISE_CUDA_FLT(clampOut, ternaryOpOutCudaImpl, OpCudaClamp);
  REG_ELEM_WISE_CUDA_FLT(clampInplace, ternaryOpInplaceCudaImpl, OpCudaClamp);

  // addcmul
  REG_ELEM_WISE_CUDA_FLT_NO_OP(addcmul, addcmulOpCudaImpl);
  REG_ELEM_WISE_CUDA_FLT_NO_OP(addcmulOut, addcmulOpOutCudaImpl);
  REG_ELEM_WISE_CUDA_FLT_NO_OP(addcmulInplace, addcmulOpInplaceCudaImpl);
}

void registerElemWiseCuda() {
  registerUnaryCudaBool();
  registerUnaryCudaFloat();

  registerBinaryCudaFloat();
  registerBinaryCudaInt64();

  registerTernaryCudaFloat();
}

}  // namespace tinytorch::op