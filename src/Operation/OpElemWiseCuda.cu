/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "OpElemWiseCuda.cuh"

namespace tinytorch::op {

#define REG_ELEM_WISE_CUDA_F32(NAME, FUNC, OP) \
  REGISTER_OP_IMPL(NAME, CUDA, Float32, &(FUNC<DTypeToType_t<DType::Float32>, OP>))

#define REG_ELEM_WISE_CUDA_I64(NAME, FUNC, OP) \
  REGISTER_OP_IMPL(NAME, CUDA, Int64, &(FUNC<DTypeToType_t<DType::Int64>, OP>))

void registerUnaryCudaFloat32() {
  // abs
  REG_ELEM_WISE_CUDA_F32(abs, unaryOpCudaImpl, OpCudaAbs);
  REG_ELEM_WISE_CUDA_F32(absOut, unaryOpOutCudaImpl, OpCudaAbs);
  REG_ELEM_WISE_CUDA_F32(absInplace, unaryOpInplaceCudaImpl, OpCudaAbs);

  // neg
  REG_ELEM_WISE_CUDA_F32(neg, unaryOpCudaImpl, OpCudaNeg);
  REG_ELEM_WISE_CUDA_F32(negOut, unaryOpOutCudaImpl, OpCudaNeg);
  REG_ELEM_WISE_CUDA_F32(negInplace, unaryOpInplaceCudaImpl, OpCudaNeg);

  // sign
  REG_ELEM_WISE_CUDA_F32(sign, unaryOpCudaImpl, OpCudaSign);
  REG_ELEM_WISE_CUDA_F32(signOut, unaryOpOutCudaImpl, OpCudaSign);
  REG_ELEM_WISE_CUDA_F32(signInplace, unaryOpInplaceCudaImpl, OpCudaSign);

  // sqrt
  REG_ELEM_WISE_CUDA_F32(sqrt, unaryOpCudaImpl, OpCudaSqrt);
  REG_ELEM_WISE_CUDA_F32(sqrtOut, unaryOpOutCudaImpl, OpCudaSqrt);
  REG_ELEM_WISE_CUDA_F32(sqrtInplace, unaryOpInplaceCudaImpl, OpCudaSqrt);

  // square
  REG_ELEM_WISE_CUDA_F32(square, unaryOpCudaImpl, OpCudaSquare);
  REG_ELEM_WISE_CUDA_F32(squareOut, unaryOpOutCudaImpl, OpCudaSquare);
  REG_ELEM_WISE_CUDA_F32(squareInplace, unaryOpInplaceCudaImpl, OpCudaSquare);

  // exp
  REG_ELEM_WISE_CUDA_F32(exp, unaryOpCudaImpl, OpCudaExp);
  REG_ELEM_WISE_CUDA_F32(expOut, unaryOpOutCudaImpl, OpCudaExp);
  REG_ELEM_WISE_CUDA_F32(expInplace, unaryOpInplaceCudaImpl, OpCudaExp);

  // log
  REG_ELEM_WISE_CUDA_F32(log, unaryOpCudaImpl, OpCudaLog);
  REG_ELEM_WISE_CUDA_F32(logOut, unaryOpOutCudaImpl, OpCudaLog);
  REG_ELEM_WISE_CUDA_F32(logInplace, unaryOpInplaceCudaImpl, OpCudaLog);

  // sin
  REG_ELEM_WISE_CUDA_F32(sin, unaryOpCudaImpl, OpCudaSin);
  REG_ELEM_WISE_CUDA_F32(sinOut, unaryOpOutCudaImpl, OpCudaSin);
  REG_ELEM_WISE_CUDA_F32(sinInplace, unaryOpInplaceCudaImpl, OpCudaSin);
  REG_ELEM_WISE_CUDA_F32(sinBackwardP1, unaryOpBackwardCudaImpl, OpCudaSinBackwardP1);

  // cos
  REG_ELEM_WISE_CUDA_F32(cos, unaryOpCudaImpl, OpCudaCos);
  REG_ELEM_WISE_CUDA_F32(cosOut, unaryOpOutCudaImpl, OpCudaCos);
  REG_ELEM_WISE_CUDA_F32(cosInplace, unaryOpInplaceCudaImpl, OpCudaCos);
  REG_ELEM_WISE_CUDA_F32(cosBackwardP1, unaryOpBackwardCudaImpl, OpCudaCosBackwardP1);

  // sigmoid
  REG_ELEM_WISE_CUDA_F32(sigmoid, unaryOpCudaImpl, OpCudaSigmoid);
  REG_ELEM_WISE_CUDA_F32(sigmoidOut, unaryOpOutCudaImpl, OpCudaSigmoid);
  REG_ELEM_WISE_CUDA_F32(sigmoidInplace, unaryOpInplaceCudaImpl, OpCudaSigmoid);

  // tanh
  REG_ELEM_WISE_CUDA_F32(tanh, unaryOpCudaImpl, OpCudaTanh);
  REG_ELEM_WISE_CUDA_F32(tanhOut, unaryOpOutCudaImpl, OpCudaTanh);
  REG_ELEM_WISE_CUDA_F32(tanhInplace, unaryOpInplaceCudaImpl, OpCudaTanh);

  // relu
  REG_ELEM_WISE_CUDA_F32(relu, unaryOpCudaImpl, OpCudaRelu);
  REG_ELEM_WISE_CUDA_F32(reluOut, unaryOpOutCudaImpl, OpCudaRelu);
  REG_ELEM_WISE_CUDA_F32(reluInplace, unaryOpInplaceCudaImpl, OpCudaRelu);

  // gelu
  REG_ELEM_WISE_CUDA_F32(gelu, unaryOpCudaImpl, OpCudaGelu);
  REG_ELEM_WISE_CUDA_F32(geluOut, unaryOpOutCudaImpl, OpCudaGelu);
  REG_ELEM_WISE_CUDA_F32(geluInplace, unaryOpInplaceCudaImpl, OpCudaGelu);
}

void registerBinaryCudaFloat32() {
  // add
  REG_ELEM_WISE_CUDA_F32(add, binaryOpAlphaCudaImpl, OpCudaAdd);
  REG_ELEM_WISE_CUDA_F32(addOut, binaryOpAlphaOutCudaImpl, OpCudaAdd);
  REG_ELEM_WISE_CUDA_F32(addInplace, binaryOpAlphaInplaceCudaImpl, OpCudaAdd);

  // sub
  REG_ELEM_WISE_CUDA_F32(sub, binaryOpAlphaCudaImpl, OpCudaSub);
  REG_ELEM_WISE_CUDA_F32(subOut, binaryOpAlphaOutCudaImpl, OpCudaSub);
  REG_ELEM_WISE_CUDA_F32(subInplace, binaryOpAlphaInplaceCudaImpl, OpCudaSub);

  // mul
  REG_ELEM_WISE_CUDA_F32(mul, binaryOpCudaImpl, OpCudaMul);
  REG_ELEM_WISE_CUDA_F32(mulOut, binaryOpOutCudaImpl, OpCudaMul);
  REG_ELEM_WISE_CUDA_F32(mulInplace, binaryOpInplaceCudaImpl, OpCudaMul);

  // div
  REG_ELEM_WISE_CUDA_F32(div, binaryOpCudaImpl, OpCudaDiv);
  REG_ELEM_WISE_CUDA_F32(divOut, binaryOpOutCudaImpl, OpCudaDiv);
  REG_ELEM_WISE_CUDA_F32(divInplace, binaryOpInplaceCudaImpl, OpCudaDiv);
  REG_ELEM_WISE_CUDA_F32(divBackwardP2, binaryOpBackwardCudaImpl, OpCudaDivBackwardP2);

  // pow
  REG_ELEM_WISE_CUDA_F32(pow, binaryOpCudaImpl, OpCudaPow);
  REG_ELEM_WISE_CUDA_F32(powOut, binaryOpOutCudaImpl, OpCudaPow);
  REG_ELEM_WISE_CUDA_F32(powInplace, binaryOpInplaceCudaImpl, OpCudaPow);
  REG_ELEM_WISE_CUDA_F32(powBackwardP1, binaryOpBackwardCudaImpl, OpCudaPowBackwardP1);
  REG_ELEM_WISE_CUDA_F32(powBackwardP2, binaryOpBackwardCudaImpl, OpCudaPowBackwardP2);

  // maximum
  REG_ELEM_WISE_CUDA_F32(maximum, binaryOpCudaImpl, OpCudaMaximum);
  REG_ELEM_WISE_CUDA_F32(maximumOut, binaryOpOutCudaImpl, OpCudaMaximum);

  // minimum
  REG_ELEM_WISE_CUDA_F32(minimum, binaryOpCudaImpl, OpCudaMinimum);
  REG_ELEM_WISE_CUDA_F32(minimumOut, binaryOpOutCudaImpl, OpCudaMinimum);

  // equal
  REG_ELEM_WISE_CUDA_F32(eq, binaryOpCompareCudaImpl, OpCudaEq);
  REG_ELEM_WISE_CUDA_F32(eqOut, binaryOpCompareOutCudaImpl, OpCudaEq);

  // not equal
  REG_ELEM_WISE_CUDA_F32(ne, binaryOpCompareCudaImpl, OpCudaNe);
  REG_ELEM_WISE_CUDA_F32(neOut, binaryOpCompareOutCudaImpl, OpCudaNe);

  // less than
  REG_ELEM_WISE_CUDA_F32(lt, binaryOpCompareCudaImpl, OpCudaLt);
  REG_ELEM_WISE_CUDA_F32(ltOut, binaryOpCompareOutCudaImpl, OpCudaLt);

  // less equal
  REG_ELEM_WISE_CUDA_F32(le, binaryOpCompareCudaImpl, OpCudaLe);
  REG_ELEM_WISE_CUDA_F32(leOut, binaryOpCompareOutCudaImpl, OpCudaLe);

  // greater than
  REG_ELEM_WISE_CUDA_F32(gt, binaryOpCompareCudaImpl, OpCudaGt);
  REG_ELEM_WISE_CUDA_F32(gtOut, binaryOpCompareOutCudaImpl, OpCudaGt);

  // greater equal
  REG_ELEM_WISE_CUDA_F32(ge, binaryOpCompareCudaImpl, OpCudaGe);
  REG_ELEM_WISE_CUDA_F32(geOut, binaryOpCompareOutCudaImpl, OpCudaGe);

  // clampMin
  REG_ELEM_WISE_CUDA_F32(clampMin, binaryOpCudaImpl, OpCudaClampMin);
  REG_ELEM_WISE_CUDA_F32(clampMinOut, binaryOpOutCudaImpl, OpCudaClampMin);
  REG_ELEM_WISE_CUDA_F32(clampMinInplace, binaryOpInplaceCudaImpl, OpCudaClampMin);

  // clampMax
  REG_ELEM_WISE_CUDA_F32(clampMax, binaryOpCudaImpl, OpCudaClampMax);
  REG_ELEM_WISE_CUDA_F32(clampMaxOut, binaryOpOutCudaImpl, OpCudaClampMax);
  REG_ELEM_WISE_CUDA_F32(clampMaxInplace, binaryOpInplaceCudaImpl, OpCudaClampMax);
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

void registerTernaryCudaFloat32() {
  // clamp
  REG_ELEM_WISE_CUDA_F32(clamp, ternaryOpCudaImpl, OpCudaClamp);
  REG_ELEM_WISE_CUDA_F32(clampOut, ternaryOpOutCudaImpl, OpCudaClamp);
  REG_ELEM_WISE_CUDA_F32(clampInplace, ternaryOpInplaceCudaImpl, OpCudaClamp);
}

void registerElemWiseCuda() {
  registerUnaryCudaFloat32();

  registerBinaryCudaFloat32();
  registerBinaryCudaInt64();

  registerTernaryCudaFloat32();
}

}  // namespace tinytorch::op