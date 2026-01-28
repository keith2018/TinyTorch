/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "FlashAtten/launcher.cuh"
#include "OpNNLayerCuda.cuh"

namespace tinytorch::op {

template <typename T>
Tensor flashAttentionOpCudaImpl(const Tensor& query, const Tensor& key, const Tensor& value, bool isCausal) {
  const auto& qShape = query.shape();  // [batch, seqLenQ, numHeadsQ, headDim]
  const auto& kShape = key.shape();    // [batch, seqLenKV, numHeadsKV, headDim]
  ASSERT(qShape.size() == 4);
  ASSERT(kShape.size() == 4);

  auto batch = qShape[0];
  auto numHeadsQ = qShape[2];
  auto numHeadsKV = kShape[2];
  auto headDim = qShape[3];

  ASSERT(numHeadsQ % numHeadsKV == 0);  // GQA

  auto seqLenQ = query.size(1);
  auto seqLenKV = key.size(1);

  using CudaT = typename cuda::CudaTypeCast<T>::type;

  Tensor out(qShape, query.options().noGrad());
  auto* outPtr = out.dataPtr<CudaT>();

  const auto* qPtr = query.dataPtr<CudaT>();
  const auto* kPtr = key.dataPtr<CudaT>();
  const auto* vPtr = value.dataPtr<CudaT>();
  auto* oPtr = out.dataPtr<CudaT>();

  auto stream = cuda::getCurrentCUDAStream(query.device().index).stream();
  tfa::flashAttn<CudaT>(qPtr, kPtr, vPtr, oPtr, batch, seqLenQ, seqLenKV, numHeadsQ, numHeadsKV, headDim, isCausal,
                        stream);
  CUDA_KERNEL_CHECK();
  return out;
}

#define INSTANTIATE_FLASHATTEN_OP(T)                                                                       \
  template Tensor flashAttentionOpCudaImpl<T>(const Tensor& query, const Tensor& key, const Tensor& value, \
                                              bool isCausal);
FOR_FLT_TYPES(INSTANTIATE_FLASHATTEN_OP)
#undef INSTANTIATE_FLASHATTEN_OP

#define REG_NN_LAYER_CUDA_FLT(NAME, FUNC)                                       \
  REGISTER_OP_IMPL(NAME, CUDA, Float32, &(FUNC<DTypeToType_t<DType::Float32>>)) \
  REGISTER_OP_IMPL(NAME, CUDA, Float16, &(FUNC<DTypeToType_t<DType::Float16>>)) \
  REGISTER_OP_IMPL(NAME, CUDA, BFloat16, &(FUNC<DTypeToType_t<DType::BFloat16>>))

void registerNNLayerCuda() {
  // softmax
  REG_NN_LAYER_CUDA_FLT(softmax, softmaxOpCudaImpl);
  REG_NN_LAYER_CUDA_FLT(softmaxOut, softmaxOpOutCudaImpl);
  REG_NN_LAYER_CUDA_FLT(softmaxBackward, softmaxOpBackwardCudaImpl);

  // logSoftmax
  REG_NN_LAYER_CUDA_FLT(logSoftmax, logSoftmaxOpCudaImpl);
  REG_NN_LAYER_CUDA_FLT(logSoftmaxOut, logSoftmaxOpOutCudaImpl);
  REG_NN_LAYER_CUDA_FLT(logSoftmaxBackward, logSoftmaxOpBackwardCudaImpl);

  // dropout
  REG_NN_LAYER_CUDA_FLT(dropout, dropoutOpCudaImpl);
  REG_NN_LAYER_CUDA_FLT(dropoutMasked, dropoutMaskedOpCudaImpl);

  // layerNorm
  REG_NN_LAYER_CUDA_FLT(layerNorm, layerNormOpCudaImpl);

  // rmsNorm
  REG_NN_LAYER_CUDA_FLT(rmsNorm, rmsNormOpCudaImpl);

  // rope
  REG_NN_LAYER_CUDA_FLT(ropeInit, ropeInitOpCudaImpl);
  REG_NN_LAYER_CUDA_FLT(ropeApply, ropeApplyOpCudaImpl);

  // flashAttention
  REG_NN_LAYER_CUDA_FLT(flashAttention, flashAttentionOpCudaImpl);
}

}  // namespace tinytorch::op
