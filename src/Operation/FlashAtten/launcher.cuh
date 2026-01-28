/*
 * TinyFA
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <cassert>

#include "kernel.cuh"
#include "params.cuh"
#include "utils.cuh"

namespace tfa {

namespace detail {

template <typename Config, typename Params>
void launchKernel(const Params& params, int gridX, int numHeads, int batchSize, bool isCausal, cudaStream_t stream) {
  size_t smemSize = QTile<Config>::smemSize() + KTile<Config>::smemSize();

  dim3 grid(gridX, numHeads, batchSize);
  dim3 block(Config::kNumThreads);

  auto kernel = isCausal ? flashAttentionKernel<Config, true, Params> : flashAttentionKernel<Config, false, Params>;

  bool sharedMemOk = setDynamicSharedMemory(kernel, smemSize);
  assert(sharedMemOk && "error: shared memory not fit");

  kernel<<<grid, block, smemSize, stream>>>(params);
}

template <typename DType, int kHeadDim>
void flashAttnImpl(const DType* Q, const DType* K, const DType* V, DType* O, int batchSize, int seqLenQ, int seqLenKV,
                   int numHeadsQ, int numHeadsKV, bool isCausal, cudaStream_t stream) {
  using ConfigHelper = ConfigForHeadDim<DType, kHeadDim>;
  using Config = typename ConfigHelper::Config;
  using Params = FixLenParams<DType, kHeadDim>;

  Params params;
  params.Q = Q;
  params.K = K;
  params.V = V;
  params.O = O;
  params.seqLenQ = seqLenQ;
  params.seqLenKV = seqLenKV;
  params.numKVTiles = ceilDiv(seqLenKV, Config::kBc);
  params.seqDimQ = numHeadsQ * kHeadDim;
  params.seqDimKV = numHeadsKV * kHeadDim;
  params.groupSize = numHeadsQ / numHeadsKV;

  launchKernel<Config>(params, ceilDiv(seqLenQ, Config::kBr), numHeadsQ, batchSize, isCausal, stream);
}

template <typename DType, int kHeadDim>
void flashAttnVarLenImpl(const DType* Q, const DType* K, const DType* V, DType* O, const int* cuSeqLensQ,
                         const int* cuSeqLensKV, int batchSize, int maxSeqLenQ, int maxSeqLenKV, int numHeadsQ,
                         int numHeadsKV, bool isCausal, cudaStream_t stream) {
  using ConfigHelper = ConfigForHeadDim<DType, kHeadDim>;
  using Config = typename ConfigHelper::Config;
  using Params = VarLenParams<DType, kHeadDim>;

  Params params;
  params.Q = Q;
  params.K = K;
  params.V = V;
  params.O = O;
  params.cuSeqLensQ = cuSeqLensQ;
  params.cuSeqLensKV = cuSeqLensKV;
  params.maxSeqLenQ = maxSeqLenQ;
  params.maxSeqLenKV = maxSeqLenKV;
  params.maxKVTiles = ceilDiv(maxSeqLenKV, Config::kBc);
  params.seqDimQ = numHeadsQ * kHeadDim;
  params.seqDimKV = numHeadsKV * kHeadDim;
  params.groupSize = numHeadsQ / numHeadsKV;

  launchKernel<Config>(params, ceilDiv(maxSeqLenQ, Config::kBr), numHeadsQ, batchSize, isCausal, stream);
}

}  // namespace detail

template <typename DType>
void flashAttn(const DType* Q, const DType* K, const DType* V, DType* O, int batchSize, int seqLenQ, int seqLenKV,
               int numHeadsQ, int numHeadsKV, int headDim, bool isCausal = false, cudaStream_t stream = nullptr) {
  TFA_DISPATCH_HEAD_DIM(headDim, kHeadDim, {
    detail::flashAttnImpl<DType, kHeadDim>(Q, K, V, O, batchSize, seqLenQ, seqLenKV, numHeadsQ, numHeadsKV, isCausal,
                                           stream);
  });
}

template <typename DType>
void flashAttnVarLen(const DType* Q, const DType* K, const DType* V, DType* O, const int* cuSeqLensQ,
                     const int* cuSeqLensKV, int batchSize, int maxSeqLenQ, int maxSeqLenKV, int numHeadsQ,
                     int numHeadsKV, int headDim, bool isCausal = false, cudaStream_t stream = nullptr) {
  TFA_DISPATCH_HEAD_DIM(headDim, kHeadDim, {
    detail::flashAttnVarLenImpl<DType, kHeadDim>(Q, K, V, O, cuSeqLensQ, cuSeqLensKV, batchSize, maxSeqLenQ,
                                                 maxSeqLenKV, numHeadsQ, numHeadsKV, isCausal, stream);
  });
}

}  // namespace tfa
