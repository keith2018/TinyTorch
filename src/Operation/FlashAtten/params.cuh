/*
 * TinyFA
 * @author 	: keith@robot9.me
 *
 */

#pragma once

namespace tfa {

template <typename DType_, int kHeadDim_>
struct FixLenParams {
  using DType = DType_;
  static constexpr int kHeadDim = kHeadDim_;

  const DType* __restrict__ Q;
  const DType* __restrict__ K;
  const DType* __restrict__ V;
  DType* __restrict__ O;

  int seqLenQ;
  int seqLenKV;

  int seqDimQ;   // numHeadsQ * kHeadDim
  int seqDimKV;  // numHeadsKV * kHeadDim

  int numKVTiles;  // ceilDiv(seqLenKV, kTileKV)
  int groupSize;   // numHeadsQ / numHeadsKV (GQA)

  __device__ __forceinline__ int getSeqLenQ(int batchIdx) const { return seqLenQ; }
  __device__ __forceinline__ int getSeqLenKV(int batchIdx) const { return seqLenKV; }
  __device__ __forceinline__ int getKVHead(int headIdx) const { return headIdx / groupSize; }
  __device__ __forceinline__ int getKVTiles(int seqLen, int tileKV) const { return numKVTiles; }

  // Layout: [batch, seq, head, dim]
  template <typename Context>
  __device__ __forceinline__ const DType* qPtr(const Context& ctx) const {
    return Q + (ctx.batchIdx * seqLenQ + ctx.tileQ) * seqDimQ + ctx.headIdx * kHeadDim;
  }

  template <typename Context>
  __device__ __forceinline__ const DType* kPtr(const Context& ctx, int seqIdx) const {
    return K + (ctx.batchIdx * seqLenKV + seqIdx) * seqDimKV + ctx.headIdxKV * kHeadDim;
  }

  template <typename Context>
  __device__ __forceinline__ const DType* vPtr(const Context& ctx, int seqIdx) const {
    return V + (ctx.batchIdx * seqLenKV + seqIdx) * seqDimKV + ctx.headIdxKV * kHeadDim;
  }

  template <typename Context>
  __device__ __forceinline__ DType* oPtr(const Context& ctx) const {
    return O + (ctx.batchIdx * seqLenQ + ctx.tileQ) * seqDimQ + ctx.headIdx * kHeadDim;
  }
};

template <typename DType_, int kHeadDim_>
struct VarLenParams {
  using DType = DType_;
  static constexpr int kHeadDim = kHeadDim_;

  const DType* __restrict__ Q;  // [totalQ,  numHeadsQ,  headDim]
  const DType* __restrict__ K;  // [totalKV, numHeadsKV, headDim]
  const DType* __restrict__ V;  // [totalKV, numHeadsKV, headDim]
  DType* __restrict__ O;        // [totalQ,  numHeadsQ,  headDim]

  const int* __restrict__ cuSeqLensQ;   // [batch + 1], cumulative sequence lengths for Q
  const int* __restrict__ cuSeqLensKV;  // [batch + 1], cumulative sequence lengths for KV

  int maxSeqLenQ;
  int maxSeqLenKV;

  int seqDimQ;   // numHeadsQ * headDim
  int seqDimKV;  // numHeadsKV * headDim

  int maxKVTiles;  // ceilDiv(maxSeqLenKV, kTileKV)
  int groupSize;   // numHeadsQ / numHeadsKV (GQA)

  __device__ __forceinline__ int getSeqLenQ(int batchIdx) const {
    return cuSeqLensQ[batchIdx + 1] - cuSeqLensQ[batchIdx];
  }
  __device__ __forceinline__ int getSeqLenKV(int batchIdx) const {
    return cuSeqLensKV[batchIdx + 1] - cuSeqLensKV[batchIdx];
  }
  __device__ __forceinline__ int getKVHead(int headIdx) const { return headIdx / groupSize; }
  __device__ __forceinline__ int getKVTiles(int seqLen, int tileKV) const { return ceilDiv(seqLen, tileKV); }

  // Layout: [totalSeq, numHeads, headDim] (packed, no padding)
  template <typename Context>
  __device__ __forceinline__ const DType* qPtr(const Context& ctx) const {
    return Q + (cuSeqLensQ[ctx.batchIdx] + ctx.tileQ) * seqDimQ + ctx.headIdx * kHeadDim;
  }

  template <typename Context>
  __device__ __forceinline__ const DType* kPtr(const Context& ctx, int seqIdx) const {
    return K + (cuSeqLensKV[ctx.batchIdx] + seqIdx) * seqDimKV + ctx.headIdxKV * kHeadDim;
  }

  template <typename Context>
  __device__ __forceinline__ const DType* vPtr(const Context& ctx, int seqIdx) const {
    return V + (cuSeqLensKV[ctx.batchIdx] + seqIdx) * seqDimKV + ctx.headIdxKV * kHeadDim;
  }

  template <typename Context>
  __device__ __forceinline__ DType* oPtr(const Context& ctx) const {
    return O + (cuSeqLensQ[ctx.batchIdx] + ctx.tileQ) * seqDimQ + ctx.headIdx * kHeadDim;
  }
};

}  // namespace tfa
