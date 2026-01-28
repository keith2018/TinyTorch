/*
 * TinyFA
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "config.cuh"
#include "tile.cuh"
#include "utils.cuh"

namespace tfa {

template <typename Config>
struct GemmOp {
  using DType = typename Config::DType;

  static constexpr int kDimsPerLane = Config::kDimsPerLane;
  static constexpr int kRowsPerWarp = Config::kRowsPerWarp;
  static constexpr int kColsPerLane = Config::kColsPerLane;
  static constexpr int kElemsPerVec = Config::kElemsPerVecLoad;
  static constexpr int kWarpSize = Config::kWarpSize;

  static_assert(Config::kHeadDim % kElemsPerVec == 0, "kHeadDim must be divisible by kElemsPerVec");
  static constexpr int kNumVecs = Config::kHeadDim / kElemsPerVec;

  // S = Q @ K^T
  template <bool kIsCausal, typename S, typename Context>
  __device__ static void computeScore(S& s, const QTile<Config>& qTile, const KTile<Config>& kTile, Context& ctx) {
    bool needsCausal = ctx.template needsCausalMask<kIsCausal>();
    bool isPartial = ctx.isPartialTile();

    if (needsCausal) {
      isPartial ? computeScoreImpl<true, true>(s, qTile, kTile, ctx)
                : computeScoreImpl<true, false>(s, qTile, kTile, ctx);
    } else {
      isPartial ? computeScoreImpl<false, true>(s, qTile, kTile, ctx)
                : computeScoreImpl<false, false>(s, qTile, kTile, ctx);
    }
  }

 private:
  template <bool kCausalMask, bool kBoundaryMask, typename S, typename Context>
  __device__ static void computeScoreImpl(S& s, const QTile<Config>& qTile, const KTile<Config>& kTile, Context& ctx) {
    DType qReg[kRowsPerWarp][kElemsPerVec];
    DType kReg[kColsPerLane][kElemsPerVec];

    initScores(s);
    accumDotProducts<kBoundaryMask>(s, qReg, kReg, qTile, kTile, ctx);
    applyScaleAndMask<kCausalMask, kBoundaryMask>(s, ctx);
  }
  template <typename S>
  __device__ static __forceinline__ void initScores(S& s) {
#pragma unroll
    for (int m = 0; m < kRowsPerWarp; m++) {
#pragma unroll
      for (int n = 0; n < kColsPerLane; n++) {
        s[m][n] = 0.f;
      }
    }
  }

  template <bool kBoundaryMask, typename S, typename Context>
  __device__ static __forceinline__ void accumDotProducts(S& s, DType qReg[][kElemsPerVec], DType kReg[][kElemsPerVec],
                                                          const QTile<Config>& qTile, const KTile<Config>& kTile,
                                                          Context& ctx) {
#pragma unroll
    for (int v = 0; v < kNumVecs; v++) {
      loadQ(qReg, qTile, ctx, v);
      loadK<kBoundaryMask>(kReg, kTile, ctx, v);
      accumulate(s, qReg, kReg);
    }
  }

  template <typename Context>
  __device__ static __forceinline__ void loadQ(DType qReg[][kElemsPerVec], const QTile<Config>& qTile,
                                               const Context& ctx, int v) {
#pragma unroll
    for (int m = 0; m < kRowsPerWarp; m++) {
      qTile.sm2reg(&qReg[m][0], ctx.warpRowOffset + m, v);
    }
  }

  template <bool kBoundaryMask, typename Context>
  __device__ static __forceinline__ void loadK(DType kReg[][kElemsPerVec], const KTile<Config>& kTile,
                                               const Context& ctx, int v) {
#pragma unroll
    for (int n = 0; n < kColsPerLane; n++) {
      int colN = ctx.laneId + n * kWarpSize;
      if constexpr (kBoundaryMask) {
        if (colN < ctx.curTileSizeKV) {
          kTile.sm2reg(&kReg[n][0], colN, v);
        } else {
#pragma unroll
          for (int e = 0; e < kElemsPerVec; e++) {
            kReg[n][e] = fromFloat<DType>(0.f);
          }
        }
      } else {
        kTile.sm2reg(&kReg[n][0], colN, v);
      }
    }
  }

  template <typename S>
  __device__ static __forceinline__ void accumulate(S& s, const DType qReg[][kElemsPerVec],
                                                    const DType kReg[][kElemsPerVec]) {
#pragma unroll
    for (int e = 0; e < kElemsPerVec; e++) {
#pragma unroll
      for (int m = 0; m < kRowsPerWarp; m++) {
        float q = toFloat(qReg[m][e]);
#pragma unroll
        for (int n = 0; n < kColsPerLane; n++) {
          s[m][n] += q * toFloat(kReg[n][e]);
        }
      }
    }
  }

  template <bool kCausalMask, bool kBoundaryMask, typename S, typename Context>
  __device__ static __forceinline__ void applyScaleAndMask(S& s, const Context& ctx) {
#pragma unroll
    for (int m = 0; m < kRowsPerWarp; m++) {
      int globalQ = ctx.globalRowQ(m);
      int globalKV = ctx.globalKV(0);
#pragma unroll
      for (int n = 0; n < kColsPerLane; n++) {
        bool masked = false;
        if constexpr (kBoundaryMask) {
          masked |= (ctx.laneId + n * kWarpSize >= ctx.curTileSizeKV);
        }
        if constexpr (kCausalMask) {
          masked |= (globalKV > globalQ);
          globalKV += kWarpSize;
        }
        s[m][n] = masked ? -INFINITY : (s[m][n] * ctx.kAttnScale);
      }
    }
  }

 public:
  // O += P @ V
  template <typename P, typename Context>
  __device__ static void computeOutput(OTile<Config>& oTile, const P& prob, const VTile<Config>& vTile,
                                       const Context& ctx) {
    constexpr int kVecsPerThread = kDimsPerLane / kElemsPerVec;
    const int vecBase = ctx.laneId * kVecsPerThread;
    DType vReg[kDimsPerLane];

#pragma unroll
    for (int n = 0; n < Config::kBc; n++) {
      loadV<kVecsPerThread>(vReg, vTile, ctx, n, vecBase);
      accumPV(oTile.acc, prob, vReg, n);
    }
  }

 private:
  template <int kVecsPerThread, typename Context>
  __device__ static __forceinline__ void loadV(DType vReg[], const VTile<Config>& vTile, const Context& ctx, int n,
                                               int vecBase) {
    constexpr int kRemainElems = kDimsPerLane % kElemsPerVec;

#pragma unroll
    for (int vi = 0; vi < kVecsPerThread; vi++) {
      vTile.sm2reg(&vReg[vi * kElemsPerVec], n, vecBase + vi);
    }

    // remaining elements
    if constexpr (kRemainElems > 0) {
      int remainOffset = ctx.laneId * kDimsPerLane + kVecsPerThread * kElemsPerVec;
#pragma unroll
      for (int ri = 0; ri < kRemainElems; ri++) {
        vReg[kVecsPerThread * kElemsPerVec + ri] = vTile.at(n, remainOffset + ri);
      }
    }
  }

  template <typename AccO, typename P>
  __device__ static __forceinline__ void accumPV(AccO& accO, const P& prob, const DType vReg[], int n) {
    int srcLane = n % kWarpSize;
    int colIdx = n / kWarpSize;

#pragma unroll
    for (int k = 0; k < kDimsPerLane; k++) {
      float v = toFloat(vReg[k]);
#pragma unroll
      for (int m = 0; m < kRowsPerWarp; m++) {
        accO[m][k] += __shfl_sync(0xffffffff, prob[m][colIdx], srcLane) * v;
      }
    }
  }
};

}  // namespace tfa
