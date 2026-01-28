/*
 * TinyFA
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "layout.cuh"

namespace tfa {

template <typename Config>
struct MemLoader {
  using DType = typename Config::DType;
  using Layout = TileLayout<Config>;
  using VecT = int4;  // 128-bit

  static constexpr int kElemsPerVec = Config::kElemsPerVecLoad;
  static constexpr int kHeadDim = Config::kHeadDim;
  static constexpr int kNumVecs = kHeadDim / kElemsPerVec;

  template <int Rows, typename Context, bool IsLoad>
  __device__ __forceinline__ static void copy(DType* smem, DType* gmem, int stride, int validRows, const Context& ctx) {
    static_assert(kHeadDim % kElemsPerVec == 0, "kHeadDim must be divisible by kElemsPerVec");

    for (int idx = ctx.threadId; idx < Rows * kNumVecs; idx += ctx.blockSize) {
      int row = idx / kNumVecs;
      int col = (idx % kNumVecs) * kElemsPerVec;
      int smemIdx = Layout::map(row, col, kHeadDim);

      VecT* sPtr = reinterpret_cast<VecT*>(&smem[smemIdx]);
      VecT* gPtr = reinterpret_cast<VecT*>(gmem + row * stride + col);

      if (IsLoad) {
        // global -> shared
        *sPtr = (row < validRows) ? *gPtr : make_int4(0, 0, 0, 0);
      } else if (row < validRows) {
        // shared -> global
        *gPtr = *sPtr;
      }
    }
  }

  template <int Rows, typename Context>
  __device__ __forceinline__ static void gm2sm(DType* __restrict__ smem, const DType* __restrict__ gmem, int stride,
                                               int validRows, const Context& ctx) {
    copy<Rows, Context, true>(smem, const_cast<DType*>(gmem), stride, validRows, ctx);
  }

  template <int Rows, typename Context>
  __device__ __forceinline__ static void sm2gm(DType* __restrict__ gmem, const DType* __restrict__ smem, int stride,
                                               int validRows, const Context& ctx) {
    copy<Rows, Context, false>(const_cast<DType*>(smem), gmem, stride, validRows, ctx);
  }

  __device__ __forceinline__ static void sm2reg(DType* __restrict__ reg, const DType* __restrict__ smem) {
    *reinterpret_cast<VecT*>(reg) = *reinterpret_cast<const VecT*>(smem);
  }
};

}  // namespace tfa
