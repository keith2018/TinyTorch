/*
 * TinyFA
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "layout.cuh"
#include "memory.cuh"
#include "utils.cuh"

namespace tfa {

template <typename Config, int NumRows>
struct Tile {
  using DType = typename Config::DType;
  using Layout = TileLayout<Config>;
  static constexpr int kHeadDim = Config::kHeadDim;
  static constexpr int kElemsPerVec = Config::kElemsPerVecLoad;

  DType* smem;

  __device__ explicit Tile(DType* smemBase) : smem(smemBase) {}

  static constexpr size_t numElems() { return NumRows * kHeadDim; }
  static constexpr size_t smemSize() { return numElems() * sizeof(DType); }

  template <typename Context>
  __device__ __forceinline__ void gm2sm(const DType* __restrict__ globalPtr, int stride, int validRows,
                                        const Context& ctx) {
    MemLoader<Config>::template gm2sm<NumRows>(smem, globalPtr, stride, validRows, ctx);
  }

  __device__ __forceinline__ DType at(int row, int col) const { return smem[Layout::map(row, col, kHeadDim)]; }

  __device__ __forceinline__ void sm2reg(DType* __restrict__ regPtr, int row, int vecIdx) const {
    int smemIdx = Layout::map(row, vecIdx * kElemsPerVec, kHeadDim);
    MemLoader<Config>::sm2reg(regPtr, &smem[smemIdx]);
  }
};

template <typename Config>
using QTile = Tile<Config, Config::kBr>;

template <typename Config>
using KTile = Tile<Config, Config::kBc>;

template <typename Config>
using VTile = Tile<Config, Config::kBc>;

template <typename Config>
struct OTile {
  using DType = typename Config::DType;
  using Layout = TileLayout<Config>;

  static constexpr int kHeadDim = Config::kHeadDim;
  static constexpr int kRowsPerWarp = Config::kRowsPerWarp;
  static constexpr int kDimsPerLane = Config::kDimsPerLane;

  float acc[kRowsPerWarp][kDimsPerLane]{};
  DType* smemPtr;

  __device__ explicit OTile(DType* smemBase) : smemPtr(smemBase) {
#pragma unroll
    for (int m = 0; m < kRowsPerWarp; m++) {
#pragma unroll
      for (int k = 0; k < kDimsPerLane; k++) {
        acc[m][k] = 0.f;
      }
    }
  }

  template <typename Context, typename SoftmaxT>
  __device__ __forceinline__ void normalize(const Context& ctx, const SoftmaxT& softmax) {
    const int validRows = ctx.validWarpRows();
#pragma unroll
    for (int m = 0; m < kRowsPerWarp; m++) {
      if (m < validRows) {
        float norm = softmax.getNorm(m);
        int row = ctx.warpRowOffset + m;
#pragma unroll
        for (int k = 0; k < kDimsPerLane; k++) {
          int col = ctx.laneId * kDimsPerLane + k;
          int smemIdx = Layout::map(row, col, kHeadDim);
          smemPtr[smemIdx] = fromFloat<DType>(acc[m][k] * norm);
        }
      }
    }
  }

  template <typename Context>
  __device__ __forceinline__ void store(DType* __restrict__ globalPtr, int stride, const Context& ctx) {
    MemLoader<Config>::template sm2gm<Config::kBr>(globalPtr, smemPtr, stride, ctx.tileSizeQ, ctx);
  }
};

}  // namespace tfa
