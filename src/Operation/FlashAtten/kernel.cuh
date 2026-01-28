/*
 * TinyFA
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "gemm.cuh"
#include "softmax.cuh"
#include "tile.cuh"

namespace tfa {

template <typename Config>
struct ThreadInfo {
  static constexpr int kWarpSize = Config::kWarpSize;
  static constexpr int kRowsPerWarp = Config::kRowsPerWarp;
  static constexpr int kDimsPerLane = Config::kDimsPerLane;

  const int threadId = threadIdx.x;
  const int blockSize = blockDim.x;
  const int warpId = threadId / kWarpSize;
  const int laneId = threadId % kWarpSize;
  const int warpRowOffset = warpId * kRowsPerWarp;

  __device__ ThreadInfo() = default;
};

template <typename Config, typename Params>
struct BlockInfo {
  static constexpr int kTileQ = Config::kBr;   // tile size for Q
  static constexpr int kTileKV = Config::kBc;  // tile size for KV

  const int batchIdx = blockIdx.z;
  const int headIdx = blockIdx.y;
  const int tileIdxQ = blockIdx.x;
  const int tileQ = tileIdxQ * kTileQ;
  const int headIdxKV;
  const int seqLenQ;   // sequence length for Q
  const int seqLenKV;  // sequence length for KV
  const int tileSizeQ;

  __device__ explicit BlockInfo(const Params& params)
      : headIdxKV(params.getKVHead(headIdx)),
        seqLenQ(params.getSeqLenQ(batchIdx)),
        seqLenKV(params.getSeqLenKV(batchIdx)),
        tileSizeQ(min(kTileQ, seqLenQ - tileQ)) {}
};

template <typename Config, typename Params>
struct KernelContext : ThreadInfo<Config>, BlockInfo<Config, Params> {
  using DType = typename Config::DType;
  using Thread = ThreadInfo<Config>;
  using Block = BlockInfo<Config, Params>;

  static constexpr int kRowsPerWarp = Config::kRowsPerWarp;
  static constexpr int kColsPerLane = Config::kColsPerLane;
  static constexpr int kDimsPerLane = Config::kDimsPerLane;
  static constexpr float kAttnScale = AttentionScale<Config::kHeadDim>::value;

  const Params& params;

  int tileKV = 0;
  int curTileSizeKV = 0;

  __device__ explicit KernelContext(const Params& p) : Thread(), Block(p), params(p) {}

  using Block::batchIdx;
  using Block::headIdx;
  using Block::headIdxKV;
  using Block::kTileKV;
  using Block::seqLenKV;
  using Block::seqLenQ;
  using Block::tileQ;
  using Block::tileSizeQ;

  using Thread::blockSize;
  using Thread::laneId;
  using Thread::threadId;
  using Thread::warpRowOffset;

  __device__ __forceinline__ bool isValidTile() const { return tileQ < seqLenQ; }

  __device__ __forceinline__ int validWarpRows() const {
    int warpStartQ = tileQ + warpRowOffset;
    return (warpStartQ < seqLenQ) ? min(kRowsPerWarp, seqLenQ - warpStartQ) : 0;
  }

  __device__ __forceinline__ int globalRowQ(int localRow) const { return tileQ + warpRowOffset + localRow; }

  __device__ __forceinline__ int globalKV(int n) const { return tileKV + laneId + n * Config::kWarpSize; }

  template <bool kIsCausal>
  __device__ __forceinline__ void setTileKV(int tileIdx) {
    tileKV = tileIdx * kTileKV;
    curTileSizeKV = min(kTileKV, seqLenKV - tileKV);
    if constexpr (kIsCausal) {
      curTileSizeKV = min(curTileSizeKV, tileQ + tileSizeQ - tileKV);
    }
  }

  template <bool kIsCausal>
  __device__ __forceinline__ int numTilesKV() const {
    int tiles = params.getKVTiles(seqLenKV, kTileKV);
    if constexpr (kIsCausal) {
      tiles = min(tiles, ceilDiv(tileQ + tileSizeQ, kTileKV));
    }
    return tiles;
  }

  template <bool kIsCausal>
  __device__ __forceinline__ bool needsCausalMask() const {
    return kIsCausal && (tileKV + kTileKV > tileQ);
  }

  // check if current KV tile is partial
  __device__ __forceinline__ bool isPartialTile() const { return curTileSizeKV < kTileKV; }

  __device__ __forceinline__ const DType* qPtr() const { return params.qPtr(*this); }
  __device__ __forceinline__ const DType* kPtr() const { return params.kPtr(*this, tileKV); }
  __device__ __forceinline__ const DType* vPtr() const { return params.vPtr(*this, tileKV); }
  __device__ __forceinline__ DType* oPtr() const { return params.oPtr(*this); }
  __device__ __forceinline__ int seqDimQ() const { return params.seqDimQ; }
  __device__ __forceinline__ int seqDimKV() const { return params.seqDimKV; }
};

template <typename Config, bool kIsCausal, typename Params>
__global__ void flashAttentionKernel(Params params) {
  using DType = typename Config::DType;
  using Context = KernelContext<Config, Params>;

  constexpr int kRowsPerWarp = Config::kRowsPerWarp;
  constexpr int kColsPerLane = Config::kColsPerLane;

  Context ctx(params);
  if (!ctx.isValidTile()) return;

  extern __shared__ char smemBuf[];
  auto* smem = reinterpret_cast<DType*>(smemBuf);

  QTile<Config> qTile(smem);
  OTile<Config> oTile(smem);
  KTile<Config> kTile(smem + qTile.numElems());
  VTile<Config> vTile(smem + qTile.numElems());

  Softmax<Config> softmax;
  softmax.init();

  // load Q tile
  qTile.gm2sm(ctx.qPtr(), ctx.seqDimQ(), ctx.tileSizeQ, ctx);
  __syncthreads();

  // main loop over KV tiles
  const int numTilesKV = ctx.template numTilesKV<kIsCausal>();
  for (int tileIdx = 0; tileIdx < numTilesKV; tileIdx++) {
    ctx.template setTileKV<kIsCausal>(tileIdx);

    // load K
    kTile.gm2sm(ctx.kPtr(), ctx.seqDimKV(), ctx.curTileSizeKV, ctx);
    __syncthreads();

    // S = Q @ K^T
    float score[kRowsPerWarp][kColsPerLane];
    GemmOp<Config>::template computeScore<kIsCausal>(score, qTile, kTile, ctx);

    // softmax
    softmax.update(score, score, oTile.acc);
    __syncthreads();

    // load V
    vTile.gm2sm(ctx.vPtr(), ctx.seqDimKV(), ctx.curTileSizeKV, ctx);
    __syncthreads();

    // O += P @ V
    GemmOp<Config>::computeOutput(oTile, score, vTile, ctx);
    __syncthreads();
  }

  // normalize
  oTile.normalize(ctx, softmax);
  __syncthreads();

  // store O
  oTile.store(ctx.oPtr(), ctx.seqDimQ(), ctx);
}

}  // namespace tfa
