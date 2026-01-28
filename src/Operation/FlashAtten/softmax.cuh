/*
 * TinyFA
 * @author 	: keith@robot9.me
 *
 */

#pragma once

namespace tfa {

template <typename Config>
struct Softmax {
  static constexpr int kWarpSize = Config::kWarpSize;
  static constexpr int kRowsPerWarp = Config::kRowsPerWarp;
  static constexpr int kColsPerLane = Config::kColsPerLane;
  static constexpr int kDimsPerLane = Config::kDimsPerLane;

  float rowMax[kRowsPerWarp];
  float rowSum[kRowsPerWarp];

  __device__ void init() {
#pragma unroll
    for (int m = 0; m < kRowsPerWarp; m++) {
      rowMax[m] = -INFINITY;
      rowSum[m] = 0.f;
    }
  }

  template <typename AccO, typename P, typename S>
  __device__ __forceinline__ void update(P& prob, const S& score, AccO& accO) {
#pragma unroll
    for (int m = 0; m < kRowsPerWarp; m++) {
      float newMax = computeRowMax(score[m]);
      float correction = rescaleState(m, newMax);
      float localSum = computeExpAndSum(prob[m], score[m], rowMax[m]);
      updateRowSum(m, correction, localSum);
      rescaleOutput(accO[m], correction);
    }
  }

  __device__ __forceinline__ float getNorm(int m) const { return (rowSum[m] > 0.f) ? (1.f / rowSum[m]) : 0.f; }

 private:
  template <typename Row>
  __device__ __forceinline__ float computeRowMax(const Row& row) const {
    float localMax = row[0];
#pragma unroll
    for (int n = 1; n < kColsPerLane; n++) {
      localMax = fmaxf(localMax, row[n]);
    }
    return warpReduceMax(localMax);
  }

  __device__ __forceinline__ float rescaleState(int m, float newMax) {
    float prevMax = rowMax[m];
    rowMax[m] = fmaxf(prevMax, newMax);
    return fastExp(prevMax - rowMax[m]);
  }

  template <typename PRow, typename SRow>
  __device__ __forceinline__ float computeExpAndSum(PRow& prob, const SRow& score, float maxVal) const {
    float localSum = 0.f;
#pragma unroll
    for (int n = 0; n < kColsPerLane; n++) {
      float p = fastExp(score[n] - maxVal);
      prob[n] = p;
      localSum += p;
    }
    return localSum;
  }

  __device__ __forceinline__ void updateRowSum(int m, float correction, float localSum) {
    float warpSum = warpReduceSum(localSum);
    rowSum[m] = rowSum[m] * correction + warpSum;
  }

  template <typename AccRow>
  __device__ __forceinline__ void rescaleOutput(AccRow& accRow, float correction) const {
#pragma unroll
    for (int k = 0; k < kDimsPerLane; k++) {
      accRow[k] *= correction;
    }
  }

  __device__ __forceinline__ float warpReduceMax(float val) const {
#pragma unroll
    for (int delta = kWarpSize / 2; delta > 0; delta >>= 1) {
      val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, delta));
    }
    return val;
  }

  __device__ __forceinline__ float warpReduceSum(float val) const {
#pragma unroll
    for (int delta = kWarpSize / 2; delta > 0; delta >>= 1) {
      val += __shfl_xor_sync(0xffffffff, val, delta);
    }
    return val;
  }
};

}  // namespace tfa
