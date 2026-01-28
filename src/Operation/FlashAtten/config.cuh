/*
 * TinyFA
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace tfa {

template <typename DType_, int HeadDim_, int Br_, int Bc_, int NumWarps_>
struct KernelConfig {
  using DType = DType_;

  static constexpr int kHeadDim = HeadDim_;
  static constexpr int kBr = Br_;
  static constexpr int kBc = Bc_;
  static constexpr int kNumWarps = NumWarps_;

  static constexpr int kWarpSize = 32;
  static constexpr int kNumThreads = kNumWarps * kWarpSize;

  static constexpr bool kUseSwizzle = true;
  static constexpr int kBytesPerVecLoad = 16;  // uint4
  static constexpr int kElemsPerVecLoad = kBytesPerVecLoad / sizeof(DType);

  static_assert(kBr % kNumWarps == 0, "kBr must be divisible by kNumWarps");
  static_assert(kBc % kWarpSize == 0, "kBc must be divisible by kWarpSize");
  static_assert(kHeadDim % kWarpSize == 0, "kHeadDim must be divisible by kWarpSize");

  static constexpr int kRowsPerWarp = kBr / kNumWarps;
  static constexpr int kColsPerLane = kBc / kWarpSize;
  static constexpr int kDimsPerLane = kHeadDim / kWarpSize;
};

#define TFA_HEAD_DIM_CASE(N, HEAD_DIM_VAR, ...) \
  case N: {                                     \
    constexpr int HEAD_DIM_VAR = N;             \
    __VA_ARGS__                                 \
    break;                                      \
  }

#define TFA_DISPATCH_HEAD_DIM(headDim, HEAD_DIM_VAR, ...) \
  [&] {                                                   \
    switch (headDim) {                                    \
      TFA_HEAD_DIM_CASE(32, HEAD_DIM_VAR, __VA_ARGS__)    \
      TFA_HEAD_DIM_CASE(64, HEAD_DIM_VAR, __VA_ARGS__)    \
      TFA_HEAD_DIM_CASE(128, HEAD_DIM_VAR, __VA_ARGS__)   \
      TFA_HEAD_DIM_CASE(256, HEAD_DIM_VAR, __VA_ARGS__)   \
      default:                                            \
        printf("Unsupported headDim: %d\n", headDim);     \
        break;                                            \
    }                                                     \
  }()

// default configuration
template <typename DType, int HeadDim>
struct ConfigForHeadDim {
  using Config = KernelConfig<DType, HeadDim, 32, 32, 4>;
};

#define TFA_DEFINE_CONFIG(DTYPE, HEADDIM, BR, BC, WARPS)        \
  template <>                                                   \
  struct ConfigForHeadDim<DTYPE, HEADDIM> {                     \
    using Config = KernelConfig<DTYPE, HEADDIM, BR, BC, WARPS>; \
  }

using FP32 = float;
using FP16 = __half;
using BF16 = __nv_bfloat16;

TFA_DEFINE_CONFIG(FP32, 128, 64, 64, 8);
TFA_DEFINE_CONFIG(FP16, 128, 128, 64, 8);
TFA_DEFINE_CONFIG(BF16, 128, 128, 64, 8);

}  // namespace tfa
