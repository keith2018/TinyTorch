/*
 * TinyFA
 * @author 	: keith@robot9.me
 *
 */

#pragma once

namespace tfa {

struct LayoutIdentity {
  __device__ __forceinline__ static int map(int row, int col, int stride) { return row * stride + col; }
};

// Ref https://leimao.github.io/blog/CuTe-Swizzle/.
template <int BBits = 3, int MBase = 0, int SShift = 3>
struct CuteSwizzle {
  static constexpr int mbase = MBase;
  static constexpr int mask_bits = BBits;
  static constexpr int mask_shift = SShift;

  static constexpr int bit_mask = (1 << mask_bits) - 1;
  static constexpr int yy_mask = bit_mask << (mbase + mask_shift);
  static constexpr int yy_mask_lowest_bit = yy_mask & -yy_mask;

  __device__ __forceinline__ constexpr static int apply(int offset) {
    const int row_shifted = (offset & yy_mask) >> mask_shift;
    return offset ^ row_shifted;
  }
};

template <typename DType, int HeadDim>
struct LayoutSwizzle {
  static constexpr int kVecBytes = 16;
  static constexpr int kDTypeBytes = sizeof(DType);
  static constexpr int kVecElem = kVecBytes / kDTypeBytes;

  static constexpr int MBase = (kVecElem == 8) ? 3 : (kVecElem == 4) ? 2 : 0;

  static constexpr int kHeadDimBits = (HeadDim == 256)   ? 8
                                      : (HeadDim == 128) ? 7
                                      : (HeadDim == 64)  ? 6
                                      : (HeadDim == 32)  ? 5
                                                         : 0;

  static_assert(kHeadDimBits > 0, "Unsupported HeadDim");
  static constexpr int kSShift = kHeadDimBits - MBase;

  using Swizzle = CuteSwizzle<3, MBase, kSShift>;

  __device__ __forceinline__ static int map(int row, int col, int stride) {
    int offset = row * stride + col;
    return Swizzle::apply(offset);
  }
};

template <typename Config>
using TileLayout =
    typename std::conditional<Config::kUseSwizzle, LayoutSwizzle<typename Config::DType, Config::kHeadDim>,
                              LayoutIdentity>::type;

}  // namespace tfa
