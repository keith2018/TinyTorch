/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <cublas_v2.h>

#include "TensorImpl.h"

namespace TinyTorch {

struct TensorCudaCtx {
  int32_t dimCount_;
  int32_t elemCount_;
  int32_t shape_[TENSOR_MAX_DIMS];
  int32_t strides_[TENSOR_MAX_DIMS];
  float *data_;
};

class AllocatorCUDA : public Allocator {
 public:
  void allocate(void **ptr, size_t size) override;
  void deallocate(void *ptr) override;
};

class RandomGeneratorCUDA {
 public:
  static void setSeed(unsigned long seed) { seed_ = seed; }
  static unsigned long getSeed() { return seed_; }
  static unsigned long nextSequence() { return sequence_++; }

 private:
  static unsigned long seed_;
  static unsigned long sequence_;
};

class TensorOpsCUDA : public TensorOperations {
 public:
  explicit TensorOpsCUDA(size_t blockSize = 512);
  ~TensorOpsCUDA() override;

  unsigned int getGridSize(size_t n, int32_t batch = 1) const {
    return (unsigned int)((n + (blockSize_ * batch) - 1) /
                          (blockSize_ * batch));
  }
  unsigned int getBlockSize() const { return (unsigned int)blockSize_; }

  cublasHandle_t getCublasHandle();
  static TensorCudaCtx getTensorCtx(const TensorImpl &t);

  TENSOR_OPS_DECLARE(, override)

 protected:
  // op single
  template <typename OP>
  void opSingle_(TensorImpl &t) const;
  template <typename OP>
  TensorImpl opSingle(const TensorImpl &t) const;

  // op pair
  template <typename OP>
  TensorImpl opPair(const TensorImpl &a, const TensorImpl &b) const;
  template <typename OP>
  TensorImpl opPair(const TensorImpl &a, float b) const;
  template <typename OP>
  TensorImpl opPair(float a, const TensorImpl &b) const;
  template <typename OP>
  TensorImpl opPairScalarFirst(const TensorImpl &a, const TensorImpl &b) const;
  template <typename OP>
  TensorImpl opPairScalarSecond(const TensorImpl &a, const TensorImpl &b) const;

  // op pair inplace
  template <typename OP>
  void opPair_(TensorImpl &t, float b) const;
  template <typename OP>
  void opPair_(TensorImpl &t, const TensorImpl &b) const;
  template <typename OP>
  void opPairScalarFirst_(TensorImpl &a, const TensorImpl &b) const;
  template <typename OP>
  void opPairScalarSecond_(TensorImpl &a, const TensorImpl &b) const;

  // op pair broadcast
  template <typename OP>
  void broadcastImpl(TensorImpl &result, const TensorImpl &a,
                     const TensorImpl &b) const;
  template <typename OP>
  TensorImpl opPairBroadcast(const TensorImpl &a, const TensorImpl &b) const;
  template <typename OP>
  void opPairBroadcast_(TensorImpl &a, const TensorImpl &b) const;

  // reduce
  template <typename Compare>
  std::pair<TensorImpl, TensorImpl> reduceDim(const TensorImpl &t, int32_t dim,
                                              bool keepDims, float initVal,
                                              Compare comp);

 protected:
  int32_t cudaDeviceIdx_ = 0;
  size_t blockSize_;
  cublasHandle_t blasHandle_ = nullptr;
};

}  // namespace TinyTorch
