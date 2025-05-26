/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <functional>
#include <optional>

#include "../Allocator.h"
#include "TensorImpl.h"

namespace tinytorch {

class AllocatorCPU : public Allocator {
 public:
  Device device() override { return Device::CPU; }

  void allocate(void **ptr, size_t size) override;
  void deallocate(void *ptr) override;

 private:
  static void *allocateAlign(size_t size, size_t alignment);
  static void deallocateAlign(void *ptr);

  static void *allocatePinned(size_t size);
  static void deallocatePinned(void *ptr);

  bool pinned_ = false;
};

class RandomGeneratorCPU {
 public:
  static void setSeed(const unsigned long seed) {
    seed_ = seed;
    randomEngine_ = std::default_random_engine(seed_.value());
  }
  static std::default_random_engine getGenerator() {
    if (seed_.has_value()) {
      return randomEngine_;
    }
    std::random_device r;
    return std::default_random_engine(r());
  }

 private:
  static std::optional<unsigned long> seed_;
  static std::default_random_engine randomEngine_;
};

class TensorOpsCPU : public TensorOperations {
 public:
  TensorOpsCPU();
  ~TensorOpsCPU() override;

  TENSOR_OPS_DECLARE(, override)

 protected:
  // op single
  template <typename OP>
  static void opSingle_(TensorImpl &t);

  template <typename OP>
  static TensorImpl opSingle(const TensorImpl &t);

  // op pair
  template <typename OP>
  static TensorImpl opPair(const TensorImpl &a, const TensorImpl &b);
  template <typename OP>
  static TensorImpl opPair(const TensorImpl &a, float b);
  template <typename OP>
  static TensorImpl opPair(float a, const TensorImpl &b);

  // op pair inplace
  template <typename OP>
  static void opPair_(TensorImpl &t, float b);
  template <typename OP>
  static void opPair_(TensorImpl &t, const TensorImpl &b);

  // op pair broadcast
  template <typename OP, bool REVERSE>
  static void broadcastImplLeadingOnes(TensorImpl &result,
                                       const TensorImpl &larger,
                                       const TensorImpl &smaller);

  template <typename OP, bool REVERSE>
  static void broadcastImplTrailingOnes(TensorImpl &result,
                                        const TensorImpl &larger,
                                        const TensorImpl &smaller);

  template <typename OP>
  static void broadcastImplCommon(TensorImpl &result, const TensorImpl &a,
                                  const TensorImpl &b);

  template <typename OP>
  static void broadcastImpl(TensorImpl &result, const TensorImpl &a,
                            const TensorImpl &b);
  template <typename OP>
  static TensorImpl opPairBroadcast(const TensorImpl &a, const TensorImpl &b);
  template <typename OP>
  static void opPairBroadcast_(TensorImpl &a, const TensorImpl &b);

  // reduce
  static int32_t getReduceSrcIndex(const TensorImpl &ret, const TensorImpl &t,
                                   int32_t idx, int32_t dim, bool keepDims);
  static int32_t getReduceDstIndex(const TensorImpl &t, int32_t idx,
                                   int32_t dim);
  static int32_t getReduceDstIndex(const TensorImpl &t, int32_t idx,
                                   const FixedVector<uint8_t> &inAxis);

  template <typename OP>
  void reduceAll(float *output, const float *input, int32_t n);
  template <typename OP>
  void reduceIdxAll(float *output, const float *input, int32_t n);

  template <typename Compare, bool IsLastDim>
  static void reduceIdxDimImpl(TensorImpl &values, TensorImpl &indices,
                               const TensorImpl &t, int32_t dim, bool keepDims,
                               float initVal, Compare comp);
  template <typename Compare>
  std::pair<TensorImpl, TensorImpl> reduceIdxDim(const TensorImpl &t,
                                                 int32_t dim, bool keepDims,
                                                 float initVal, Compare comp);

  template <typename Op>
  TensorImpl reduceMultiDim(const TensorImpl &t,
                            const std::vector<int32_t> &dims, bool keepDims,
                            Op op);

  // index
  static void getSubIndices(
      int32_t *subIndices, const TensorImpl &t,
      const std::vector<std::reference_wrapper<TensorImpl>> &indices,
      int32_t idx);

 protected:
  CachedAllocator allocator_;
};

}  // namespace tinytorch
