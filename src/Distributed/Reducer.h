/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "DistributedProcessGroup.h"
#include "Tensor.h"

namespace tinytorch::distributed {

constexpr auto kDefaultBucketBytesCap = 25 * 1024 * 1024;  // 25MB

class Reducer {
 public:
  struct ParameterHookCtx {
    Reducer* self;
    int64_t bIdx;
    int64_t pIdx;
  };

  Reducer(std::vector<TensorPtr> parameters, std::shared_ptr<DistributedProcessGroup> processGroup,
          int64_t bucketBytesCap = kDefaultBucketBytesCap);

  ~Reducer() = default;

  void prepareForBackward();
  void broadcastParameters(int rootRank = 0) const;
  void synchronizeGradients();

  bool hasUnfinishedOperations();
  void waitForAllOperations();
  bool allGradientsReady();

  void onGradReady(int64_t bucketIdx, int64_t paramIdx, const Tensor& grad);

 private:
  struct Bucket {
    std::vector<TensorPtr> params;
    std::vector<int64_t> paramSizes;
    std::vector<int64_t> paramOffsets;

    Tensor flatBuffer;

    int64_t totalSize = 0;
    int64_t readyCount = 0;
    bool allReduceStarted = false;
  };

  void buildBuckets();
  void registerHooks();
  void reduceBucket(int64_t bucketIdx);

  void initBucket(Bucket& bucket, Device device, DType dtype);

  void copyParamsToFlattenedBuffer(int64_t bucketIdx) const;
  void copyFlattenedBufferToParams(int64_t bucketIdx) const;
  void copyGradsToFlattenedBuffer(int64_t bucketIdx) const;
  void copyFlattenedBufferToGrads(int64_t bucketIdx) const;

  std::vector<TensorPtr> parameters_;
  std::shared_ptr<DistributedProcessGroup> processGroup_;
  int64_t bucketBytesCap_;

  std::vector<Bucket> buckets_;
  std::vector<std::unique_ptr<ParameterHookCtx>> hookCtxs_;
};

}  // namespace tinytorch::distributed
