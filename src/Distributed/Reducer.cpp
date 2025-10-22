/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Reducer.h"

#include <algorithm>
#include <thread>

#include "Utils/CUDAUtils.h"

namespace tinytorch::distributed {

static void parameterHookFn(void* ctx, Tensor& grad) {
  auto* c = static_cast<Reducer::ParameterHookCtx*>(ctx);
  c->self->onGradReady(c->bIdx, c->pIdx, grad);
}

Reducer::Reducer(std::vector<TensorPtr> parameters, std::shared_ptr<DistributedProcessGroup> processGroup,
                 int64_t bucketBytesCap)
    : parameters_(std::move(parameters)),
      processGroup_(std::move(processGroup)),
      bucketBytesCap_(bucketBytesCap),
      bucketReadyCnt_(0) {
  buildBuckets();
  registerHooks();
}

void Reducer::buildBuckets() {
  if (parameters_.empty()) {
    return;
  }

  auto device = parameters_[0]->device();
  auto dtype = parameters_[0]->dtype();

  int64_t currentBytes = 0;
  std::vector<TensorPtr> currentParams;

  for (auto& param : parameters_) {
    ASSERT(param != nullptr);
    ASSERT(param->device() == device);
    ASSERT(param->dtype() == dtype);

    int64_t paramBytes = param->nbytes();
    if (currentBytes + paramBytes > bucketBytesCap_ && !currentParams.empty()) {
      buckets_.push_back(createBucket(currentParams, device, dtype));
      currentParams.clear();
      currentBytes = 0;
    }

    currentParams.push_back(param);
    currentBytes += paramBytes;
  }

  if (!currentParams.empty()) {
    buckets_.push_back(createBucket(currentParams, device, dtype));
  }
}

Reducer::Bucket Reducer::createBucket(const std::vector<TensorPtr>& params, Device device, DType dtype) {
  Bucket bucket;
  bucket.params = params;

  int64_t offset = 0;
  for (auto& p : bucket.params) {
    int64_t size = p->numel();
    bucket.paramSizes.push_back(size);
    bucket.paramOffsets.push_back(offset);
    offset += size;
    bucket.totalSize += size;
  }

  if (bucket.totalSize > 0) {
    bucket.flatBuffer = Tensor::zeros({bucket.totalSize}, Options(device, dtype));
  }

  return bucket;
}

void Reducer::registerHooks() {
  for (int64_t bIdx = 0; bIdx < static_cast<int64_t>(buckets_.size()); bIdx++) {
    auto& bucket = buckets_[bIdx];
    for (int64_t pIdx = 0; pIdx < static_cast<int64_t>(bucket.params.size()); pIdx++) {
      auto& param = bucket.params[pIdx];
      ASSERT(param != nullptr);

      auto ctx = std::make_unique<ParameterHookCtx>();
      ctx->self = this;
      ctx->bIdx = bIdx;
      ctx->pIdx = pIdx;
      param->registerHook(&parameterHookFn, ctx.get());
      hookCtxs_.push_back(std::move(ctx));
    }
  }
}

void Reducer::onGradReady(int64_t bucketIdx, int64_t paramIdx, const Tensor& grad) {
  ASSERT(bucketIdx >= 0 && bucketIdx < static_cast<int64_t>(buckets_.size()));
  auto& bucket = buckets_[bucketIdx];

  ASSERT(bucket.flatBuffer.defined());
  ASSERT(paramIdx >= 0 && paramIdx < static_cast<int64_t>(bucket.params.size()));

  int64_t offset = bucket.paramOffsets[paramIdx];
  int64_t size = bucket.paramSizes[paramIdx];
  bucket.flatBuffer.narrow(0, offset, size).copy_(grad.flatten());
  bucket.readyCount++;

  ASSERT(!bucket.reduceStarted);
  if (bucket.readyCount == static_cast<int64_t>(bucket.params.size())) {
    bucket.reduceStarted = true;
    reduceBucket(bucketIdx);
  }
}

void Reducer::reduceBucket(int64_t bucketIdx) {
  ASSERT(bucketIdx >= 0 && bucketIdx < static_cast<int64_t>(buckets_.size()));
  auto& bucket = buckets_[bucketIdx];

  ASSERT(bucket.flatBuffer.defined());
  ASSERT(!bucket.params.empty());

  std::vector<Tensor> tensors = {bucket.flatBuffer};
  bucket.work = processGroup_->allReduce(tensors);
  ASSERT(bucket.work);

  bucketReadyCnt_++;
  checkAllBucketsReady();
}

void Reducer::prepareForBackward() {
  bucketReadyCnt_ = 0;
  for (auto& bucket : buckets_) {
    bucket.readyCount = 0;
    bucket.reduceStarted = false;
    bucket.work = nullptr;
  }
}

void Reducer::broadcastParameters(int rootRank) const {
  for (int64_t bIdx = 0; bIdx < static_cast<int64_t>(buckets_.size()); bIdx++) {
    auto& bucket = buckets_[bIdx];

    copyParamsToFlattenedBuffer(bIdx);

    ASSERT(bucket.flatBuffer.defined());
    std::vector<Tensor> tensors = {bucket.flatBuffer};
    auto work = processGroup_->broadcast(tensors, rootRank);
    if (work) {
      work->wait();
    }

    copyFlattenedBufferToParams(bIdx);
  }
}

void Reducer::checkAllBucketsReady() {
  if (bucketReadyCnt_ < static_cast<int64_t>(buckets_.size())) {
    return;
  }

  for (int64_t bIdx = 0; bIdx < static_cast<int64_t>(buckets_.size()); bIdx++) {
    auto& bucket = buckets_[bIdx];
    if (bucket.work) {
      bool success = bucket.work->wait();
      ASSERT(success);

      cuda::CudaDeviceGuard guard(bucket.flatBuffer.device().index);
      auto worldSize = processGroup_->getWorldSize();
      bucket.flatBuffer /= static_cast<float>(worldSize);
      copyFlattenedBufferToGrads(bIdx);

      bucket.work = nullptr;
    }
  }
}

void Reducer::copyParamsToFlattenedBuffer(int64_t bucketIdx) const {
  ASSERT(bucketIdx >= 0 && bucketIdx < static_cast<int64_t>(buckets_.size()));
  auto& bucket = buckets_[bucketIdx];
  ASSERT(bucket.flatBuffer.defined());

  for (int64_t i = 0; i < static_cast<int64_t>(bucket.params.size()); i++) {
    ASSERT(bucket.params[i] != nullptr);
    ASSERT(bucket.params[i]->defined());
    bucket.flatBuffer.narrow(0, bucket.paramOffsets[i], bucket.paramSizes[i]).copy_(bucket.params[i]->flatten());
  }
}

void Reducer::copyFlattenedBufferToParams(int64_t bucketIdx) const {
  ASSERT(bucketIdx >= 0 && bucketIdx < static_cast<int64_t>(buckets_.size()));
  auto& bucket = buckets_[bucketIdx];
  ASSERT(bucket.flatBuffer.defined());

  for (int64_t i = 0; i < static_cast<int64_t>(bucket.params.size()); i++) {
    ASSERT(bucket.params[i] != nullptr);
    ASSERT(bucket.params[i]->defined());
    auto slice = bucket.flatBuffer.narrow(0, bucket.paramOffsets[i], bucket.paramSizes[i]);
    bucket.params[i]->copy_(slice.view(bucket.params[i]->sizes()));
  }
}

void Reducer::copyGradsToFlattenedBuffer(int64_t bucketIdx) const {
  ASSERT(bucketIdx >= 0 && bucketIdx < static_cast<int64_t>(buckets_.size()));
  auto& bucket = buckets_[bucketIdx];
  ASSERT(bucket.flatBuffer.defined());
  for (int64_t i = 0; i < static_cast<int64_t>(bucket.params.size()); i++) {
    ASSERT(bucket.params[i] != nullptr);
    ASSERT(bucket.params[i]->defined());
    auto& grad = bucket.params[i]->grad();
    ASSERT(grad.defined());
    bucket.flatBuffer.narrow(0, bucket.paramOffsets[i], bucket.paramSizes[i]).copy_(grad.flatten());
  }
}

void Reducer::copyFlattenedBufferToGrads(int64_t bucketIdx) const {
  ASSERT(bucketIdx >= 0 && bucketIdx < static_cast<int64_t>(buckets_.size()));
  auto& bucket = buckets_[bucketIdx];
  ASSERT(bucket.flatBuffer.defined());
  for (int64_t i = 0; i < static_cast<int64_t>(bucket.params.size()); i++) {
    ASSERT(bucket.params[i] != nullptr);
    ASSERT(bucket.params[i]->defined());
    auto& grad = bucket.params[i]->grad();
    ASSERT(grad.defined());
    auto slice = bucket.flatBuffer.narrow(0, bucket.paramOffsets[i], bucket.paramSizes[i]);
    grad.copy_(slice.view(grad.sizes()));
  }
}

}  // namespace tinytorch::distributed