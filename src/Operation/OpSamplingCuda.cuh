/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sort.h>

#include "OpSampling.h"
#include "Utils/CUDAUtils.h"
#include "Utils/RandomGenerator.h"

namespace tinytorch::op {

template <typename T>
__global__ void kPrepareProbabilities(const T* prob, float* floatProb, int64_t n, int64_t batch) {
  int64_t batchIdx = blockIdx.x;
  if (batchIdx >= batch) {
    return;
  }

  const T* batchProb = prob + batchIdx * n;
  float* batchFloatProb = floatProb + batchIdx * n;

  for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
    batchFloatProb[i] = static_cast<float>(batchProb[i]);
  }
}

inline __global__ void kMultinomialWithReplacement(const float* cdf, int64_t* output, int64_t batch, int64_t n,
                                                   int64_t numSamples, uint64_t seed, uint64_t seq) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t totalSamples = batch * numSamples;
  if (idx >= totalSamples) {
    return;
  }

  int64_t batchIdx = idx / numSamples;

  curandState state;
  curand_init(seed, idx, seq, &state);

  const float* batchCdf = cdf + batchIdx * n;
  float total = batchCdf[n - 1];

  if (total > 0) {
    float r = curand_uniform(&state) * total;
    int64_t left = 0, right = n - 1;
    while (left < right) {
      int64_t mid = left + (right - left) / 2;
      if (batchCdf[mid] < r)
        left = mid + 1;
      else
        right = mid;
    }
    output[idx] = left;
  } else {
    output[idx] = 0;
  }
}

template <typename T>
__global__ void kMultinomialNoReplacement(const T* prob, int64_t* output, float* workspace, int64_t batch, int64_t n,
                                          int64_t numSamples, uint64_t seed, uint64_t seq) {
  int64_t batchIdx = blockIdx.x;
  if (batchIdx >= batch) {
    return;
  }

  const T* batchProb = prob + batchIdx * n;
  float* batchWorkspace = workspace + batchIdx * n;

  for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
    batchWorkspace[i] = static_cast<float>(batchProb[i]);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    curandState state;
    curand_init(seed, batchIdx, seq, &state);

    for (int64_t s = 0; s < numSamples; s++) {
      float total = 0.f;
      for (int64_t i = 0; i < n; i++) {
        total += batchWorkspace[i];
      }

      int64_t outputIdx = batchIdx * numSamples + s;
      if (total > 0) {
        float r = curand_uniform(&state) * total;
        float cumSum = 0.f;
        int64_t selectedIdx = 0;
        for (int64_t i = 0; i < n; i++) {
          cumSum += batchWorkspace[i];
          if (cumSum >= r) {
            selectedIdx = i;
            break;
          }
        }
        output[outputIdx] = selectedIdx;
        batchWorkspace[selectedIdx] = 0.f;
      } else {
        output[outputIdx] = 0;
      }
    }
  }
}

template <typename T>
TensorPair topkOpCudaImpl(const Tensor& self, int64_t k, int64_t dim, bool largest, bool sorted) {
  if (dim < 0) dim += self.dim();
  ASSERT(dim >= 0 && dim < self.dim());

  int64_t n = self.shape(dim);
  ASSERT(k > 0 && k <= n);

  SizeVector retShape(self.shape());
  retShape[dim] = k;
  Tensor values = Tensor::empty(retShape, self.options().noGrad());
  Tensor indices = Tensor::empty(retShape, self.options().noGrad().indices());

  int64_t outerSize = 1;
  int64_t innerSize = 1;
  for (int64_t i = 0; i < dim; i++) {
    outerSize *= self.shape(i);
  }
  for (int64_t i = dim + 1; i < self.dim(); i++) {
    innerSize *= self.shape(i);
  }

  using CudaT = typename cuda::CudaTypeCast<T>::type;
  const CudaT* selfPtr = self.dataPtr<CudaT>();
  CudaT* valPtr = values.dataPtr<CudaT>();
  auto* idxPtr = indices.dataPtr<int64_t>();

  const auto stream = cuda::getCurrentCUDAStream(self.device().index).stream;
  auto policy = thrust::cuda::par.on(stream);

  Storage tmpValues(static_cast<int64_t>(n * sizeof(CudaT)), self.device());
  Storage tmpIndices(static_cast<int64_t>(n * sizeof(int64_t)), self.device());

  CudaT* tmpValuesPtr = tmpValues.dataPtr<CudaT>();
  auto* tmpIndicesPtr = tmpIndices.dataPtr<int64_t>();

  for (int64_t o = 0; o < outerSize; o++) {
    for (int64_t in = 0; in < innerSize; in++) {
      int64_t base = o * n * innerSize + in;

      // init values & indices
      thrust::device_ptr<CudaT> dstPtr = thrust::device_pointer_cast(tmpValuesPtr);
      auto inputIter =
          thrust::make_transform_iterator(thrust::counting_iterator<int64_t>(0),
                                          [=] __host__ __device__(int64_t i) { return selfPtr[base + i * innerSize]; });
      thrust::copy(policy, inputIter, inputIter + n, dstPtr);
      thrust::sequence(policy, thrust::device_pointer_cast(tmpIndicesPtr),
                       thrust::device_pointer_cast(tmpIndicesPtr + n), 0);

      // sort
      if (largest) {
        thrust::sort_by_key(policy, thrust::device_pointer_cast(tmpValuesPtr),
                            thrust::device_pointer_cast(tmpValuesPtr + n), thrust::device_pointer_cast(tmpIndicesPtr),
                            thrust::greater<CudaT>());
      } else {
        thrust::sort_by_key(policy, thrust::device_pointer_cast(tmpValuesPtr),
                            thrust::device_pointer_cast(tmpValuesPtr + n), thrust::device_pointer_cast(tmpIndicesPtr),
                            thrust::less<T>());
      }

      // copy results
      auto outputValIter = thrust::make_transform_iterator(
          thrust::counting_iterator<int64_t>(0),
          [=] __host__ __device__(int64_t i) -> CudaT& { return valPtr[(o * k + i) * innerSize + in]; });

      auto outputIdxIter = thrust::make_transform_iterator(
          thrust::counting_iterator<int64_t>(0),
          [=] __host__ __device__(int64_t i) -> int64_t& { return idxPtr[(o * k + i) * innerSize + in]; });
      thrust::copy(policy, thrust::device_pointer_cast(tmpValuesPtr), thrust::device_pointer_cast(tmpValuesPtr + k),
                   outputValIter);
      thrust::copy(policy, thrust::device_pointer_cast(tmpIndicesPtr), thrust::device_pointer_cast(tmpIndicesPtr + k),
                   outputIdxIter);
    }
  }

  return {values, indices};
}

template <typename T>
Tensor multinomialOpCudaImpl(const Tensor& self, int64_t nSamples, bool replacement) {
  ASSERT(self.dim() == 1 || self.dim() == 2);

  int64_t batch = (self.dim() == 2) ? self.shape(0) : 1;
  int64_t n = (self.dim() == 2) ? self.shape(1) : self.shape(0);

  SizeVector retShape;
  if (self.dim() == 2) {
    retShape = {batch, nSamples};
  } else {
    retShape = {nSamples};
  }

  Tensor ret = Tensor::empty(retShape, self.options().noGrad().indices());
  if (n == 0) {
    return ret;
  }

  using CudaT = typename cuda::CudaTypeCast<T>::type;
  const CudaT* selfPtr = self.dataPtr<CudaT>();
  auto* retPtr = ret.dataPtr<int64_t>();

  auto seed = RandomGeneratorCUDA::getSeed();
  auto seq = RandomGeneratorCUDA::nextSequence();

  const auto stream = cuda::getCurrentCUDAStream(self.device().index).stream;
  auto blockSize = cuda::getKernelBlockSize(self.device().index);

  Storage tmpProb(static_cast<int64_t>(batch * n * sizeof(float)), self.device());
  auto* tmpPtr = tmpProb.dataPtr<float>();

  if (nSamples == 1) {
    // faster path
    replacement = true;
  }

  if (replacement) {
    kPrepareProbabilities<<<batch, blockSize, 0, stream>>>(selfPtr, tmpPtr, n, batch);
    CUDA_KERNEL_CHECK();

    for (int64_t b = 0; b < batch; b++) {
      thrust::device_ptr<float> batchPtr(tmpPtr + b * n);
      thrust::inclusive_scan(thrust::cuda::par.on(stream), batchPtr, batchPtr + n, batchPtr);
    }

    int64_t totalSamples = batch * nSamples;
    auto gridSize = (totalSamples + blockSize - 1) / blockSize;
    kMultinomialWithReplacement<<<gridSize, blockSize, 0, stream>>>(tmpPtr, retPtr, batch, n, nSamples, seed, seq);
    CUDA_KERNEL_CHECK();
  } else {
    // TODO optimize
    kMultinomialNoReplacement<<<batch, blockSize, 0, stream>>>(selfPtr, retPtr, tmpPtr, batch, n, nSamples, seed, seq);
    CUDA_KERNEL_CHECK();
  }
  return ret;
}

template <typename T>
TensorPair sortOpCudaImpl(const Tensor& self, int64_t dim, bool descending) {
  if (dim < 0) {
    dim += self.dim();
  }
  ASSERT(dim >= 0 && dim < self.dim());

  SizeVector retShape(self.shape());
  Tensor values = Tensor::empty(retShape, self.options().noGrad());
  Tensor indices = Tensor::empty(retShape, self.options().noGrad().indices());

  int64_t outer = 1, inner = 1, n = self.shape(dim);
  for (int64_t i = 0; i < dim; i++) {
    outer *= self.shape(i);
  }
  for (int64_t i = dim + 1; i < self.dim(); i++) {
    inner *= self.shape(i);
  }

  using CudaT = typename cuda::CudaTypeCast<T>::type;
  const CudaT* selfPtr = self.dataPtr<CudaT>();
  CudaT* valPtr = values.dataPtr<CudaT>();
  auto* idxPtr = indices.dataPtr<int64_t>();

  const auto stream = cuda::getCurrentCUDAStream(self.device().index).stream;

  thrust::copy(thrust::cuda::par.on(stream), selfPtr, selfPtr + self.numel(), valPtr);

  auto initIndices = [=] __host__ __device__(int64_t idx) {
    int64_t i = (idx % (n * inner)) / inner;
    return i;
  };
  thrust::transform(thrust::cuda::par.on(stream), thrust::make_counting_iterator<int64_t>(0),
                    thrust::make_counting_iterator<int64_t>(outer * n * inner), thrust::device_pointer_cast(idxPtr),
                    initIndices);

  for (int64_t seg = 0; seg < outer * inner; seg++) {
    int64_t o = seg / inner;
    int64_t in = seg % inner;

    CudaT* valSegmentStart = valPtr + o * n * inner + in;
    int64_t* idxSegmentStart = idxPtr + o * n * inner + in;

    auto valIter = thrust::make_permutation_iterator(
        thrust::device_pointer_cast(valSegmentStart),
        thrust::make_transform_iterator(thrust::make_counting_iterator<int64_t>(0),
                                        [inner] __host__ __device__(int64_t i) { return i * inner; }));
    auto idxIter = thrust::make_permutation_iterator(
        thrust::device_pointer_cast(idxSegmentStart),
        thrust::make_transform_iterator(thrust::make_counting_iterator<int64_t>(0),
                                        [inner] __host__ __device__(int64_t i) { return i * inner; }));

    if (descending) {
      thrust::stable_sort_by_key(thrust::cuda::par.on(stream), valIter, valIter + n, idxIter, thrust::greater<T>());
    } else {
      thrust::stable_sort_by_key(thrust::cuda::par.on(stream), valIter, valIter + n, idxIter, thrust::less<T>());
    }
  }

  return {values, indices};
}

template <typename T>
Tensor cumsumOpCudaImpl(const Tensor& self, int64_t dim) {
  if (dim < 0) {
    dim += self.dim();
  }
  ASSERT(dim >= 0 && dim < self.dim());

  using CudaT = typename cuda::CudaTypeCast<T>::type;
  Tensor ret = Tensor::empty(self.shape(), self.options().noGrad());
  const CudaT* selfPtr = self.dataPtr<CudaT>();
  CudaT* retPtr = ret.dataPtr<CudaT>();

  int64_t outer = 1, inner = 1, n = self.shape(dim);
  for (int64_t i = 0; i < dim; i++) {
    outer *= self.shape(i);
  }
  for (int64_t i = dim + 1; i < self.dim(); i++) {
    inner *= self.shape(i);
  }

  const auto stream = cuda::getCurrentCUDAStream(self.device().index).stream;

  if (inner == 1) {
    for (int64_t o = 0; o < outer; ++o) {
      int64_t offset = o * n;
      thrust::inclusive_scan(thrust::cuda::par.on(stream), selfPtr + offset, selfPtr + offset + n, retPtr + offset);
    }
  } else {
    thrust::for_each_n(thrust::cuda::par.on(stream), thrust::counting_iterator<int64_t>(0), outer * inner,
                       [=] __host__ __device__(int64_t row) {
                         int64_t o = row / inner;
                         int64_t inIdx = row % inner;
                         int64_t base = o * n * inner + inIdx;

                         CudaT sum = 0;
                         for (int64_t i = 0; i < n; i++) {
                           sum += selfPtr[base + i * inner];
                           retPtr[base + i * inner] = sum;
                         }
                       });
  }
  return ret;
}

}  // namespace tinytorch::op
