/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <numeric>

#include "OpSampling.h"
#include "Utils/RandomGenerator.h"

namespace tinytorch::op {

template <typename T>
TensorPair topkOpCpuImpl(const Tensor& self, int64_t k, int64_t dim, bool largest, bool sorted) {
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

  const auto* selfPtr = self.dataPtr<T>();
  auto* valPtr = values.dataPtr<T>();
  auto* idxPtr = indices.dataPtr<int64_t>();

  std::function<bool(const std::pair<T, int64_t>&, const std::pair<T, int64_t>&)> cmp;
  if (largest) {
    cmp = [](const std::pair<T, int64_t>& a, const std::pair<T, int64_t>& b) { return a.first > b.first; };
  } else {
    cmp = [](const std::pair<T, int64_t>& a, const std::pair<T, int64_t>& b) { return a.first < b.first; };
  }

  for (int64_t o = 0; o < outerSize; o++) {
    for (int64_t in = 0; in < innerSize; in++) {
      int64_t base = o * n * innerSize + in;
      std::vector<std::pair<T, int64_t>> vec(n);
      for (int64_t i = 0; i < n; i++) {
        vec[i] = {selfPtr[base + i * innerSize], i};
      }
      std::partial_sort(vec.begin(), vec.begin() + k, vec.end(), cmp);
      if (sorted) {
        std::sort(vec.begin(), vec.begin() + k, cmp);
      }
      for (int64_t i = 0; i < k; i++) {
        valPtr[(o * k + i) * innerSize + in] = vec[i].first;
        idxPtr[(o * k + i) * innerSize + in] = vec[i].second;
      }
    }
  }
  return {values, indices};
}

template <typename T>
Tensor multinomialOpCpuImpl(const Tensor& self, int64_t numSamples, bool replacement) {
  ASSERT(self.dim() == 1 || self.dim() == 2);

  int64_t batch = (self.dim() == 2) ? self.shape(0) : 1;
  int64_t n = (self.dim() == 2) ? self.shape(1) : self.shape(0);

  SizeVector retShape;
  if (self.dim() == 2) {
    retShape = {batch, numSamples};
  } else {
    retShape = {numSamples};
  }

  Tensor ret = Tensor::empty(retShape, self.options().noGrad().indices());
  if (n == 0) {
    return ret;
  }

  const T* selfPtr = self.dataPtr<T>();
  auto* retPtr = ret.dataPtr<int64_t>();

  auto generator = RandomGeneratorCPU::getGenerator();
  std::uniform_real_distribution<float> dist(0.f, 1.f);

  for (int64_t b = 0; b < batch; b++) {
    std::vector<float> prob(n);
    if (self.dim() == 2) {
      for (int64_t i = 0; i < n; i++) {
        prob[i] = static_cast<float>(selfPtr[b * n + i]);
      }
    } else {
      for (int64_t i = 0; i < n; i++) {
        prob[i] = static_cast<float>(selfPtr[i]);
      }
    }

    for (int64_t s = 0; s < numSamples; s++) {
      std::vector<float> cdf(n);
      std::partial_sum(prob.begin(), prob.end(), cdf.begin());
      float total = cdf.back();
      ASSERT(total > 0);

      float r = dist(generator) * total;
      auto it = std::upper_bound(cdf.begin(), cdf.end(), r);
      int64_t idx = std::distance(cdf.begin(), it);

      if (self.dim() == 2) {
        retPtr[b * numSamples + s] = idx;
      } else {
        retPtr[s] = idx;
      }

      if (!replacement) {
        prob[idx] = 0;
      }
    }
  }
  return ret;
}

template <typename T>
TensorPair sortOpCpuImpl(const Tensor& self, int64_t dim, bool descending) {
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

  const T* selfPtr = self.dataPtr<T>();
  T* valPtr = values.dataPtr<T>();
  auto* idxPtr = indices.dataPtr<int64_t>();

  for (int64_t o = 0; o < outer; o++) {
    for (int64_t in = 0; in < inner; in++) {
      std::vector<std::pair<T, int64_t>> vec(n);
      for (int64_t i = 0; i < n; i++) {
        int64_t offset = o * n * inner + i * inner + in;
        vec[i] = {selfPtr[offset], i};
      }
      if (descending) {
        std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
      } else {
        std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
      }
      for (int64_t i = 0; i < n; i++) {
        int64_t offset = o * n * inner + i * inner + in;
        valPtr[offset] = vec[i].first;
        idxPtr[offset] = vec[i].second;
      }
    }
  }
  return {values, indices};
}

template <typename T>
Tensor cumsumOpCpuImpl(const Tensor& self, int64_t dim) {
  if (dim < 0) {
    dim += self.dim();
  }
  ASSERT(dim >= 0 && dim < self.dim());

  Tensor ret = Tensor::empty(self.shape(), self.options().noGrad());
  const T* selfPtr = self.dataPtr<T>();
  T* retPtr = ret.dataPtr<T>();

  int64_t outer = 1, inner = 1, n = self.shape(dim);
  for (int64_t i = 0; i < dim; i++) {
    outer *= self.shape(i);
  }
  for (int64_t i = dim + 1; i < self.dim(); i++) {
    inner *= self.shape(i);
  }

  for (int64_t o = 0; o < outer; o++) {
    for (int64_t in = 0; in < inner; in++) {
      T sum = 0;
      for (int64_t i = 0; i < n; i++) {
        int64_t offset = o * n * inner + i * inner + in;
        sum += selfPtr[offset];
        retPtr[offset] = sum;
      }
    }
  }
  return ret;
}

}  // namespace tinytorch::op
