/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <algorithm>
#include <numeric>

#include "OpFilling.h"
#include "OpTransform.h"
#include "Utils/Logger.h"
#include "Utils/RandomGenerator.h"

namespace tinytorch::op {

template <typename T>
Tensor permuteOpCpuImpl(const Tensor& self, const IntArrayView dims) {
  ASSERT(static_cast<int64_t>(dims.size()) == self.dim());
  auto retShape = SizeVector(self.shape());
  reorderIndices(retShape.data(), self.dim(), dims);
  auto ret = Tensor::empty(retShape, self.options().noGrad());

  const T* selfPtr = self.dataPtr<T>();
  T* retPtr = ret.dataPtr<T>();

  for (auto i = 0; i < self.numel(); i++) {
    int64_t originIndex = 0;
    int64_t offset = i;
    for (int64_t d = 0; d < self.dim(); d++) {
      originIndex += (offset / ret.stride(d)) * self.stride(dims[d]);
      offset %= ret.stride(d);
    }
    retPtr[i] = selfPtr[originIndex];
  }
  return ret;
}

template <typename T>
Tensor permuteAllOpCpuImpl(const Tensor& self) {
  SizeVector dims(self.dim());
  std::iota(dims.begin(), dims.end(), 0);
  std::reverse(dims.begin(), dims.end());
  return permuteOpCpuImpl<T>(self, dims);
}

template <typename T>
Tensor transposeOpCpuImpl(const Tensor& self, int64_t dim0, int64_t dim1) {
  if (dim0 < 0) {
    dim0 += self.dim();
  }
  if (dim1 < 0) {
    dim1 += self.dim();
  }
  if (dim0 < 0 || dim0 >= self.dim() || dim1 < 0 || dim1 >= self.dim()) {
    LOGE("Invalid axis: (%lld, %lld)", dim0, dim1);
    ASSERT(false);
    return {};
  }

  SizeVector dims(self.dim());
  std::iota(dims.begin(), dims.end(), 0);
  dims[dim0] = dim1;
  dims[dim1] = dim0;
  return permuteOpCpuImpl<T>(self, dims);
}

template <typename T>
Tensor transpose2dOpCpuImpl(const Tensor& self) {
  ASSERT(self.dim() == 2);
  return permuteOpCpuImpl<T>(self, {1, 0});
}

inline void getSubIndices(int64_t* subIndices, const Tensor& t, ArrayView<Tensor> indices, int64_t idx) {
  for (size_t i = 0; i < indices.size(); i++) {
    ASSERT(indices[i].dtype() == DType::Int64);
    auto* ptr = indices[i].dataPtr<int64_t>();
    auto ind = ptr[idx];
    subIndices[i] = ind >= 0 ? ind : ind + t.shape(static_cast<int64_t>(i));
  }
}

template <typename T, typename F>
void index2DCpuImpl(const Tensor& self, ArrayView<Tensor> indices, int64_t firstDim, F func) {
  auto dim0 = self.shape(0);
  auto dim1 = self.shape(1);
  ASSERT(indices[0].dtype() == DType::Int64);
  ASSERT(indices[1].dtype() == DType::Int64);
  auto* idx0Ptr = indices[0].dataPtr<int64_t>();
  auto* idx1Ptr = indices[1].dataPtr<int64_t>();
  for (int64_t i = 0; i < firstDim; i++) {
    int64_t idx0 = idx0Ptr[i];
    int64_t idx1 = idx1Ptr[i];
    if (idx0 < 0) idx0 += dim0;
    if (idx1 < 0) idx1 += dim1;
    int64_t dataIdx = idx0 * dim1 + idx1;
    func(i, dataIdx);
  }
}

template <typename T>
Tensor indexAdvanceOpCpuImpl(const Tensor& self, ArrayView<Tensor> indices) {
  auto len = static_cast<int64_t>(indices.size());
  auto firstDim = indices[0].numel();
  auto dimStride = self.stride(len - 1);
  SizeVector retShape(indices[0].shape());
  for (auto i = len; i < self.dim(); i++) {
    retShape.pushBack(self.shape(i));
  }
  auto ret = Tensor::empty(retShape, self.options().noGrad());
  const T* selfPtr = self.dataPtr<T>();
  T* retPtr = ret.dataPtr<T>();

  // 2D
  if (self.dim() == 2 && len == 2) {
    index2DCpuImpl<T>(self, indices, firstDim, [&](int64_t i, int64_t dataIdx) { retPtr[i] = selfPtr[dataIdx]; });
    return ret;
  }

  DimArray<int64_t> subIndices{};
  for (int64_t i = 0; i < firstDim; i++) {
    getSubIndices(subIndices.data, self, indices, i);
    int64_t dataIdx = indicesToOffset(self.strides(), subIndices.data);
    Storage::copyOnDevice(retPtr + (dimStride * i), selfPtr + dataIdx, dimStride * sizeof(T), self.device());
  }
  return ret;
}

template <typename T>
void indexPutAdvanceOpCpuImpl(Tensor& self, ArrayView<Tensor> indices, const Tensor& val) {
  auto len = static_cast<int64_t>(indices.size());
  auto firstDim = indices[0].numel();
  auto dimStride = self.stride(len - 1);

  T* selfPtr = self.dataPtr<T>();
  const T* valPtr = val.dataPtr<T>();

  // 2D
  if (self.dim() == 2 && len == 2) {
    if (val.isScalar()) {
      index2DCpuImpl<T>(self, indices, firstDim, [&](int64_t /*i*/, int64_t dataIdx) { selfPtr[dataIdx] = *valPtr; });
    } else {
      ASSERT(val.numel() == firstDim);
      index2DCpuImpl<T>(self, indices, firstDim, [&](int64_t i, int64_t dataIdx) { selfPtr[dataIdx] = valPtr[i]; });
    }
    return;
  }

  DimArray<int64_t> subIndices{};
  if (val.isScalar()) {
    for (int64_t i = 0; i < firstDim; i++) {
      getSubIndices(subIndices.data, self, indices, i);
      int64_t dataIdx = indicesToOffset(self.strides(), subIndices.data);
      op::fillOffset(self, val, dataIdx, dimStride);
    }
  } else {
    ASSERT(val.numel() == dimStride * firstDim);
    for (int64_t i = 0; i < firstDim; i++) {
      getSubIndices(subIndices.data, self, indices, i);
      int64_t dataIdx = indicesToOffset(self.strides(), subIndices.data);
      Storage::copyOnDevice(selfPtr + dataIdx, valPtr + (dimStride * i), dimStride * sizeof(T), self.device());
    }
  }
}

template <typename T, bool LOWER>
Tensor triangleOpCpuImpl(const Tensor& self, int64_t diagonal) {
  auto ret = Tensor::empty(self.shape(), self.options().noGrad());
  const auto rows = self.shape(0);
  const auto cols = self.shape(1);

  const T* selfPtr = self.dataPtr<T>();
  T* retPtr = ret.dataPtr<T>();

  int64_t idx = 0;
  for (auto i = 0; i < rows; i++) {
    idx = i * cols;
    for (auto j = 0; j < cols; j++) {
      if ((LOWER && j <= i + diagonal) || (!LOWER && j >= i + diagonal)) {
        retPtr[idx] = selfPtr[idx];
      } else {
        retPtr[idx] = 0;
      }
      idx++;
    }
  }
  return ret;
}

template <typename T>
Tensor trilOpCpuImpl(const Tensor& self, int64_t diagonal = 0) {
  return triangleOpCpuImpl<T, true>(self, diagonal);
}

template <typename T>
Tensor triuOpCpuImpl(const Tensor& self, int64_t diagonal = 0) {
  return triangleOpCpuImpl<T, false>(self, diagonal);
}

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
  ASSERT(self.dtype() == DType::Float32);

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

}  // namespace tinytorch::op
