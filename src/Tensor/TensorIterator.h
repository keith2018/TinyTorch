/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <algorithm>
#include <tuple>
#include <utility>

#include "Tensor.h"

namespace tinytorch {

class TensorIteratorBase {
 public:
  template <typename... Tensors>
  explicit TensorIteratorBase(const Tensors&... inputs) : inputs_{&inputs...} {}

  IntArrayView setupBroadcast();

  bool isBroadcastOk() const { return broadcastOk_; }

 protected:
  static bool broadcastShape(SizeVector& ret, const SmallVector<IntArrayView>& shapes, int64_t ndim);
  static void alignShape(SizeVector& ret, const Tensor& t, int64_t ndim);
  static void alignStrides(SizeVector& ret, const Tensor& t, int64_t ndim);

  SmallVector<const Tensor*> inputs_;
  SmallVector<SizeVector> shapes_;
  SmallVector<SizeVector> strides_;
  SmallVector<bool> isScalar_;

  int64_t ndim_ = 0;
  int64_t total_ = 0;
  SizeVector shape_;
  bool broadcastOk_ = false;
  bool needBroadcast_ = false;
  bool allNonScalarSameShape_ = false;
};

template <typename... Tensors>
class TensorIteratorCpu : public TensorIteratorBase {
 public:
  explicit TensorIteratorCpu(const Tensors&... inputs) : TensorIteratorBase(inputs...) {}

  static constexpr size_t nInputs = sizeof...(Tensors);

  template <typename OutType, typename InType = OutType, typename Func>
  void forEach(Tensor& out, Func f) {
    forEachImpl<OutType, InType, Func>(out, f, std::make_index_sequence<nInputs>());
  }

 private:
  template <typename Func, typename Indexer, typename... DataPtrs, size_t... I>
  auto callWithIndices(Func& f, Indexer&& indexer, int64_t i, std::index_sequence<I...>, DataPtrs... dataPtrs) {
    auto tup = std::forward_as_tuple(dataPtrs...);
    return f((std::get<I>(tup)[indexer(i, I)])...);
  }

  template <typename Func, typename... DataPtrs, size_t... I>
  auto callWithOffsets(Func& f, const SizeVector& offsets, std::index_sequence<I...>, DataPtrs... dataPtrs) {
    auto tup = std::forward_as_tuple(dataPtrs...);
    return f((std::get<I>(tup)[offsets[I]])...);
  }

  template <typename OutType, typename Func, typename Indexer, typename... DataPtrs>
  void forEachGeneric(OutType* outPtr, Func& f, Indexer&& indexer, int64_t total, DataPtrs... dataPtrs) {
    for (int64_t i = 0; i < total; i++) {
      *outPtr++ = static_cast<OutType>(
          callWithIndices(f, indexer, i, std::make_index_sequence<sizeof...(DataPtrs)>(), dataPtrs...));
    }
  }

  template <typename OutType, typename Func, typename... DataPtrs>
  void forEachBroadcast(OutType* outPtr, Func& f, DataPtrs... dataPtrs) {
    constexpr size_t nInputs = sizeof...(DataPtrs);
    SmallVector increment(nInputs, SizeVector(ndim_, 0));
    SizeVector stridesTmp(nInputs, 1);
    SizeVector dimSize(ndim_, 0);
    for (int64_t d = ndim_ - 1; d >= 0; d--) {
      dimSize[d] = shape_[d];
      for (size_t j = 0; j < nInputs; j++) {
        increment[j][d] = (strides_[j][d] == 0 ? 0 : stridesTmp[j]);
        stridesTmp[j] *= shapes_[j][d];
      }
    }

    SizeVector coord(ndim_, 0);
    SizeVector offsets(nInputs, 0);
    for (int64_t i = 0; i < total_; i++) {
      *outPtr++ = static_cast<OutType>(callWithOffsets(f, offsets, std::make_index_sequence<nInputs>(), dataPtrs...));

      for (int64_t d = ndim_ - 1; d >= 0; d--) {
        coord[d]++;
        for (size_t j = 0; j < nInputs; j++) {
          offsets[j] += increment[j][d];
        }
        if (coord[d] < dimSize[d]) {
          break;
        }
        for (size_t j = 0; j < nInputs; j++) {
          offsets[j] -= coord[d] * increment[j][d];
        }
        coord[d] = 0;
      }
    }
  }

  template <typename OutType, typename InType, typename Func, size_t... I>
  void forEachImpl(Tensor& out, Func f, std::index_sequence<I...>) {
    OutType* outPtr = out.dataPtr<OutType>();

    // all scalar
    if (std::all_of(isScalar_.begin(), isScalar_.end(), [](bool s) { return s; })) {
      auto indexer = [](int64_t, size_t) { return 0; };
      forEachGeneric(outPtr, f, indexer, total_, inputs_[I]->template dataPtr<InType>()...);
      return;
    }

    // all same shape (exclude scalars)
    if (allNonScalarSameShape_) {
      if (std::all_of(isScalar_.begin(), isScalar_.end(), [](bool s) { return !s; })) {
        // no scalar
        auto indexer = [](int64_t i, size_t) { return i; };
        forEachGeneric(outPtr, f, indexer, total_, inputs_[I]->template dataPtr<InType>()...);
      } else {
        auto indexer = [this](int64_t i, size_t j) { return isScalar_[j] ? 0 : i; };
        forEachGeneric(outPtr, f, indexer, total_, inputs_[I]->template dataPtr<InType>()...);
      }
      return;
    }

    // broadcast
    forEachBroadcast<OutType, Func>(outPtr, f, inputs_[I]->template dataPtr<InType>()...);
  }
};

}  // namespace tinytorch
