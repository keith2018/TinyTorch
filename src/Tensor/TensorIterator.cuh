/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "TensorIterator.h"
#include "Utils/CUDAUtils.h"

namespace tinytorch {

template <size_t nInputs>
struct ALIGN(16) ScalarKernelParams {
  const void* inputs[nInputs];
};

template <size_t nInputs>
struct ALIGN(16) SameShapeKernelParams {
  const void* inputs[nInputs];
  bool isScalar[nInputs];
  int64_t total;
};

template <size_t nInputs>
struct ALIGN(16) BroadcastKernelParams {
  const void* inputs[nInputs];
  bool isScalar[nInputs];
  int64_t strides[nInputs * MAX_TENSOR_DIM];
  int64_t shape[MAX_TENSOR_DIM];
  int64_t total;
  int64_t ndim;
};

template <typename InType, typename Func, size_t... I>
__device__ __forceinline__ auto applyImpl(Func f, const InType* args, std::index_sequence<I...>) {
  return f(args[I]...);
}

template <typename InType, typename Func, size_t nInputs>
__device__ __forceinline__ auto apply(Func f, const InType* args) {
  return applyImpl(f, args, std::make_index_sequence<nInputs>());
}
template <typename OutType, typename InType, typename Func, size_t nInputs>
__global__ void scalarKernel(OutType* out, ScalarKernelParams<nInputs> params, Func f) {
  if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
    InType args[nInputs];
#pragma unroll
    for (auto j = 0; j < nInputs; j++) {
      const auto* input = static_cast<const InType*>(params.inputs[j]);
      args[j] = input[0];
    }
    out[0] = static_cast<OutType>(apply<InType, Func, nInputs>(f, args));
  }
}

template <typename OutType, typename InType, typename Func, size_t nInputs>
__global__ void sameShapeKernel(OutType* out, ScalarKernelParams<nInputs> params, int64_t total, Func f) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  InType args[nInputs];
#pragma unroll
  for (auto j = 0; j < nInputs; j++) {
    const auto* input = static_cast<const InType*>(params.inputs[j]);
    args[j] = input[idx];
  }
  out[idx] = static_cast<OutType>(apply<InType, Func, nInputs>(f, args));
}

template <typename OutType, typename InType, typename Func, size_t nInputs>
__global__ void sameShapeKernel(OutType* out, SameShapeKernelParams<nInputs> params, Func f) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= params.total) return;

  InType args[nInputs];
#pragma unroll
  for (auto j = 0; j < nInputs; j++) {
    const auto* input = static_cast<const InType*>(params.inputs[j]);
    args[j] = params.isScalar[j] ? input[0] : input[idx];
  }
  out[idx] = static_cast<OutType>(apply<InType, Func, nInputs>(f, args));
}

template <typename OutType, typename InType, typename Func, size_t nInputs>
__global__ void broadcastKernel(OutType* out, BroadcastKernelParams<nInputs> params, Func f) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= params.total) return;

  int64_t coord[MAX_TENSOR_DIM];
  int64_t remaining = idx;
#pragma unroll
  for (int64_t d = params.ndim - 1; d >= 0; --d) {
    coord[d] = remaining % params.shape[d];
    remaining /= params.shape[d];
  }

  InType args[nInputs];
#pragma unroll
  for (auto j = 0; j < nInputs; j++) {
    const auto* input = static_cast<const InType*>(params.inputs[j]);
    if (params.isScalar[j]) {
      args[j] = input[0];
    } else {
      int64_t offset = 0;
#pragma unroll
      for (auto d = 0; d < params.ndim; d++) {
        offset += coord[d] * params.strides[j * MAX_TENSOR_DIM + d];
      }
      args[j] = input[offset];
    }
  }
  out[idx] = static_cast<OutType>(apply<InType, Func, nInputs>(f, args));
}

template <typename... Tensors>
class TensorIteratorCuda : public TensorIteratorBase {
 public:
  explicit TensorIteratorCuda(const Tensors&... inputs) : TensorIteratorBase(inputs...) {}

  static constexpr size_t nInputs = sizeof...(Tensors);

  template <typename OutType, typename InType = OutType, typename Func>
  void forEach(Tensor& out, Func f) {
    OutType* outPtr = out.dataPtr<OutType>();

    if (std::all_of(isScalar_.begin(), isScalar_.end(), [](bool s) { return s; })) {
      // all scalar
      ScalarKernelParams<nInputs> params;
      fillInputs<ScalarKernelParams<nInputs>, InType>(params);
      auto kParams = cuda::getKernelLaunchParams(out.device().index, 1);
      CUDA_LAUNCH_KERNEL((scalarKernel<OutType, InType, Func, nInputs>), kParams, outPtr, params, f);
    } else if (allNonScalarSameShape_) {
      // all same shape (exclude scalars)
      if (std::all_of(isScalar_.begin(), isScalar_.end(), [](bool s) { return !s; })) {
        // no scalar
        ScalarKernelParams<nInputs> params;
        fillInputs<ScalarKernelParams<nInputs>, InType>(params);
        auto kParams = cuda::getKernelLaunchParams(out.device().index, total_);
        CUDA_LAUNCH_KERNEL((sameShapeKernel<OutType, InType, Func, nInputs>), kParams, outPtr, params, total_, f);
      } else {
        SameShapeKernelParams<nInputs> params;
        params.total = total_;
        fillInputs<SameShapeKernelParams<nInputs>, InType>(params);
        fillIsScalar(params);
        auto kParams = cuda::getKernelLaunchParams(out.device().index, params.total);
        CUDA_LAUNCH_KERNEL((sameShapeKernel<OutType, InType, Func, nInputs>), kParams, outPtr, params, f);
      }
    } else {
      // broadcast
      BroadcastKernelParams<nInputs> params;
      params.total = total_;
      params.ndim = ndim_;
      fillInputs<BroadcastKernelParams<nInputs>, InType>(params);
      fillIsScalar(params);
      fillShape(params);
      fillStrides(params);
      auto kParams = cuda::getKernelLaunchParams(out.device().index, params.total);
      CUDA_LAUNCH_KERNEL((broadcastKernel<OutType, InType, Func, nInputs>), kParams, outPtr, params, f);
    }
  }

 private:
  template <typename ParamType, typename InType>
  void fillInputs(ParamType& params) {
    for (auto i = 0; i < nInputs; i++) {
      params.inputs[i] = inputs_[i]->dataPtr<InType>();
    }
  }

  template <typename ParamType>
  void fillIsScalar(ParamType& params) {
    for (auto i = 0; i < nInputs; i++) {
      params.isScalar[i] = isScalar_[i];
    }
  }

  template <typename ParamType>
  void fillShape(ParamType& params) {
    for (auto d = 0; d < ndim_; d++) {
      params.shape[d] = shape_[d];
    }
  }

  template <typename ParamType>
  void fillStrides(ParamType& params) {
    for (auto j = 0; j < nInputs; j++) {
      for (auto d = 0; d < ndim_; d++) {
        params.strides[j * MAX_TENSOR_DIM + d] = strides_[j][d];
      }
      for (auto d = ndim_; d < MAX_TENSOR_DIM; d++) {
        params.strides[j * MAX_TENSOR_DIM + d] = 0;
      }
    }
  }
};

}  // namespace tinytorch
