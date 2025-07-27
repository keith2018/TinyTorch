/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Tensor/Dispatch.h"

namespace tinytorch::op {

SizeVector broadcastShape(IntArrayView t0, IntArrayView t1, int64_t skipLast);

template <typename T, DeviceType type>
void gemmImpl(T*, const T*, const T*, int64_t, int64_t, int64_t, bool, bool, DeviceIndex);

template <typename T>
using GemmFunc = void (*)(T*, const T*, const T*, int64_t, int64_t, int64_t, bool, bool, DeviceIndex);

template <typename T>
GemmFunc<T> getGemmFunc(DeviceType deviceType) {
  switch (deviceType) {
    case DeviceType::CPU:
      return &gemmImpl<T, DeviceType::CPU>;
#ifdef USE_CUDA
    case DeviceType::CUDA:
      return &gemmImpl<T, DeviceType::CUDA>;
#endif
    default:
      return nullptr;
  }
}

using DotOpFn = Tensor (*)(const Tensor& self, const Tensor& other);
using Im2ColOpFn = Tensor (*)(const Tensor& self, Dim2D kernel, Dim2D stride, Dim2D padding);
using Col2ImOpFn = Tensor (*)(const Tensor& self, IntArrayView shape, Dim2D kernel, Dim2D stride, Dim2D padding);
using MatmulOpFn = Tensor (*)(const Tensor& a, const Tensor& b, bool transA, bool transB);

// dot
DEFINE_OP(dot, DotOpFn)

// matmul
DEFINE_OP(im2col, Im2ColOpFn);
DEFINE_OP(col2im, Col2ImOpFn);
DEFINE_OP(matmul, MatmulOpFn)

void registerLinalgCommon();
STATIC_CALL(registerLinalgCommon);

void registerLinalgCpu();
STATIC_CALL(registerLinalgCpu);

#ifdef USE_CUDA
void registerLinalgCuda();
STATIC_CALL(registerLinalgCuda);
#endif

}  // namespace tinytorch::op
