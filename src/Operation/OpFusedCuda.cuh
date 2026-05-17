/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "OpElemWiseCuda.cuh"
#include "OpFused.h"
#include "OpNNLayerCuda.cuh"
#include "Utils/CUDAUtils.h"

namespace tinytorch::op {

template <typename T, int VEC_ELEMENTS>
__global__ void kSiluMulVec(T* __restrict__ outPtr, const T* __restrict__ selfPtr, const int d) {
  const unsigned int row = blockIdx.x;
  const T* gatePtr = selfPtr + static_cast<int64_t>(row) * d * 2;
  const T* upPtr = gatePtr + d;
  T* rowOut = outPtr + static_cast<int64_t>(row) * d;

  const int numVecs = d / VEC_ELEMENTS;

  const int4* gateVec = reinterpret_cast<const int4*>(gatePtr);
  const int4* upVec = reinterpret_cast<const int4*>(upPtr);
  int4* outVec = reinterpret_cast<int4*>(rowOut);

  for (auto i = threadIdx.x; i < numVecs; i += blockDim.x) {
    // 128-bit load via read-only cache (__ldg)
    int4 gv = __ldg(&gateVec[i]);
    int4 uv = __ldg(&upVec[i]);

    const T* g = reinterpret_cast<const T*>(&gv);
    const T* u = reinterpret_cast<const T*>(&uv);
    T r[VEC_ELEMENTS];

#pragma unroll
    for (int j = 0; j < VEC_ELEMENTS; ++j) {
      auto gf = static_cast<float>(g[j]);
      auto uf = static_cast<float>(u[j]);
      r[j] = static_cast<T>(gf / (1.f + __expf(-gf)) * uf);
    }

    outVec[i] = *reinterpret_cast<const int4*>(r);
  }
}

template <typename T>
__global__ void kSiluMulScalar(T* retPtr, const T* selfPtr, const int64_t halfLastDim, const int64_t n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    const int64_t sliceIdx = index / halfLastDim;
    const int64_t innerIdx = index % halfLastDim;

    const int64_t gateIdx = sliceIdx * halfLastDim * 2 + innerIdx;
    const int64_t upIdx = gateIdx + halfLastDim;

    const T gateVal = selfPtr[gateIdx];
    const T upVal = selfPtr[upIdx];
    retPtr[index] = OpCudaSilu::apply(gateVal) * upVal;
  }
}

template <typename T>
Tensor siluMulOpCudaImpl(const Tensor& self) {
  ASSERT(self.size(-1) % 2 == 0);
  SizeVector retShape = self.shape();
  retShape.back() /= 2;
  auto ret = Tensor::empty(retShape, self.options().noGrad());

  using CudaT = typename cuda::CudaTypeCast<T>::type;
  const CudaT* selfPtr = self.dataPtr<CudaT>();
  CudaT* retPtr = ret.dataPtr<CudaT>();

  const int64_t lastDim = self.size(-1);
  const int64_t halfLastDim = lastDim / 2;
  const int64_t n = ret.numel();
  const int64_t numRows = n / halfLastDim;

  constexpr int kVecBytes = 16;  // int4
  constexpr int kVecElements = kVecBytes / static_cast<int>(sizeof(CudaT));

  const bool useVec = (halfLastDim % kVecElements == 0);

  if (useVec) {
    const int d = static_cast<int>(halfLastDim);
    dim3 grid(static_cast<unsigned>(numRows));
    dim3 block(std::min(d / kVecElements, 1024));
    auto stream = cuda::getCurrentCUDAStream(self.device().index).stream();
    kSiluMulVec<CudaT, kVecElements><<<grid, block, 0, stream>>>(retPtr, selfPtr, d);
    CUDA_KERNEL_CHECK();
  } else {
    auto params = cuda::getKernelLaunchParams(self.device().index, n);
    CUDA_LAUNCH_KERNEL(kSiluMulScalar<CudaT>, params, retPtr, selfPtr, halfLastDim, n);
  }

  return ret;
}

template <typename T>
__global__ void kFusedAddRMSNorm(T* __restrict__ input, T* __restrict__ residual, const T* __restrict__ weight,
                                 int64_t dim, float eps) {
  const auto row = blockIdx.x;
  const auto tid = threadIdx.x;
  const auto base = row * dim;

  // add residual + accumulate sum‑of‑squares
  float sumSq = 0.f;
  for (auto i = tid; i < dim; i += blockDim.x) {
    float r = static_cast<float>(input[base + i]) + static_cast<float>(residual[base + i]);
    residual[base + i] = static_cast<T>(r);
    sumSq += r * r;
  }

  sumSq = cudaBlockReduce<float, OpCudaReduceSum>(sumSq, 0.f);
  float invRms = cuda::rsqrt(sumSq / static_cast<float>(dim) + eps);

  // normalize + affine
  for (auto i = tid; i < dim; i += blockDim.x) {
    auto r = static_cast<float>(residual[base + i]);
    input[base + i] = static_cast<T>(r * invRms * static_cast<float>(weight[i]));
  }
}

template <typename T>
void fusedAddRmsNormOpCudaImpl(Tensor& input, Tensor& residual, const Tensor& weight, float eps) {
  ASSERT(input.shape() == residual.shape());
  int64_t dim = input.size(-1);
  int64_t numRows = input.numel() / dim;

  using CudaT = typename cuda::CudaTypeCast<T>::type;
  CudaT* inputPtr = input.dataPtr<CudaT>();
  CudaT* residualPtr = residual.dataPtr<CudaT>();
  const CudaT* weightPtr = weight.dataPtr<CudaT>();

  auto stream = cuda::getCurrentCUDAStream(input.device().index).stream();
  dim3 blockSize(std::clamp(nextPow2(dim), 32u, 1024u));
  dim3 gridSize(numRows);
  kFusedAddRMSNorm<CudaT><<<gridSize, blockSize, 0, stream>>>(inputPtr, residualPtr, weightPtr, dim, eps);
  CUDA_KERNEL_CHECK();
}

}  // namespace tinytorch::op
