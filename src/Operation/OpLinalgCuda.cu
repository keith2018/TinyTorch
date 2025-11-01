/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>

#include "OpLinalgCuda.cuh"

namespace tinytorch::op {

template <typename T>
Tensor dotOpCudaImpl(const Tensor& self, const Tensor& other) {
  using CudaT = typename cuda::CudaTypeCast<T>::type;
  auto ret = Tensor::scalar(0, self.options().noGrad());

  int64_t n = self.numel();
  const auto& stream = cuda::getCurrentCUDAStream(self.device().index).stream();

  thrust::device_ptr<const CudaT> selfPtr(self.dataPtr<CudaT>());
  thrust::device_ptr<const CudaT> otherPtr(other.dataPtr<CudaT>());
  thrust::device_ptr<CudaT> retPtr(ret.dataPtr<CudaT>());

  *retPtr = thrust::inner_product(thrust::cuda::par.on(stream), selfPtr, selfPtr + n, otherPtr, CudaT(0));
  return ret;
}

#define INSTANTIATE_DOT_OP(T) template Tensor dotOpCudaImpl<T>(const Tensor&, const Tensor&);
FOR_FLT_TYPES(INSTANTIATE_DOT_OP)
#undef INSTANTIATE_DOT_OP

#define REG_LINALG_CUDA_FLT(NAME, FUNC)                                         \
  REGISTER_OP_IMPL(NAME, CUDA, Float32, &(FUNC<DTypeToType_t<DType::Float32>>)) \
  REGISTER_OP_IMPL(NAME, CUDA, Float16, &(FUNC<DTypeToType_t<DType::Float16>>)) \
  REGISTER_OP_IMPL(NAME, CUDA, BFloat16, &(FUNC<DTypeToType_t<DType::BFloat16>>))

void registerLinalgCudaFloat() {
  // dot
  REG_LINALG_CUDA_FLT(dot, dotOpCudaImpl);

  // matmul
  REG_LINALG_CUDA_FLT(im2col, im2colOpCudaImpl);
  REG_LINALG_CUDA_FLT(col2im, col2imOpCudaImpl);
}

void registerLinalgCuda() { registerLinalgCudaFloat(); }

}  // namespace tinytorch::op