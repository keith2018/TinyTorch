/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "OpTransformCuda.cuh"

namespace tinytorch::op {

template <typename SrcT, typename DstT>
__global__ void kDtypeCast(const SrcT* src, DstT* dst, int64_t numel) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    dst[idx] = static_cast<DstT>(src[idx]);
  }
}

template <typename SrcT, typename DstT>
void dtypeCastCudaKernelLauncher(const void* src, void* dst, int64_t numel, const Device& device) {
  const SrcT* srcPtr = static_cast<const SrcT*>(src);
  DstT* dstPtr = static_cast<DstT*>(dst);

  auto params = cuda::getKernelLaunchParams(device.index, numel);
  CUDA_LAUNCH_KERNEL((kDtypeCast<SrcT, DstT>), params, srcPtr, dstPtr, numel);
}

template <typename SrcT>
struct DTypeDstDispatchCuda {
  const Tensor& src;
  Tensor& dst;

  void operator()() const { dtypeCastDispatch(dst.dtype(), *this); }

  template <typename DstT>
  void operator()() const {
    dtypeCastCudaKernelLauncher<SrcT, DstT>(src.dataPtr(), dst.dataPtr(), src.numel(), src.device());
  }
};

struct DTypeSrcDispatchCuda {
  const Tensor& src;
  Tensor& dst;

  template <typename SrcT>
  void operator()() const {
    DTypeDstDispatchCuda<SrcT> dstDispatch{src, dst};
    dtypeCastDispatch(dst.dtype(), dstDispatch);
  }
};

void dtypeCastOpCudaImpl(Tensor& dst, const Tensor& src) {
  ASSERT(src.numel() == dst.numel());
  DTypeSrcDispatchCuda srcDispatch{src, dst};
  dtypeCastDispatch(src.dtype(), srcDispatch);
}

void registerTransformCuda() {
  // dtype cast
  REGISTER_OP_IMPL_ALL_DTYPES(dtypeCast, CUDA, dtypeCastOpCudaImpl);

  // permute
  REGISTER_OP_IMPL_DTYPE_TPL(permute, CUDA, permuteOpCudaImpl);
  REGISTER_OP_IMPL_DTYPE_TPL(permuteAll, CUDA, permuteAllOpCudaImpl);

  // transpose
  REGISTER_OP_IMPL_DTYPE_TPL(transpose, CUDA, transposeOpCudaImpl);
  REGISTER_OP_IMPL_DTYPE_TPL(transpose2d, CUDA, transpose2dOpCudaImpl);

  // indexAdvance
  REGISTER_OP_IMPL_DTYPE_TPL(indexAdvance, CUDA, indexAdvanceOpCudaImpl);

  // indexPutAdvance
  REGISTER_OP_IMPL_DTYPE_TPL(indexPutAdvance, CUDA, indexPutAdvanceOpCudaImpl);

  // tril/triu
  REGISTER_OP_IMPL_DTYPE_TPL(tril, CUDA, trilOpCudaImpl);
  REGISTER_OP_IMPL_DTYPE_TPL(triu, CUDA, triuOpCudaImpl);

  // topk
  REGISTER_OP_IMPL_DTYPE_TPL(topk, CUDA, topkOpCudaImpl);

  // multinomial
  REGISTER_OP_IMPL_DTYPE_TPL(multinomial, CUDA, multinomialOpCudaImpl);

  // sort
  REGISTER_OP_IMPL_DTYPE_TPL(sort, CUDA, sortOpCudaImpl)

  // cumsum
  REGISTER_OP_IMPL_DTYPE_TPL(cumsum, CUDA, cumsumOpCudaImpl)

  // gather
  REGISTER_OP_IMPL_DTYPE_TPL(gather, CUDA, gatherOpCudaImpl)

  // scatter
  REGISTER_OP_IMPL_DTYPE_TPL(scatter, CUDA, scatterOpCudaImpl)
  REGISTER_OP_IMPL_DTYPE_TPL(scatterInplace, CUDA, scatterOpInplaceCudaImpl)
}

}  // namespace tinytorch::op