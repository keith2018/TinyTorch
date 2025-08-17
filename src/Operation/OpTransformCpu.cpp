/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "OpTransformCpu.h"

namespace tinytorch::op {

template <typename SrcT, typename DstT>
void dtypeCastCpuKernel(const void* src, void* dst, int64_t numel) {
  const SrcT* srcPtr = static_cast<const SrcT*>(src);
  DstT* dstPtr = static_cast<DstT*>(dst);
  for (auto i = 0; i < numel; i++) {
    dstPtr[i] = static_cast<DstT>(srcPtr[i]);
  }
}

template <typename SrcT>
struct DTypeDstDispatchCpu {
  const Tensor& src;
  Tensor& dst;

  void operator()() const { dtypeCastDispatch(dst.dtype(), *this); }

  template <typename DstT>
  void operator()() const {
    dtypeCastCpuKernel<SrcT, DstT>(src.dataPtr(), dst.dataPtr(), src.numel());
  }
};

struct DTypeSrcDispatchCpu {
  const Tensor& src;
  Tensor& dst;

  template <typename SrcT>
  void operator()() const {
    DTypeDstDispatchCpu<SrcT> dstDispatch{src, dst};
    dtypeCastDispatch(dst.dtype(), dstDispatch);
  }
};

void dtypeCastOpCpuImpl(Tensor& dst, const Tensor& src) {
  ASSERT(src.numel() == dst.numel());
  DTypeSrcDispatchCpu srcDispatch{src, dst};
  dtypeCastDispatch(src.dtype(), srcDispatch);
}

void registerTransformCpu() {
  // dtype cast
  REGISTER_OP_IMPL_ALL_DTYPES(dtypeCast, CPU, dtypeCastOpCpuImpl);

  // permute
  REGISTER_OP_IMPL_DTYPE_TPL(permute, CPU, permuteOpCpuImpl);
  REGISTER_OP_IMPL_DTYPE_TPL(permuteAll, CPU, permuteAllOpCpuImpl);

  // transpose
  REGISTER_OP_IMPL_DTYPE_TPL(transpose, CPU, transposeOpCpuImpl);
  REGISTER_OP_IMPL_DTYPE_TPL(transpose2d, CPU, transpose2dOpCpuImpl);

  // indexAdvance
  REGISTER_OP_IMPL_DTYPE_TPL(indexAdvance, CPU, indexAdvanceOpCpuImpl);

  // indexPutAdvance
  REGISTER_OP_IMPL_DTYPE_TPL(indexPutAdvance, CPU, indexPutAdvanceOpCpuImpl);

  // tril/triu
  REGISTER_OP_IMPL_DTYPE_TPL(tril, CPU, trilOpCpuImpl);
  REGISTER_OP_IMPL_DTYPE_TPL(triu, CPU, triuOpCpuImpl);

  // gather
  REGISTER_OP_IMPL_DTYPE_TPL(gather, CPU, gatherOpCpuImpl)

  // scatter
  REGISTER_OP_IMPL_DTYPE_TPL(scatter, CPU, scatterOpCpuImpl)
  REGISTER_OP_IMPL_DTYPE_TPL(scatterInplace, CPU, scatterOpInplaceCpuImpl)

  // expand
  REGISTER_OP_IMPL_DTYPE_TPL(expand, CPU, expandOpCpuImpl)

  // indexSelect
  REGISTER_OP_IMPL_DTYPE_TPL(indexSelect, CPU, indexSelectOpCpuImpl)

  // repeatInterleave
  REGISTER_OP_IMPL_DTYPE_TPL(repeatInterleave, CPU, repeatInterleaveOpCpuImpl)
}

}  // namespace tinytorch::op