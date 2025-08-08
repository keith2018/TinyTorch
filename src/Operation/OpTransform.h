/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Tensor/Dispatch.h"

namespace tinytorch::op {

#define DTYPE_CAST_DISPATCH_CASE(T)                   \
  case DType::T:                                      \
    f.template operator()<DTypeToType_t<DType::T>>(); \
    break

template <typename F>
void dtypeCastDispatch(DType dtype, F&& f) {
  switch (dtype) {
    DTYPE_CAST_DISPATCH_CASE(Float32);
    DTYPE_CAST_DISPATCH_CASE(Float16);
    DTYPE_CAST_DISPATCH_CASE(BFloat16);
    DTYPE_CAST_DISPATCH_CASE(Int32);
    DTYPE_CAST_DISPATCH_CASE(Int64);
    DTYPE_CAST_DISPATCH_CASE(Bool);
    default:
      LOGE("Unknown DType");
  }
}

int64_t indicesToOffset(IntArrayView strides, const int64_t* indices);
void offsetToIndices(int64_t* indices, IntArrayView shape, int64_t offset);
void reorderIndices(int64_t* indices, int64_t ndim, IntArrayView order);

using CastOpFn = void (*)(Tensor& dst, const Tensor& src);

using ReshapeOpFn = Tensor (*)(const Tensor& self, IntArrayView shape);
using ReshapeOpInplaceFn = void (*)(Tensor& self, IntArrayView shape);

using FlattenOpFn = Tensor (*)(const Tensor& self, int64_t startDim, int64_t endDim);
using FlattenOpInplaceFn = void (*)(Tensor& self, int64_t startDim, int64_t endDim);

using UnflattenOpFn = Tensor (*)(const Tensor& self, int64_t dim, IntArrayView shape);
using UnflattenOpInplaceFn = void (*)(Tensor& self, int64_t dim, IntArrayView shape);

using SqueezeOpFn = Tensor (*)(const Tensor& self, IntArrayView dims);
using SqueezeOpInplaceFn = void (*)(Tensor& self, IntArrayView dims);

using UnsqueezeOpFn = Tensor (*)(const Tensor& self, int64_t dim);
using UnsqueezeOpInplaceFn = void (*)(Tensor& self, int64_t dim);

using PermuteOpFn = Tensor (*)(const Tensor& self, IntArrayView dims);
using PermuteAllOpFn = Tensor (*)(const Tensor& self);

using TransposeOpFn = Tensor (*)(const Tensor& self, int64_t dim0, int64_t dim1);
using Transpose2dOpFn = Tensor (*)(const Tensor& self);

using IndexOpFn = Tensor (*)(const Tensor& self, IntArrayView indices);
using IndexAdvanceOpFn = Tensor (*)(const Tensor& self, ArrayView<Tensor> indices);

using IndexPutOpFn = void (*)(Tensor& self, IntArrayView indices, const Tensor& val);
using IndexPutAdvanceOpFn = void (*)(Tensor& self, ArrayView<Tensor> indices, const Tensor& val);

using TriangularOpFn = Tensor (*)(const Tensor& self, int64_t diagonal);

using SplitOpFn = std::vector<Tensor> (*)(const Tensor& self, int64_t splitSize, int64_t dim);
using ConcatOpFn = Tensor (*)(ArrayView<Tensor> tensors, int64_t dim);
using StackOpFn = Tensor (*)(ArrayView<Tensor> tensors, int64_t dim);
using HStackOpFn = Tensor (*)(ArrayView<Tensor> tensors);
using VStackOpFn = HStackOpFn;

using NarrowOpFn = Tensor (*)(const Tensor& self, int64_t dim, int64_t start, int64_t length);

using TopkOpFn = TensorPair (*)(const Tensor& self, int64_t k, int64_t dim, bool largest, bool sorted);

using MultinomialOpFn = Tensor (*)(const Tensor& self, int64_t numSamples, bool replacement);

using SortOpFn = TensorPair (*)(const Tensor& self, int64_t dim, bool descending);

using CumsumOpFn = Tensor (*)(const Tensor& self, int64_t dim);

using GatherOpFn = Tensor (*)(const Tensor& self, int64_t dim, const Tensor& index);

using ScatterOpFn = Tensor (*)(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src);
using ScatterOpInplaceFn = void (*)(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src);

// dtype cast
DEFINE_OP(dtypeCast, CastOpFn)

// reshape/view
DEFINE_OP(reshape, ReshapeOpFn)
DEFINE_OP(reshapeInplace, ReshapeOpInplaceFn)
DEFINE_OP(view, ReshapeOpFn)

// flatten
DEFINE_OP(flatten, FlattenOpFn)
DEFINE_OP(flattenInplace, FlattenOpInplaceFn)

// unflatten
DEFINE_OP(unflatten, UnflattenOpFn)
DEFINE_OP(unflattenInplace, UnflattenOpInplaceFn)

// squeeze
DEFINE_OP(squeeze, SqueezeOpFn)
DEFINE_OP(squeezeInplace, SqueezeOpInplaceFn)

// unsqueeze
DEFINE_OP(unsqueeze, UnsqueezeOpFn)
DEFINE_OP(unsqueezeInplace, UnsqueezeOpInplaceFn)

// permute
DEFINE_OP(permute, PermuteOpFn)
DEFINE_OP(permuteAll, PermuteAllOpFn)

// transpose
DEFINE_OP(transpose, TransposeOpFn)
DEFINE_OP(transpose2d, Transpose2dOpFn)

// index
DEFINE_OP(index, IndexOpFn)
DEFINE_OP(indexAdvance, IndexAdvanceOpFn)

// indexPut
DEFINE_OP(indexPut, IndexPutOpFn)
DEFINE_OP(indexPutAdvance, IndexPutAdvanceOpFn)

// tril/triu
DEFINE_OP(tril, TriangularOpFn)
DEFINE_OP(triu, TriangularOpFn)

// split
DEFINE_OP(split, SplitOpFn)

// concat
DEFINE_OP(concat, ConcatOpFn)

// stack
DEFINE_OP(stack, StackOpFn)
DEFINE_OP(vstack, VStackOpFn)
DEFINE_OP(hstack, HStackOpFn)

// narrow
DEFINE_OP(narrow, NarrowOpFn)

// topk
DEFINE_OP(topk, TopkOpFn)

// multinomial
DEFINE_OP(multinomial, MultinomialOpFn)

// sort
DEFINE_OP(sort, SortOpFn)

// cumsum
DEFINE_OP(cumsum, CumsumOpFn)

// gather
DEFINE_OP(gather, GatherOpFn)

// scatter
DEFINE_OP(scatter, ScatterOpFn)
DEFINE_OP(scatterInplace, ScatterOpInplaceFn)

void registerTransformCommon();
STATIC_CALL(registerTransformCommon);

void registerTransformCpu();
STATIC_CALL(registerTransformCpu);

#ifdef USE_CUDA
void registerTransformCuda();
STATIC_CALL(registerTransformCuda);
#endif

}  // namespace tinytorch::op
