/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "OpReduce.h"

namespace tinytorch::op {

SizeVector getReduceShape(const Tensor &t, int64_t dim, bool keepDims) {
  SizeVector retShape;
  retShape.reserve(t.dim());
  for (auto d = 0; d < t.dim(); d++) {
    if (d == dim) {
      if (keepDims) {
        retShape.pushBack(1);
      }
    } else {
      retShape.pushBack(t.shape(d));
    }
  }
  return retShape;
}

SizeVector getReduceShape(const Tensor &t, const DimArray<int64_t> &inAxis, bool keepDims) {
  SizeVector retShape;
  retShape.reserve(t.dim());
  for (auto d = 0; d < t.dim(); d++) {
    if (inAxis.data[d] != 0) {
      if (keepDims) {
        retShape.pushBack(1);
      }
    } else {
      retShape.pushBack(t.shape(d));
    }
  }
  return retShape;
}

Options getIndicesOptions(const Tensor &t) {
  Options options = t.options();
  options.dtype(DType::Int64);
  return options;
}

}  // namespace tinytorch::op