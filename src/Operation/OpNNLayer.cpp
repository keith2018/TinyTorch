/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "OpNNLayer.h"

namespace tinytorch::op {

SoftmaxDimInfo getSoftmaxDimInfo(const Tensor& self, int64_t dim) {
  const auto shape = self.shape();
  int64_t ndim = self.dim();
  if (dim < 0) {
    dim += ndim;
  }
  ASSERT(dim >= 0 && dim < ndim);
  int64_t outerSize = 1, innerSize = 1;
  for (int64_t i = 0; i < dim; i++) {
    outerSize *= shape[i];
  }
  for (int64_t i = dim + 1; i < ndim; i++) {
    innerSize *= shape[i];
  }
  return {outerSize, shape[dim], innerSize};
}

}  // namespace tinytorch::op
