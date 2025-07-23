/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "FuncElemWise.h"

#include "Operations.h"

namespace tinytorch::function {

Tensor reduceGrad(const Tensor &self, const Tensor &grad) {
  auto ndim = std::max(self.dim(), grad.dim());
  SizeVector dims;
  dims.reserve(ndim);

  auto offsetSelf = ndim - self.dim();
  auto offsetGrad = ndim - grad.dim();

  for (auto i = 0; i < ndim; i++) {
    auto s = i < offsetSelf ? 1 : self.shape(i - offsetSelf);
    auto g = i < offsetGrad ? 1 : grad.shape(i - offsetGrad);
    if (s == 1 && g != 1) {
      dims.pushBack(i);
    }
  }

  if (dims.empty()) {
    return grad;
  }

  auto ret = op::sumOnDims(grad, dims, true);
  op::reshapeInplace(ret, self.shape());
  return ret;
}

}  // namespace tinytorch::function
