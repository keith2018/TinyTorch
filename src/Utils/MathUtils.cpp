/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "MathUtils.h"

namespace tinytorch {

unsigned int nextPow2(unsigned int v) {
  if (v == 0) return 1;
  --v;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  ++v;
  return v;
}

}  // namespace tinytorch