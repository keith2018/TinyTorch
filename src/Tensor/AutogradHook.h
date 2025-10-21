/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

namespace tinytorch {

class Tensor;

struct Hook {
  using FnType = void (*)(void*, Tensor&);
  FnType fn;
  void* ctx;
};

}  // namespace tinytorch
