/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Tensor.h"

namespace tinytorch {

void print(const TensorImpl& tensor, const char* tag = nullptr, bool full = false);

void print(const Tensor& tensor, const char* tag = nullptr, bool full = false);

}  // namespace tinytorch
