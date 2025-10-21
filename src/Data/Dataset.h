/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Tensor.h"

namespace tinytorch::data {

class Dataset {
 public:
  virtual ~Dataset() = default;

  virtual size_t size() const = 0;
  virtual std::vector<Tensor> getItem(size_t idx) = 0;
};

}  // namespace tinytorch::data
