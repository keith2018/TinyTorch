/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Tensor.h"

namespace tinytorch {

namespace nn {
class Module;
}  // namespace nn

void save(const Tensor& tensor, std::ofstream& ofs);

void load(Tensor& tensor, std::ifstream& ifs);

void save(nn::Module& model, const char* path);

void load(nn::Module& model, const char* path);

}  // namespace tinytorch
