/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <fstream>

#include "Data.h"
#include "Function.h"
#include "Logger.h"
#include "Loss.h"
#include "Module.h"
#include "Optimizer.h"
#include "Scheduler.h"
#include "Tensor.h"
#include "Timer.h"

namespace TinyTorch {

constexpr float PI = 3.1415926535f;

void manualSeed(unsigned long seed);

void print(const Tensor& tensor, bool full = false);

void save(const Tensor& tensor, std::ofstream& ofs);

void load(Tensor& tensor, std::ifstream& ifs);

void save(nn::Module& model, const char* path);

void load(nn::Module& model, const char* path);

}  // namespace TinyTorch
