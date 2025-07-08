/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Initializer.h"

#include "Utils/Logger.h"

namespace tinytorch::nn {

void Initializer::uniform(Tensor &tensor, float min, float max) {
  ASSERT(tensor.dtype() == DType::Float32);
  tensor.fillUniform(min, max);
}

void Initializer::kaimingUniform(Tensor &tensor, float a, FanMode mode) {
  ASSERT(tensor.dtype() == DType::Float32);
  auto fan = calculateFan(tensor, mode);
  auto gain = calculateGain(a);
  auto std = gain / std::sqrt(static_cast<float>(fan));
  auto bound = std::sqrt(3.f) * std;
  uniform(tensor, -bound, bound);
}

std::pair<int64_t, int64_t> Initializer::calculateFan(const Tensor &tensor) {
  if (tensor.dim() < 2) {
    LOGE("Fan can not be computed for tensor with fewer than 2 dimensions");
    return std::make_pair(0, 0);
  }

  auto inputFMaps = tensor.shape()[1];
  auto outputFMaps = tensor.shape()[0];
  int64_t receptiveFieldSize = 1;
  if (tensor.dim() > 2) {
    for (int64_t i = 2; i < tensor.dim(); i++) {
      receptiveFieldSize *= tensor.shape()[i];
    }
  }
  inputFMaps *= receptiveFieldSize;
  outputFMaps *= receptiveFieldSize;
  return std::make_pair(inputFMaps, outputFMaps);
}

int64_t Initializer::calculateFan(const Tensor &tensor, FanMode mode) {
  auto [fanIn, fanOut] = calculateFan(tensor);
  switch (mode) {
    case FanMode::FAN_IN:
      return fanIn;
    case FanMode::FAN_OUT:
      return fanOut;
  }
  LOGE("calculateFan mode not support: %u", mode);
  return 0;
}

float Initializer::calculateGain(float param) { return std::sqrt(2.0f / (1.f + param * param)); }

}  // namespace tinytorch::nn
