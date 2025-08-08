/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Tensor.h"

namespace tinytorch::nn {

enum class FanMode : uint8_t {
  FAN_IN = 0,
  FAN_OUT = 1,
};

class Initializer {
 public:
  static void zeros(Tensor &tensor);
  static void ones(Tensor &tensor);
  static void normal(Tensor &tensor, float mean = 0.f, float stddev = 1.f);
  static void uniform(Tensor &tensor, float min, float max);
  static void kaimingUniform(Tensor &tensor, float a = 0, FanMode mode = FanMode::FAN_IN);

  static std::pair<int64_t, int64_t> calculateFan(const Tensor &tensor);
  static int64_t calculateFan(const Tensor &tensor, FanMode mode);
  static float calculateGain(float param);
};

}  // namespace tinytorch::nn
