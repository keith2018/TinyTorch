/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Tensor.h"

namespace TinyTorch::nn {

typedef enum {
  FAN_IN = 0,
  FAN_OUT = 1,
} FanMode;

class Init {
 public:
  static void uniform(Tensor &tensor, float min, float max);
  static void kaimingUniform(Tensor &tensor, float a = 0,
                             FanMode mode = FAN_IN);

  static std::pair<int32_t, int32_t> calculateFan(const Tensor &tensor);
  static int32_t calculateFan(const Tensor &tensor, FanMode mode);
  static float calculateGain(float param);
};

}  // namespace TinyTorch::nn
