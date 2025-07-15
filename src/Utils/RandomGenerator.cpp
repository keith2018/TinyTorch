/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "RandomGenerator.h"

namespace tinytorch {

static std::random_device _r;

std::optional<unsigned long> RandomGeneratorCPU::seed_;
std::default_random_engine RandomGeneratorCPU::randomEngine_(_r());

unsigned long RandomGeneratorCUDA::seed_ = _r();
unsigned long RandomGeneratorCUDA::sequence_ = 0;

void manualSeed(unsigned long seed) {
  RandomGeneratorCPU::setSeed(seed);
  RandomGeneratorCUDA::setSeed(seed);
}

}  // namespace tinytorch