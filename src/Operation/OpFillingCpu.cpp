/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "OpFillingCpu.h"

#include "Utils/RandomGenerator.h"

namespace tinytorch::op {

void fillOpRandUniformCpuImpl(Tensor& self, float min, float max) {
  ASSERT(self.dtype() == DType::Float32);
  self.copyOnWrite();
  auto* selfPtr = self.dataPtr<float>();
  auto generator = RandomGeneratorCPU::getGenerator();
  std::uniform_real_distribution distribution(min, max);
  for (int64_t i = 0; i < self.numel(); i++) {
    selfPtr[i] = distribution(generator);
  }
}

void fillOpRandNormalCpuImpl(Tensor& self) {
  ASSERT(self.dtype() == DType::Float32);
  self.copyOnWrite();
  auto* selfPtr = self.dataPtr<float>();
  auto generator = RandomGeneratorCPU::getGenerator();
  std::normal_distribution distribution(0.0f, 1.0f);
  for (int64_t i = 0; i < self.numel(); i++) {
    selfPtr[i] = distribution(generator);
  }
}

void fillOpRandBernoulliCpuImpl(Tensor& self, float p) {
  ASSERT(self.dtype() == DType::Float32);
  self.copyOnWrite();
  auto* selfPtr = self.dataPtr<float>();
  auto generator = RandomGeneratorCPU::getGenerator();
  std::bernoulli_distribution distribution(p);
  for (int64_t i = 0; i < self.numel(); i++) {
    selfPtr[i] = distribution(generator);
  }
}

void registerFillCpuFloat32() {
  REGISTER_OP_IMPL(fillRandUniform, CPU, Float32, &fillOpRandUniformCpuImpl);
  REGISTER_OP_IMPL(fillRandNormal, CPU, Float32, &fillOpRandNormalCpuImpl);
  REGISTER_OP_IMPL(fillRandBernoulli, CPU, Float32, &fillOpRandBernoulliCpuImpl);
}

void registerFillCpu() {
  REGISTER_OP_IMPL_DTYPE_TPL(fill, CPU, fillOpCpuImpl);
  REGISTER_OP_IMPL_DTYPE_TPL(fillOffset, CPU, fillOpOffsetCpuImpl);

  REGISTER_OP_IMPL_DTYPE_TPL(fillMasked, CPU, fillOpMaskedCpuImpl);
  REGISTER_OP_IMPL_DTYPE_TPL(fillMaskedOut, CPU, fillOpMaskedOutCpuImpl);
  REGISTER_OP_IMPL_DTYPE_TPL(fillMaskedInplace, CPU, fillOpMaskedInplaceCpuImpl);

  REGISTER_OP_IMPL_DTYPE_TPL(fillLinSpace, CPU, fillOpLinSpaceCpuImpl);

  registerFillCpuFloat32();
}

}  // namespace tinytorch::op