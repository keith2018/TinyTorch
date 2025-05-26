/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Torch.h"
#include "gtest/gtest.h"

int main(int argc, char *argv[]) {
#ifdef DEFAULT_DEVICE_CUDA
  tinytorch::setDefaultDevice(tinytorch::Device::CUDA);
#else
  tinytorch::setDefaultDevice(tinytorch::Device::CPU);
#endif

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
