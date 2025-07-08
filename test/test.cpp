/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "test.h"

#include "TinyTorch.h"

int main(int argc, char* argv[]) {
#ifdef DEFAULT_DEVICE_CUDA
  tinytorch::setDefaultDevice({tinytorch::DeviceType::CUDA, 0});
#else
  tinytorch::setDefaultDevice(tinytorch::DeviceType::CPU);
#endif

  InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
