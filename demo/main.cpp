/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "demo.h"

int main(int argc, char **argv) {
  demo_autograd();
  demo_module();
  demo_optim();
  demo_mnist();

#ifdef USE_NCCL
  demo_nccl(argc, argv);
  demo_ddp(argc, argv);
#endif

  return 0;
}