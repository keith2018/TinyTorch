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

  return 0;
}
