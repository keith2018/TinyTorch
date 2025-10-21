/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

void demo_autograd();
void demo_module();
void demo_optim();
void demo_mnist();

#ifdef USE_NCCL
void demo_nccl(int argc, char **argv);
void demo_ddp(int argc, char **argv);
#endif
