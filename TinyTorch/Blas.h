/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

namespace TinyTorch {

class Blas {
 public:
  /**
   * c = a * b
   */
  static void gemm(float *c, const float *a, const float *b, int m, int k,
                   int n, bool transA = false, bool transB = false);

 private:
  static void gemmCPU(float *c, const float *a, const float *b, int m, int k,
                      int n, bool transA = false, bool transB = false);
};

}  // namespace TinyTorch
