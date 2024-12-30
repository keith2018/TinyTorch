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
                   int n);

  /**
   * c = a * b.T
   */
  static void gemmTrans(float *c, const float *a, const float *b, int m, int k,
                        int n);

 private:
  static void gemmCPU(float *c, const float *a, const float *b, int m, int k,
                      int n, bool trans);
};

}  // namespace TinyTorch
