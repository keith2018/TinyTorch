/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Blas.h"

#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include "cblas.h"
#endif
#endif

namespace TinyTorch {

void Blas::gemm(float *c, const float *a, const float *b, int m, int k, int n) {
#ifdef USE_BLAS
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a, k, b,
              n, 0.f, c, n);
#else
  gemmCPU(c, a, b, m, k, n, false);
#endif
}

void Blas::gemmTrans(float *c, const float *a, const float *b, int m, int k,
                     int n) {
#ifdef USE_BLAS
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0f, a, k, b,
              k, 0.f, c, n);
#else
  gemmCPU(c, a, b, m, k, n, true);
#endif
}

void Blas::gemmCPU(float *c, const float *a, const float *b, int m, int k,
                   int n, bool trans) {
  for (int i = 0; i < m * n; ++i) {
    c[i] = 0.0f;
  }

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int p = 0; p < k; ++p) {
        c[i * n + j] += a[i * k + p] * b[trans ? (j * k + p) : (p * n + j)];
      }
    }
  }
}

}  // namespace TinyTorch
