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

void Blas::gemm(float *c, const float *a, const float *b, int m, int k, int n,
                bool transA, bool transB) {
#ifdef USE_BLAS
  cblas_sgemm(CblasRowMajor, transA ? CblasTrans : CblasNoTrans,
              transB ? CblasTrans : CblasNoTrans, m, n, k, 1.f, a,
              transA ? m : k, b, transB ? k : n, 0.f, c, n);
#else
  gemmCPU(c, a, b, m, k, n, transA, transB);
#endif
}

void Blas::gemmCPU(float *c, const float *a, const float *b, int m, int k,
                   int n, bool transA, bool transB) {
  for (int i = 0; i < m * n; ++i) {
    c[i] = 0.0f;
  }

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int p = 0; p < k; ++p) {
        float aVal = transA ? a[p * m + i] : a[i * k + p];
        float bVal = transB ? b[j * k + p] : b[p * n + j];
        c[i * n + j] += aVal * bVal;
      }
    }
  }
}

}  // namespace TinyTorch
