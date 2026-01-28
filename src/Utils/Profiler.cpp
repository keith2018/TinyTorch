/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Profiler.h"

#include "CUDAUtils.h"
#include "Logger.h"

#ifdef USE_CUDA
#include <cuda_profiler_api.h>
#endif

namespace tinytorch {

void Profiler::start() {
#ifdef USE_CUDA
  CUDA_CHECK(cudaProfilerStart());
  LOGI("CUDA Profiler started");
#endif
}

void Profiler::stop() {
#ifdef USE_CUDA
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaProfilerStop());
  LOGI("CUDA Profiler stopped");
#endif
}

ScopedProfiler::ScopedProfiler(const char* name) : name_(name ? name : "ScopedProfiler") { Profiler::start(); }

ScopedProfiler::~ScopedProfiler() { Profiler::stop(); }

ProfilerRange::ProfilerRange(const char* name) : name_(name) {
#ifdef USE_CUDA
  CUDA_CHECK(cudaDeviceSynchronize());
#endif
}

ProfilerRange::~ProfilerRange() {
#ifdef USE_CUDA
  CUDA_CHECK(cudaDeviceSynchronize());
#endif
}

}  // namespace tinytorch
