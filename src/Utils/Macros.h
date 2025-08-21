/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <tuple>
#include <type_traits>

#include "Logger.h"

namespace tinytorch {

// unused
#define UNUSED(x) (void)(x)

// assert
#ifdef _MSC_VER
#include <intrin.h>
#define DEBUG_BREAK() __debugbreak()
#elif defined(__GNUC__) || defined(__clang__)
#if defined(__i386__) || defined(__x86_64__)
#define DEBUG_BREAK() __asm__ volatile("int3")
#else
#define DEBUG_BREAK() __builtin_trap()
#endif
#else
#include <cstdlib>
#define DEBUG_BREAK() std::abort()
#endif

#define ASSERT(expr)                         \
  do {                                       \
    if (!(expr)) {                           \
      LOGE("Assertion failed: (" #expr ")"); \
      DEBUG_BREAK();                         \
    }                                        \
  } while (0)

// not implemented
#define NOT_IMPLEMENTED() ASSERT(!"Not implemented")

// align
#ifdef _MSC_VER
#define ALIGN(N) __declspec(align(N))
#else
#define ALIGN(N) __attribute__((aligned(N)))
#endif

// static call
struct StaticCaller {
  explicit StaticCaller(void (*func)()) { func(); }
};

#define STATIC_CALL(func) static StaticCaller _static_caller_##func(func);

#if defined(__CUDACC__)
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

}  // namespace tinytorch
