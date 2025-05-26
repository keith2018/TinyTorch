/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <chrono>
#include <string>

#include "Logger.h"

namespace tinytorch {

class Timer {
 public:
  void start();
  void mark();
  int64_t elapseMillis() const;

 private:
  std::chrono::time_point<std::chrono::steady_clock> start_;
  std::chrono::time_point<std::chrono::steady_clock> end_;
};

class ScopedTimer {
 public:
  explicit ScopedTimer(const char *str) : tag_(str) { timer_.start(); }

  ~ScopedTimer() {
    timer_.mark();
    LOGD("TIMER %s: cost: %lld ms", tag_.c_str(), timer_.elapseMillis());
  }

  explicit operator bool() const { return true; }

 private:
  Timer timer_;
  std::string tag_;
};

#ifndef NO_TIMER
#define FUNCTION_TIMED() \
  auto _functionTimer = tinytorch::ScopedTimer(__FUNCTION__)
#define SCOPE_TIMED(X) if (auto _scopeTimer = tinytorch::ScopedTimer(X))
#else
#define FUNCTION_TIMED()
#define SCOPE_TIMED(X)
#endif

}  // namespace tinytorch
