/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

namespace tinytorch {

class Profiler {
 public:
  static void start();
  static void stop();
};

class ScopedProfiler {
 public:
  explicit ScopedProfiler(const char* name = nullptr);
  ~ScopedProfiler();

  ScopedProfiler(const ScopedProfiler&) = delete;
  ScopedProfiler& operator=(const ScopedProfiler&) = delete;

  explicit operator bool() const { return true; }

 private:
  const char* name_;
};

class ProfilerRange {
 public:
  explicit ProfilerRange(const char* name);
  ~ProfilerRange();

  ProfilerRange(const ProfilerRange&) = delete;
  ProfilerRange& operator=(const ProfilerRange&) = delete;

 private:
  const char* name_;
};

#ifndef NO_PROFILER
#define PROFILE_SCOPE(name) if (auto _scopedProfiler = tinytorch::ScopedProfiler(name))
#define PROFILE_START() tinytorch::Profiler::start()
#define PROFILE_STOP() tinytorch::Profiler::stop()
#define PROFILE_RANGE(name) tinytorch::ProfilerRange _profilerRange##__LINE__(name)
#else
#define PROFILE_SCOPE(name) if (true)
#define PROFILE_START()
#define PROFILE_STOP()
#define PROFILE_RANGE(name)
#endif

}  // namespace tinytorch
