/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

namespace tinytorch {

class NoGradGuard {
 public:
  NoGradGuard() {
    prevGradEnabled = gradEnabled;
    gradEnabled = false;
  }

  ~NoGradGuard() { gradEnabled = prevGradEnabled; }

  static bool isGradEnabled() { return gradEnabled; }

  explicit operator bool() const { return true; }

 private:
  static thread_local bool gradEnabled;
  bool prevGradEnabled;
};

#define WithNoGrad if (auto _noGrad = tinytorch::NoGradGuard())

}  // namespace tinytorch
