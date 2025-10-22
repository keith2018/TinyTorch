/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "WorkNCCL.h"

#include <thread>

namespace tinytorch::distributed {

bool WorkNCCL::isCompleted() {
  if (Work::isCompleted()) {
    return true;
  }

  if (!cudaEvent_.query()) {
    return false;
  }
  return true;
}

bool WorkNCCL::wait(std::chrono::milliseconds timeout) {
  synchronize();

  if (timeout != kNoTimeout) {
    while (!isCompleted()) {
      bool timedOut = checkTimeout(timeout == kNoTimeout ? std::nullopt : std::make_optional(timeout));
      if (timedOut) {
        finish("WorkNCCL wait timed out");
        return false;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  } else if (isBarrierOp_ && !isCompleted()) {
    auto &stream = cuda::getCurrentCUDAStream(device_.index);
    stream.synchronize();
  }

  finish("");
  return true;
}

void WorkNCCL::synchronize() const {
  auto &computeStream = cuda::getCurrentCUDAStream(device_.index);
  cudaEvent_.block(computeStream);
}

bool WorkNCCL::checkTimeout(std::optional<std::chrono::milliseconds> timeout) const {
  auto currentTimepoint = std::chrono::steady_clock::now();
  auto timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(currentTimepoint - workStartTime_);
  auto workTimeout = timeout ? *timeout : opTimeout_;
  return timeElapsed >= workTimeout;
}

}  // namespace tinytorch::distributed