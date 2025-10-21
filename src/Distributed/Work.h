/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <chrono>
#include <condition_variable>
#include <string>
#include <vector>

#include "Tensor.h"

namespace tinytorch::distributed {

constexpr auto kNoTimeout = std::chrono::milliseconds::zero();

enum class OpType : uint8_t {
  UNKNOWN = 0,
  BROADCAST,
  ALL_REDUCE,
  REDUCE,
  ALL_GATHER,
  REDUCE_SCATTER,
  SEND,
  RECV,
  BARRIER,
};

inline const char* opTypeToString(OpType type) {
  switch (type) {
    case OpType::BROADCAST:
      return "BROADCAST";
    case OpType::ALL_REDUCE:
      return "ALL_REDUCE";
    case OpType::REDUCE:
      return "REDUCE";
    case OpType::ALL_GATHER:
      return "ALL_GATHER";
    case OpType::REDUCE_SCATTER:
      return "REDUCE_SCATTER";
    case OpType::SEND:
      return "SEND";
    case OpType::RECV:
      return "RECV";
    case OpType::BARRIER:
      return "BARRIER";
    default:
      return "UNKNOWN";
  }
}

// Ref: https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/Work.hpp
class Work {
 public:
  explicit Work(int rank = -1, OpType opType = OpType::UNKNOWN) : rank_(rank), opType_(opType) {}

  virtual ~Work() = default;

  virtual bool isCompleted() {
    std::lock_guard<std::mutex> lock(mutex_);
    return completed_;
  }

  virtual bool isSuccess() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return exception_.empty();
  }

  virtual std::string exception() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return exception_;
  }

  virtual std::vector<Tensor> result() {
    NOT_IMPLEMENTED();
    return {};
  }

  virtual bool wait(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (timeout == kNoTimeout) {
      cv_.wait(lock, [&] { return completed_; });
    } else {
      cv_.wait_for(lock, timeout, [&] { return completed_; });
      if (!completed_) {
        LOGE("Work: Operation timed out!");
        ASSERT(false);
      }
    }
    return exception_.empty();
  }

  bool wait() { return wait(kNoTimeout); }

  int getRank() const { return rank_; }

  OpType getOpType() const { return opType_; }

 protected:
  void finish(std::string exception) {
    std::unique_lock<std::mutex> lock(mutex_);
    completed_ = true;
    exception_ = std::move(exception);
    lock.unlock();
    cv_.notify_all();
  }

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  bool completed_ = false;
  std::string exception_;

  const int rank_;
  OpType opType_;
};

}  // namespace tinytorch::distributed
