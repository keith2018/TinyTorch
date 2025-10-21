/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <chrono>
#include <string>
#include <vector>

#include "Store.h"
#include "Tensor.h"
#include "Types.h"
#include "Work.h"

namespace tinytorch::distributed {

constexpr auto kBackendDefaultTimeout = std::chrono::milliseconds(30 * 60 * 1000);

enum BackendType : uint8_t {
  UNDEFINED = 0,
  NCCL = 1,
};

inline const char* backendTypeToString(const BackendType& type) {
  switch (type) {
    case NCCL:
      return "nccl";
    default:
      return "undefined";
  }
}

enum class BackendErrorType : uint8_t {
  SUCCESS = 0,
  TIMEOUT = 1,
  COMM_ERROR = 2,
  REMOTE_ERROR = 3,
};

// Ref: https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/Backend.hpp
class Backend {
 public:
  struct BackendOptions {
    explicit BackendOptions(std::string backend, std::chrono::milliseconds timeout = kBackendDefaultTimeout)
        : timeout(timeout), backend(std::move(backend)) {}
    ~BackendOptions() = default;

    std::chrono::milliseconds timeout;
    const std::string backend;
    std::string groupName;
    std::vector<uint64_t> globalRanksInGroup;
  };

  Backend(int rank, int size) : rank_(rank), size_(size) {}
  virtual ~Backend() = default;

  int getRank() const { return rank_; }

  int getSize() const { return size_; }

  int64_t getID() const { return reinterpret_cast<std::intptr_t>(this); }

  virtual void setTimeout(std::chrono::milliseconds timeout) = 0;

  virtual std::string getBackendName() const = 0;

  virtual std::shared_ptr<BackendOptions> getBackendOptions() = 0;

  virtual std::shared_ptr<Work> broadcast(std::vector<Tensor>& tensors, const BroadcastOptions& opts) {
    NOT_IMPLEMENTED();
    return nullptr;
  }

  virtual std::shared_ptr<Work> allReduce(std::vector<Tensor>& tensors, const AllReduceOptions& opts) {
    NOT_IMPLEMENTED();
    return nullptr;
  }

  virtual std::shared_ptr<Work> reduce(std::vector<Tensor>& tensors, const ReduceOptions& opts) {
    NOT_IMPLEMENTED();
    return nullptr;
  }

  virtual std::shared_ptr<Work> allGather(std::vector<std::vector<Tensor>>& outputTensors,
                                          std::vector<Tensor>& inputTensors, const AllGatherOptions& opts) {
    NOT_IMPLEMENTED();
    return nullptr;
  }

  virtual std::shared_ptr<Work> reduceScatter(std::vector<Tensor>& outputTensors,
                                              std::vector<std::vector<Tensor>>& inputTensors,
                                              const ReduceScatterOptions& opts) {
    NOT_IMPLEMENTED();
    return nullptr;
  }

  virtual std::shared_ptr<Work> send(std::vector<Tensor>& tensors, int dstRank) {
    NOT_IMPLEMENTED();
    return nullptr;
  }

  virtual std::shared_ptr<Work> recv(std::vector<Tensor>& tensors, int srcRank) {
    NOT_IMPLEMENTED();
    return nullptr;
  }

  virtual std::shared_ptr<Work> barrier(const BarrierOptions& opts) {
    NOT_IMPLEMENTED();
    return nullptr;
  }

  void setGroupName(const std::string& name) { pgName_ = name; }
  const std::string& getGroupName() const { return pgName_; }

  void setGroupDesc(const std::string& desc) { pgDesc_ = desc; }
  const std::string& getGroupDesc() const { return pgDesc_; }

  std::optional<Device> getBoundDeviceId() const { return boundDeviceId_; }
  void setBoundDeviceId(std::optional<Device> device) { boundDeviceId_ = device; }

  virtual BackendErrorType getError() {
    NOT_IMPLEMENTED();
    return BackendErrorType::SUCCESS;
  }

  virtual void abort() {}
  virtual void shutdown() {}

 protected:
  const int rank_;
  const int size_;

  std::string pgName_;
  std::string pgDesc_;

  std::optional<Device> boundDeviceId_;
};

}  // namespace tinytorch::distributed
