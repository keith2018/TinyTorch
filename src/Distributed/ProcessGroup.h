/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <map>
#include <unordered_set>

#include "Backend.h"

namespace tinytorch::distributed {

constexpr auto kProcessGroupDefaultTimeout = std::chrono::milliseconds(30 * 60 * 1000);

// Ref: https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/ProcessGroup.hpp
class ProcessGroup {
 public:
  ProcessGroup(const std::shared_ptr<Store>& store, int rank, int size)
      : store_(store), rank_(rank), size_(size), backendType_(UNDEFINED) {}
  ~ProcessGroup() = default;

  const std::shared_ptr<Store>& getStore() const { return store_; }

  int getRank() const { return rank_; }

  int getSize() const { return size_; }

  int64_t getID() const { return reinterpret_cast<std::intptr_t>(this); }
  int64_t getBackendID(BackendType backend_type) const {
    return reinterpret_cast<std::intptr_t>(getBackend(backend_type).get());
  }

  std::string getBackendName() const { return backendTypeToString(backendType_); }
  BackendType getBackendType() const { return backendType_; }

  void setTimeout(std::chrono::milliseconds timeout);

  std::shared_ptr<Work> broadcast(std::vector<Tensor>& tensors, const BroadcastOptions& opts);
  std::shared_ptr<Work> allReduce(std::vector<Tensor>& tensors, const AllReduceOptions& opts);
  std::shared_ptr<Work> reduce(std::vector<Tensor>& tensors, const ReduceOptions& opts);
  std::shared_ptr<Work> allGather(std::vector<std::vector<Tensor>>& outputTensors, std::vector<Tensor>& inputTensors,
                                  const AllGatherOptions& opts);
  std::shared_ptr<Work> reduceScatter(std::vector<Tensor>& outputTensors,
                                      std::vector<std::vector<Tensor>>& inputTensors, const ReduceScatterOptions& opts);
  std::shared_ptr<Work> send(std::vector<Tensor>& tensors, int dstRank);
  std::shared_ptr<Work> recv(std::vector<Tensor>& tensors, int srcRank);
  std::shared_ptr<Work> barrier(const BarrierOptions& opts);

  bool hasBackends() const;

  void setBackend(DeviceType deviceType, BackendType backendType,
                  const std::optional<std::shared_ptr<Backend>>& backend);

  std::shared_ptr<Backend> getDefaultBackend() const;

  void setDefaultBackend(const BackendType& backendType);

  std::shared_ptr<Backend> getBackend(DeviceType deviceType);

  std::shared_ptr<Backend> getBackend(BackendType backendType) const;
  std::vector<Device> getDeviceTypes() const;

  void abort();
  void shutdown();

  const std::string& getGroupName() const;
  void setGroupName(const std::string& name);

  const std::string& getGroupDesc() const;
  void setGroupDesc(const std::string& name);

  std::optional<Device> getBoundDeviceId() const { return boundDeviceId_; }
  void setBoundDeviceId(std::optional<Device> device) { boundDeviceId_ = device; }

  void releaseResources();

 protected:
  std::shared_ptr<Store> store_;
  const int rank_;
  const int size_;
  BackendType backendType_;
  std::string pgDesc_;

  std::unordered_set<DeviceType> deviceTypes_;
  std::map<DeviceType, BackendType> deviceTypeToBackendType_;
  std::unordered_map<DeviceType, std::shared_ptr<Backend>> deviceTypeToBackend_;
  std::unordered_map<BackendType, std::shared_ptr<Backend>> backendTypeToBackend_;

  std::optional<Device> boundDeviceId_;
};

}  // namespace tinytorch::distributed
