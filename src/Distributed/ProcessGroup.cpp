/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "ProcessGroup.h"

namespace tinytorch::distributed {

void ProcessGroup::setTimeout(std::chrono::milliseconds timeout) {
  for (auto& backend : backendTypeToBackend_) {
    backend.second->setTimeout(timeout);
  }
}

std::shared_ptr<Work> ProcessGroup::broadcast(std::vector<Tensor>& tensors, const BroadcastOptions& opts) {
  ASSERT(!tensors.empty());
  auto deviceType = tensors[0].device().type;
  return getBackend(deviceType)->broadcast(tensors, opts);
}

std::shared_ptr<Work> ProcessGroup::allReduce(std::vector<Tensor>& tensors, const AllReduceOptions& opts) {
  ASSERT(!tensors.empty());
  auto deviceType = tensors[0].device().type;
  return getBackend(deviceType)->allReduce(tensors, opts);
}

std::shared_ptr<Work> ProcessGroup::reduce(std::vector<Tensor>& tensors, const ReduceOptions& opts) {
  ASSERT(!tensors.empty());
  auto deviceType = tensors[0].device().type;
  return getBackend(deviceType)->reduce(tensors, opts);
}

std::shared_ptr<Work> ProcessGroup::allGather(std::vector<std::vector<Tensor>>& outputTensors,
                                              std::vector<Tensor>& inputTensors, const AllGatherOptions& opts) {
  ASSERT(!inputTensors.empty());
  auto deviceType = inputTensors[0].device().type;
  return getBackend(deviceType)->allGather(outputTensors, inputTensors, opts);
}

std::shared_ptr<Work> ProcessGroup::reduceScatter(std::vector<Tensor>& outputTensors,
                                                  std::vector<std::vector<Tensor>>& inputTensors,
                                                  const ReduceScatterOptions& opts) {
  ASSERT(!inputTensors.empty());
  ASSERT(!inputTensors.front().empty());
  auto deviceType = inputTensors[0][0].device().type;
  return getBackend(deviceType)->reduceScatter(outputTensors, inputTensors, opts);
}

std::shared_ptr<Work> ProcessGroup::send(std::vector<Tensor>& tensors, int dstRank) {
  ASSERT(!tensors.empty());
  auto deviceType = tensors[0].device().type;
  return getBackend(deviceType)->send(tensors, dstRank);
}

std::shared_ptr<Work> ProcessGroup::recv(std::vector<Tensor>& tensors, int srcRank) {
  ASSERT(!tensors.empty());
  auto deviceType = tensors[0].device().type;
  return getBackend(deviceType)->recv(tensors, srcRank);
}

std::shared_ptr<Work> ProcessGroup::barrier(const BarrierOptions& opts) {
  if (opts.device.has_value()) {
    return getBackend(opts.device.value().type)->barrier(opts);
  }
  return getDefaultBackend()->barrier(opts);
}

bool ProcessGroup::hasBackends() const { return !deviceTypeToBackendType_.empty(); }

void ProcessGroup::setBackend(DeviceType deviceType, BackendType backendType,
                              const std::optional<std::shared_ptr<Backend>>& backend) {
  deviceTypeToBackendType_[deviceType] = backendType;
  deviceTypes_.insert(deviceType);
  if (backendTypeToBackend_.find(backendType) != backendTypeToBackend_.end()) {
    auto existingBackend = backendTypeToBackend_.at(backendType);
    deviceTypeToBackend_[deviceType] = existingBackend;
    ASSERT(existingBackend->getBoundDeviceId() == (*backend)->getBoundDeviceId());
  } else {
    if (backend.has_value()) {
      deviceTypeToBackend_[deviceType] = backend.value();
      backendTypeToBackend_[backendType] = backend.value();
      (*backend)->setBoundDeviceId(boundDeviceId_);
    }
  }
}

std::shared_ptr<Backend> ProcessGroup::getDefaultBackend() const {
  auto backendIter = backendTypeToBackend_.find(backendType_);
  ASSERT(backendIter != backendTypeToBackend_.end());
  return backendIter->second;
}

void ProcessGroup::setDefaultBackend(const BackendType& backendType) { backendType_ = backendType; }

std::shared_ptr<Backend> ProcessGroup::getBackend(DeviceType deviceType) {
  if (deviceTypeToBackend_.find(deviceType) != deviceTypeToBackend_.end()) {
    return deviceTypeToBackend_.at(deviceType);
  }

  ASSERT(deviceTypeToBackendType_.find(deviceType) != deviceTypeToBackendType_.end());
  BackendType backendType = deviceTypeToBackendType_.at(deviceType);

  if (backendTypeToBackend_.find(backendType) != backendTypeToBackend_.end()) {
    auto backend = backendTypeToBackend_.at(backendType);
    deviceTypeToBackend_[deviceType] = backend;
    return backend;
  }

  LOGE("Could not retrieve or create the backend %s for device type %s", backendTypeToString(backendType),
       deviceTypeToString(deviceType));
  return nullptr;
}

std::shared_ptr<Backend> ProcessGroup::getBackend(BackendType backendType) const {
  ASSERT(backendTypeToBackend_.find(backendType) != backendTypeToBackend_.end());
  return backendTypeToBackend_.at(backendType);
}

std::vector<Device> ProcessGroup::getDeviceTypes() const {
  std::vector<Device> devices;
  devices.reserve(deviceTypes_.size());
  for (auto& dt : deviceTypes_) {
    devices.emplace_back(dt);
  }
  return devices;
}

void ProcessGroup::abort() {
  for (auto& backend : backendTypeToBackend_) {
    backend.second->abort();
  }
}

void ProcessGroup::shutdown() {
  for (auto& backend : backendTypeToBackend_) {
    backend.second->shutdown();
  }
}

const std::string& ProcessGroup::getGroupName() const {
  ASSERT(!deviceTypeToBackend_.empty());
  return deviceTypeToBackend_.begin()->second->getGroupName();
}

void ProcessGroup::setGroupName(const std::string& name) {
  for (auto& kv : deviceTypeToBackend_) {
    kv.second->setGroupName(name);
  }
}

const std::string& ProcessGroup::getGroupDesc() const { return pgDesc_; }

void ProcessGroup::setGroupDesc(const std::string& name) {
  pgDesc_ = name;
  for (auto& kv : deviceTypeToBackend_) {
    kv.second->setGroupDesc(name);
  }
}

void ProcessGroup::releaseResources() {
  store_.reset();
  deviceTypeToBackend_.clear();
  backendTypeToBackend_.clear();
}

}  // namespace tinytorch::distributed