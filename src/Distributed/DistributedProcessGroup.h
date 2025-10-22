/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "ProcessGroup.h"
#include "TCPStore.h"

namespace tinytorch::distributed {

enum class InitMethod : uint8_t {
  UNKNOWN = 0,
  ENV = 1,
  TCP = 2,
  FILE = 3,
};

inline const char* initMethodToString(const InitMethod& method) {
  switch (method) {
    case InitMethod::ENV:
      return "env";
    case InitMethod::TCP:
      return "tcp";
    case InitMethod::FILE:
      return "file";
    default:
      return "unknown";
  }
}

class DistributedProcessGroup {
 public:
  DistributedProcessGroup() = default;
  ~DistributedProcessGroup() { destroyProcessGroup(); }

  static std::shared_ptr<DistributedProcessGroup>& getInstance() {
    static std::shared_ptr<DistributedProcessGroup> instance = std::make_shared<DistributedProcessGroup>();
    return instance;
  }

  bool initProcessGroup(BackendType backend, const std::string& initMethod = "env://", int rank = -1,
                        int worldSize = -1, std::chrono::milliseconds timeout = kProcessGroupDefaultTimeout,
                        bool waitWorkers = true);

  std::shared_ptr<ProcessGroup> getProcessGroup() {
    if (!processGroup_) {
      ASSERT(false && "ProcessGroup not initialized");
      return nullptr;
    }
    return processGroup_;
  }

  void destroyProcessGroup() {
    if (processGroup_) {
      processGroup_->shutdown();
      processGroup_.reset();
    }
  }

  bool isInitialized() const { return processGroup_ != nullptr; }

  int getRank() { return getProcessGroup()->getRank(); }

  int getWorldSize() { return getProcessGroup()->getSize(); }

  std::shared_ptr<Work> broadcast(std::vector<Tensor>& tensors, int root = 0) {
    BroadcastOptions opts;
    opts.rootRank = root;
    return getProcessGroup()->broadcast(tensors, opts);
  }

  std::shared_ptr<Work> allReduce(std::vector<Tensor>& tensors, ReduceOpType op = SUM) {
    AllReduceOptions opts;
    opts.reduceOp = op;
    return getProcessGroup()->allReduce(tensors, opts);
  }

  std::shared_ptr<Work> reduce(std::vector<Tensor>& tensors, int dst = 0, ReduceOpType op = SUM) {
    ReduceOptions opts;
    opts.rootRank = dst;
    opts.reduceOp = op;
    return getProcessGroup()->reduce(tensors, opts);
  }

  std::shared_ptr<Work> allGather(std::vector<std::vector<Tensor>>& output_tensors,
                                  std::vector<Tensor>& input_tensors) {
    AllGatherOptions opts;
    return getProcessGroup()->allGather(output_tensors, input_tensors, opts);
  }

  std::shared_ptr<Work> reduceScatter(std::vector<Tensor>& output_tensors,
                                      std::vector<std::vector<Tensor>>& input_tensors, ReduceOpType op = SUM) {
    ReduceScatterOptions opts;
    opts.reduceOp = op;
    return getProcessGroup()->reduceScatter(output_tensors, input_tensors, opts);
  }

  std::shared_ptr<Work> send(std::vector<Tensor>& tensors, int dst) { return getProcessGroup()->send(tensors, dst); }

  std::shared_ptr<Work> recv(std::vector<Tensor>& tensors, int src) { return getProcessGroup()->recv(tensors, src); }

  std::shared_ptr<Work> barrier(bool async = false, const std::vector<int64_t>& deviceIds = {}) {
    BarrierOptions opts;
    opts.deviceIds = deviceIds;
    auto work = getProcessGroup()->barrier(opts);
    if (async) {
      return work;
    }

    if (work) {
      work->wait();
    }
    return nullptr;
  }

 private:
  struct InitConfig {
    InitMethod method;
    std::string masterAddr;
    int masterPort;
    std::string filePath;
    int rank;
    int worldSize;
    std::chrono::milliseconds timeout;
    bool waitWorkers;

    InitConfig()
        : method(InitMethod::UNKNOWN),
          masterPort(TCPStore::kDefaultPort),
          rank(-1),
          worldSize(-1),
          timeout(Store::kDefaultTimeout),
          waitWorkers(true) {}
  };

  static InitConfig parseInitString(const std::string& initString, int rank, int worldSize);
  static std::shared_ptr<Store> createStore(const InitConfig& config);

  std::shared_ptr<ProcessGroup> processGroup_;
};

}  // namespace tinytorch::distributed
