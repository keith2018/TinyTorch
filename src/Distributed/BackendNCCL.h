/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <list>
#include <set>

#include "Backend.h"
#include "Store.h"
#include "WorkNCCL.h"

namespace tinytorch::distributed {

#define NCCL_CALL(expr)                                \
  do {                                                 \
    ncclResult_t ret = (expr);                         \
    if (ret != ncclSuccess && ret != ncclInProgress) { \
      NCCL_ERROR(ret);                                 \
      setError(BackendErrorType::COMM_ERROR);          \
      return nullptr;                                  \
    }                                                  \
  } while (0)

constexpr const char* NCCL_BACKEND_NAME = "nccl";
constexpr const char* NCCL_COMM_ID_PREFIX = "nccl_unique_id_";
constexpr auto kBackendNCCLDefaultTimeout = std::chrono::milliseconds(10 * 60 * 1000);

class BackendNCCL : public Backend {
 public:
  BackendNCCL(const std::shared_ptr<Store>& store, int rank, int size,
              std::shared_ptr<BackendOptions> options = std::make_shared<BackendOptions>(NCCL_BACKEND_NAME,
                                                                                         kBackendNCCLDefaultTimeout))
      : Backend(rank, size), store_(store), options_(std::move(options)) {}

  void setTimeout(std::chrono::milliseconds timeout) override { options_->timeout = timeout; }

  std::string getBackendName() const override { return NCCL_BACKEND_NAME; }

  std::shared_ptr<BackendOptions> getBackendOptions() override { return options_; }

  std::shared_ptr<Work> broadcast(std::vector<Tensor>& tensors, const BroadcastOptions& opts) override;
  std::shared_ptr<Work> allReduce(std::vector<Tensor>& tensors, const AllReduceOptions& opts) override;
  std::shared_ptr<Work> reduce(std::vector<Tensor>& tensors, const ReduceOptions& opts) override;
  std::shared_ptr<Work> allGather(std::vector<std::vector<Tensor>>& outputTensors, std::vector<Tensor>& inputTensors,
                                  const AllGatherOptions& opts) override;
  std::shared_ptr<Work> reduceScatter(std::vector<Tensor>& outputTensors,
                                      std::vector<std::vector<Tensor>>& inputTensors,
                                      const ReduceScatterOptions& opts) override;
  std::shared_ptr<Work> send(std::vector<Tensor>& tensors, int dstRank) override;
  std::shared_ptr<Work> recv(std::vector<Tensor>& tensors, int srcRank) override;
  std::shared_ptr<Work> barrier(const BarrierOptions& opts) override;

  BackendErrorType getError() override {
    std::lock_guard<std::mutex> lock(mutex_);
    return error_;
  }

  void abort() override {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& kv : devNCCLCommMap_) {
      auto& comm = kv.second;
      if (comm && comm->isInitialized()) {
        comm->abort();
      }
    }
    activeWorks_.clear();
    ncclStreams_.clear();
    devNCCLCommMap_.clear();
    error_ = BackendErrorType::COMM_ERROR;
  }

  void shutdown() override {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& kv : devNCCLCommMap_) {
      auto& comm = kv.second;
      if (comm && comm->isInitialized()) {
        comm->shutdown();
      }
    }
    activeWorks_.clear();
    ncclStreams_.clear();
    devNCCLCommMap_.clear();
    error_ = BackendErrorType::SUCCESS;
  }

 protected:
  void setError(BackendErrorType error) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (error_ == BackendErrorType::SUCCESS) {
      error_ = error;
    }
  }

  void clearError() {
    std::lock_guard<std::mutex> lock(mutex_);
    error_ = BackendErrorType::SUCCESS;
  }

  bool hasError() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return error_ != BackendErrorType::SUCCESS;
  }

  template <typename Func>
  std::shared_ptr<Work> collective(Tensor& input, Tensor& output, Func&& ncclFunc, OpType opType);

  template <typename Func>
  std::shared_ptr<Work> pointToPoint(Tensor& tensor, Func&& ncclFunc, OpType opType) {
    return collective(tensor, tensor, std::forward<Func>(ncclFunc), opType);
  }

  void workEnqueue(const std::shared_ptr<WorkNCCL>& work) {
    std::lock_guard<std::mutex> lock(mutex_);
    activeWorks_.push_back(work);
  }

  void cleanupCompletedWorks() {
    std::lock_guard<std::mutex> lock(mutex_);
    activeWorks_.remove_if([](const std::weak_ptr<WorkNCCL>& work) {
      auto workPtr = work.lock();
      return !workPtr || workPtr->isCompleted();
    });
  }

  std::shared_ptr<NCCLComm> getNCCLComm(const std::string& deviceKey, const Device& device);
  cuda::CUDAStream& getNCCLStream(const std::string& deviceKey, DeviceIndex deviceIndex);

  bool waitNcclCommandResult(const std::shared_ptr<NCCLComm>& comm, ncclResult_t state) const;

  static ncclDataType_t getNcclDataType(DType dtype);
  static ncclRedOp_t getNcclReduceOp(ReduceOpType reduceOp);

 private:
  std::shared_ptr<Store> store_;
  const std::shared_ptr<BackendOptions> options_;
  uint64_t ncclCommCounter_ = 0;

  std::unordered_map<std::string, std::shared_ptr<NCCLComm>> devNCCLCommMap_;
  std::unordered_map<std::string, cuda::CUDAStream> ncclStreams_;

  std::list<std::weak_ptr<WorkNCCL>> activeWorks_;

  mutable std::mutex mutex_;
  BackendErrorType error_ = BackendErrorType::SUCCESS;
};

template <typename Func>
std::shared_ptr<Work> BackendNCCL::collective(Tensor& input, Tensor& output, Func&& ncclFunc, OpType opType) {
  if (!input.defined()) {
    LOGE("BackendNCCL::collective error: input undefined");
    return nullptr;
  }

  Device device = input.device();
  std::string deviceKey = std::to_string(device.index);

  auto comm = getNCCLComm(deviceKey, device);
  if (!comm) {
    LOGE("BackendNCCL::collective getNCCLComm error");
    return nullptr;
  }

  auto& ncclStream = getNCCLStream(deviceKey, device.index);
  auto& computeSteam = cuda::getCurrentCUDAStream(device.index);
  ncclStream.waitStream(computeSteam);

  auto work = std::make_shared<WorkNCCL>(std::to_string(getID()), getGroupDesc(), device, getRank(), opType);
  work->setOpTimeout(options_->timeout);
  work->setNCCLComm(comm);
  work->setStore(store_);
  work->setIsBarrierOp(opType == OpType::BARRIER);
  work->setCudaEvent(cuda::createCUDAEvent(device.index));
  work->setOutputs({output});

  cuda::CudaDeviceGuard guard(device.index);
  NCCL_CALL(ncclGroupStart());
  NCCL_CALL(ncclFunc(comm, ncclStream.stream()));
  ncclResult_t state = ncclGroupEnd();

  if (!waitNcclCommandResult(comm, state)) {
    return nullptr;
  }

  work->getCudaEvent().record(ncclStream);
  workEnqueue(work);
  return work;
}

}  // namespace tinytorch::distributed
