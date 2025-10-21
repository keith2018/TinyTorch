/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "NCCLUtils.h"
#include "Store.h"
#include "Work.h"

namespace tinytorch::distributed {

class WorkNCCL : public Work {
 public:
  WorkNCCL(std::string pgUID, std::string pgDesc, Device& device, int rank, OpType opType)
      : Work(rank, opType),
        pgUID_(std::move(pgUID)),
        pgDesc_(std::move(pgDesc)),
        device_(device),
        workStartTime_(std::chrono::steady_clock::now()) {}

  bool isCompleted() override;

  bool wait(std::chrono::milliseconds timeout) override;

  std::vector<Tensor> result() override { return outputs_; }

  void setOpTimeout(std::chrono::milliseconds timeout) { opTimeout_ = timeout; }
  void setNCCLComm(const std::shared_ptr<NCCLComm>& comm) { ncclComm_ = comm; }
  void setStore(const std::shared_ptr<Store>& store) { store_ = store; }
  void setIsBarrierOp(bool isBarrierOp) { isBarrierOp_ = isBarrierOp; }
  void setCudaEvent(cuda::CUDAEvent&& event) { cudaEvent_ = std::move(event); }
  void setOutputs(std::vector<Tensor>&& output) { outputs_ = std::move(output); }

  const Device& getDevice() const { return device_; }
  const std::string& getPgUID() const { return pgUID_; }
  const std::string& getPgDesc() const { return pgDesc_; }
  const cuda::CUDAEvent& getCudaEvent() const { return cudaEvent_; }

  void synchronize() const;
  bool checkTimeout(std::optional<std::chrono::milliseconds> timeout) const;

 protected:
  std::string pgUID_;
  std::string pgDesc_;
  Device device_;

  std::chrono::milliseconds opTimeout_{};
  std::chrono::time_point<std::chrono::steady_clock> workStartTime_;

  std::shared_ptr<NCCLComm> ncclComm_;
  std::shared_ptr<Store> store_;

  bool isBarrierOp_{false};
  cuda::CUDAEvent cudaEvent_;
  std::vector<Tensor> outputs_;
};

}  // namespace tinytorch::distributed
