/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <nccl.h>

#include <memory>
#include <thread>

#include "Utils/CUDAUtils.h"

namespace tinytorch::distributed {

#define NCCL_CHECK(call)                                                                                             \
  do {                                                                                                               \
    ncclResult_t err = call;                                                                                         \
    if (err != ncclSuccess && err != ncclInProgress) {                                                               \
      std::cerr << "NCCL error in file '" << __FILE__ << "' in line " << __LINE__ << ": " << ncclGetErrorString(err) \
                << " (" << err << ")" << std::endl;                                                                  \
      std::abort();                                                                                                  \
    }                                                                                                                \
  } while (0)

#define NCCL_ERROR(err)                                                                                              \
  do {                                                                                                               \
    if ((err) != ncclSuccess && err != ncclInProgress) {                                                             \
      std::cerr << "NCCL error in file '" << __FILE__ << "' in line " << __LINE__ << ": " << ncclGetErrorString(err) \
                << " (" << (err) << ")" << std::endl;                                                                \
    }                                                                                                                \
  } while (0)

// non blocking
static_assert(NCCL_VERSION_CODE >= NCCL_VERSION(2, 14, 0), "NCCL version must be 2.14 or later");

// Ref: https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/NCCLUtils.hpp
class NCCLComm {
 public:
  explicit NCCLComm(ncclComm_t ncclComm) : ncclComm_(ncclComm) {}

  NCCLComm() = default;

  ~NCCLComm() noexcept { shutdown(); }

  static std::shared_ptr<NCCLComm> create(int numRanks, int rank, ncclUniqueId commId, DeviceIndex deviceIndex) {
    cuda::CudaDeviceGuard gpuGuard(deviceIndex);
    auto comm = std::make_shared<NCCLComm>();

    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = 0;  // Enable non-blocking mode

    NCCL_CHECK(ncclCommInitRankConfig(&(comm->ncclComm_), numRanks, commId, rank, &config));
    comm->rank_ = rank;
    comm->deviceIndex_ = deviceIndex;
    comm->initialized_ = false;
    return comm;
  }

  NCCLComm(const NCCLComm&) = delete;
  NCCLComm& operator=(const NCCLComm&) = delete;

  DeviceIndex getDeviceIndex() const { return deviceIndex_; }

  ncclComm_t getNcclComm() {
    waitReady();
    return ncclComm_;
  }

  bool isInitialized() const { return initialized_; }

  ncclResult_t checkForNcclError() const {
    ASSERT(ncclComm_);
    ncclResult_t ncclAsyncErr;
    NCCL_CHECK(ncclCommGetAsyncError(ncclComm_, &ncclAsyncErr));
    return ncclAsyncErr;
  }

  void waitReady() {
    ASSERT(ncclComm_);
    if (initialized_) {
      return;
    }

    ncclResult_t asyncErr;
    do {
      NCCL_CHECK(ncclCommGetAsyncError(ncclComm_, &asyncErr));
      if (asyncErr == ncclInProgress) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        continue;
      } else if (asyncErr != ncclSuccess) {
        NCCL_ERROR(asyncErr);
        LOGE("NCCL communicator initialization failed");
        return;
      }
      break;
    } while (true);

    initialized_ = true;
  }

  void abort() noexcept {
    if (ncclComm_) {
      cuda::CudaDeviceGuard gpuGuard(deviceIndex_);
      NCCL_CHECK(ncclCommAbort(ncclComm_));
      ncclComm_ = nullptr;
      initialized_ = false;
    }
  }

  void finalize() const noexcept {
    if (ncclComm_) {
      cuda::CudaDeviceGuard gpuGuard(deviceIndex_);
      NCCL_CHECK(ncclCommFinalize(ncclComm_));
    }
  }

  void shutdown() noexcept {
    if (ncclComm_) {
      cuda::CudaDeviceGuard gpuGuard(deviceIndex_);
      NCCL_CHECK(ncclCommDestroy(ncclComm_));
      ncclComm_ = nullptr;
      initialized_ = false;
    }
  }

 protected:
  int rank_ = -1;
  bool initialized_ = false;
  DeviceIndex deviceIndex_ = -1;

 private:
  ncclComm_t ncclComm_ = nullptr;
};

}  // namespace tinytorch::distributed
