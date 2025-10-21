/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "BackendNCCL.h"

#include "Operation/OpTransform.h"

namespace tinytorch::distributed {

std::shared_ptr<Work> BackendNCCL::broadcast(std::vector<Tensor>& tensors, const BroadcastOptions& opts) {
  ASSERT(tensors.size() == 1);
  auto& tensor = tensors.back();
  return collective(
      tensor, tensor,
      [&](const std::shared_ptr<NCCLComm>& comm, cudaStream_t stream) {
        return ncclBroadcast(tensor.dataPtr<>(), tensor.dataPtr<>(), tensor.numel(), getNcclDataType(tensor.dtype()),
                             opts.rootRank, comm->getNcclComm(), stream);
      },
      OpType::BROADCAST);
}

std::shared_ptr<Work> BackendNCCL::allReduce(std::vector<Tensor>& tensors, const AllReduceOptions& opts) {
  ASSERT(tensors.size() == 1);
  auto& tensor = tensors.back();
  return collective(
      tensor, tensor,
      [&](const std::shared_ptr<NCCLComm>& comm, cudaStream_t stream) {
        return ncclAllReduce(tensor.dataPtr<>(), tensor.dataPtr<>(), tensor.numel(), getNcclDataType(tensor.dtype()),
                             getNcclReduceOp(opts.reduceOp), comm->getNcclComm(), stream);
      },
      OpType::ALL_REDUCE);
}

std::shared_ptr<Work> BackendNCCL::reduce(std::vector<Tensor>& tensors, const ReduceOptions& opts) {
  ASSERT(tensors.size() == 1);
  auto& tensor = tensors.back();
  return collective(
      tensor, tensor,
      [&](const std::shared_ptr<NCCLComm>& comm, cudaStream_t stream) {
        return ncclReduce(tensor.dataPtr<>(), tensor.dataPtr<>(), tensor.numel(), getNcclDataType(tensor.dtype()),
                          getNcclReduceOp(opts.reduceOp), opts.rootRank, comm->getNcclComm(), stream);
      },
      OpType::REDUCE);
}

std::shared_ptr<Work> BackendNCCL::allGather(std::vector<std::vector<Tensor>>& outputTensors,
                                             std::vector<Tensor>& inputTensors, const AllGatherOptions& opts) {
  ASSERT(outputTensors.size() == 1);
  ASSERT(inputTensors.size() == 1);
  auto& input = inputTensors.back();
  auto output = op::stack(ArrayView<Tensor>(outputTensors.back()), 0);
  return collective(
      input, output,
      [&](const std::shared_ptr<NCCLComm>& comm, cudaStream_t stream) {
        return ncclAllGather(input.dataPtr<>(), output.dataPtr<>(), input.numel(), getNcclDataType(input.dtype()),
                             comm->getNcclComm(), stream);
      },
      OpType::ALL_GATHER);
}

std::shared_ptr<Work> BackendNCCL::reduceScatter(std::vector<Tensor>& outputTensors,
                                                 std::vector<std::vector<Tensor>>& inputTensors,
                                                 const ReduceScatterOptions& opts) {
  ASSERT(outputTensors.size() == 1);
  ASSERT(inputTensors.size() == 1);
  auto input = op::stack(ArrayView<Tensor>(inputTensors.back()), 0);
  auto& output = outputTensors.back();
  return collective(
      input, output,
      [&](const std::shared_ptr<NCCLComm>& comm, cudaStream_t stream) {
        return ncclReduceScatter(input.dataPtr<>(), output.dataPtr<>(), output.numel(), getNcclDataType(output.dtype()),
                                 getNcclReduceOp(opts.reduceOp), comm->getNcclComm(), stream);
      },
      OpType::REDUCE_SCATTER);
}

std::shared_ptr<Work> BackendNCCL::send(std::vector<Tensor>& tensors, int dstRank) {
  ASSERT(tensors.size() == 1);
  auto& tensor = tensors.back();
  return pointToPoint(
      tensor,
      [&](const std::shared_ptr<NCCLComm>& comm, cudaStream_t stream) {
        return ncclSend(tensor.dataPtr<>(), tensor.numel(), getNcclDataType(tensor.dtype()), dstRank,
                        comm->getNcclComm(), stream);
      },
      OpType::SEND);
}

std::shared_ptr<Work> BackendNCCL::recv(std::vector<Tensor>& tensors, int srcRank) {
  ASSERT(tensors.size() == 1);
  auto& tensor = tensors.back();
  return pointToPoint(
      tensor,
      [&](const std::shared_ptr<NCCLComm>& comm, cudaStream_t stream) {
        return ncclRecv(tensor.dataPtr<>(), tensor.numel(), getNcclDataType(tensor.dtype()), srcRank,
                        comm->getNcclComm(), stream);
      },
      OpType::RECV);
}

std::shared_ptr<Work> BackendNCCL::barrier(const BarrierOptions& opts) {
  DeviceIndex barDevIdx = -1;
  if (!opts.deviceIds.empty()) {
    barDevIdx = static_cast<DeviceIndex>(opts.deviceIds.front());
  } else if (opts.device.has_value()) {
    ASSERT(opts.device.value().type == DeviceType::CUDA);
    barDevIdx = opts.device.value().index;
  } else if (boundDeviceId_.has_value()) {
    barDevIdx = boundDeviceId_.value().index;
  }
  ASSERT(barDevIdx >= 0);
  auto tensor = Tensor::zeros({1}, Options(Device(DeviceType::CUDA, barDevIdx), DType::Float32));
  std::vector<Tensor> tensors = {tensor};
  return allReduce(tensors, {});
}

std::shared_ptr<NCCLComm> BackendNCCL::getNCCLComm(const std::string& deviceKey, const Device& device) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = devNCCLCommMap_.find(deviceKey);
  if (it != devNCCLCommMap_.end()) {
    return it->second;
  }

  ncclUniqueId commId;
  std::string key = NCCL_COMM_ID_PREFIX + std::to_string(ncclCommCounter_++);
  if (getRank() == 0) {
    NCCL_CALL(ncclGetUniqueId(&commId));
    store_->set(key, std::string(reinterpret_cast<char*>(&commId), sizeof(commId)));
  } else {
    auto idData = store_->get(key);
    if (idData.size() != sizeof(commId)) {
      setError(BackendErrorType::COMM_ERROR);
      return nullptr;
    }
    std::memcpy(&commId, idData.data(), sizeof(commId));
  }

  auto comm = NCCLComm::create(getSize(), getRank(), commId, device.index);
  if (!comm) {
    setError(BackendErrorType::COMM_ERROR);
    return nullptr;
  }

  devNCCLCommMap_[deviceKey] = comm;
  return comm;
}

cuda::CUDAStream& BackendNCCL::getNCCLStream(const std::string& deviceKey, DeviceIndex deviceIndex) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = ncclStreams_.find(deviceKey);
  if (it != ncclStreams_.end()) {
    return it->second;
  }

  auto stream = cuda::createCUDAStream(deviceIndex);
  ncclStreams_[deviceKey] = std::move(stream);
  return ncclStreams_[deviceKey];
}

bool BackendNCCL::waitNcclCommandResult(const std::shared_ptr<NCCLComm>& comm, ncclResult_t state) const {
  auto startTimepoint = std::chrono::steady_clock::now();
  if (state == ncclInProgress) {
    do {
      auto now = std::chrono::steady_clock::now();
      auto timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTimepoint);
      if (timeElapsed > options_->timeout) {
        LOGE("BackendNCCL::waitNcclGroupEnd timeout");
        return false;
      }
      sched_yield();
      NCCL_CHECK(ncclCommGetAsyncError(comm->getNcclComm(), &state));
    } while (state == ncclInProgress);
  }
  if (state != ncclSuccess) {
    NCCL_ERROR(state);
    return false;
  }
  return true;
}

ncclDataType_t BackendNCCL::getNcclDataType(DType dtype) {
  switch (dtype) {
    case DType::Float32:
      return ncclFloat32;
    case DType::Float16:
      return ncclFloat16;
    case DType::BFloat16:
      return ncclBfloat16;
    case DType::Int32:
      return ncclInt32;
    case DType::Int64:
      return ncclInt64;
    case DType::Bool:
      return ncclUint8;
    default:
      ASSERT(false && "Unsupported DType for NCCL");
      return ncclFloat32;
  }
}

ncclRedOp_t BackendNCCL::getNcclReduceOp(ReduceOpType reduceOp) {
  switch (reduceOp) {
    case SUM:
      return ncclSum;
    case PRODUCT:
      return ncclProd;
    case MIN:
      return ncclMin;
    case MAX:
      return ncclMax;
    case AVG:
      return ncclAvg;
    default:
      ASSERT(false && "Unsupported reduce operation for NCCL");
      return ncclSum;
  }
}

}  // namespace tinytorch::distributed