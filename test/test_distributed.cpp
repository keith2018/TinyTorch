/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#if defined(USE_CUDA) && defined(USE_NCCL)

#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

#include "Distributed/DistributedProcessGroup.h"
#include "TinyTorch.h"
#include "Utils/CUDAUtils.h"
#include "test.h"

using namespace tinytorch;
using namespace tinytorch::distributed;

namespace {

constexpr int kBasePort = 29600;
std::atomic<int> gPortOffset{0};

int getNextPort() { return kBasePort + gPortOffset.fetch_add(1); }

bool runMultiProcess(int worldSize, const std::function<void(int rank, int worldSize, int port)>& fn) {
  int port = getNextPort();
  std::vector<pid_t> children;
  children.reserve(worldSize);

  for (int rank = 0; rank < worldSize; rank++) {
    pid_t pid = fork();
    if (pid < 0) {
      std::cerr << "fork failed for rank " << rank << std::endl;
      return false;
    }
    if (pid == 0) {
      // child process
      try {
        fn(rank, worldSize, port);
      } catch (const std::exception& e) {
        std::cerr << "Rank " << rank << " exception: " << e.what() << std::endl;
        _exit(1);
      } catch (...) {
        std::cerr << "Rank " << rank << " unknown exception" << std::endl;
        _exit(1);
      }
      _exit(0);
    }
    children.push_back(pid);
  }

  // parent: wait for all children
  bool allSuccess = true;
  for (pid_t pid : children) {
    int status = 0;
    waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      allSuccess = false;
    }
  }
  return allSuccess;
}

int getDeviceCountNoCudaInit() {
  // try /proc/driver/nvidia/gpus - each GPU has a numbered directory
  int count = 0;
  for (int i = 0; i < 16; i++) {
    std::string path = "/proc/driver/nvidia/gpus/" + std::to_string(i) + "/information";
    if (access(path.c_str(), F_OK) == 0) {
      count++;
    } else {
      break;
    }
  }
  if (count > 0) {
    return count;
  }
  // fallback: use nvidia-smi
  FILE* fp = popen("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l", "r");
  if (fp) {
    char buf[32] = {};
    if (fgets(buf, sizeof(buf), fp)) {
      count = std::atoi(buf);
    }
    pclose(fp);
  }
  return count;
}

#define SKIP_IF_NOT_ENOUGH_GPUS()                             \
  do {                                                        \
    int deviceCount = getDeviceCountNoCudaInit();             \
    if (deviceCount < 2) {                                    \
      GTEST_SKIP() << "Need >= 2 GPUs, have " << deviceCount; \
    }                                                         \
  } while (0)

}  // namespace

TEST(TEST_Distributed, AllReduce_SUM) {
  SKIP_IF_NOT_ENOUGH_GPUS();

  int worldSize = 2;
  bool ok = runMultiProcess(worldSize, [](int rank, int worldSize, int port) {
    cuda::setDevice(rank);
    auto dpg = std::make_shared<DistributedProcessGroup>();
    std::string initMethod = "tcp://127.0.0.1:" + std::to_string(port);
    if (!dpg->initProcessGroup(NCCL, initMethod, rank, worldSize)) {
      _exit(1);
    }

    // each rank has tensor filled with (rank + 1)
    std::vector<float> data = {1.0f * (rank + 1), 2.0f * (rank + 1), 3.0f * (rank + 1), 4.0f * (rank + 1)};
    auto tensor = Tensor(data, Options({DeviceType::CUDA, static_cast<DeviceIndex>(rank)}, DType::Float32));

    std::vector<Tensor> tensors = {tensor};
    auto work = dpg->allReduce(tensors, SUM);
    if (!work || !work->wait()) {
      _exit(1);
    }

    // expected: sum of (rank+1) for all ranks = worldSize*(worldSize+1)/2
    auto expectedScale = static_cast<float>(worldSize * (worldSize + 1) / 2);
    auto result = tensor.toList<float>();
    for (int i = 0; i < 4; i++) {
      float expected = static_cast<float>(i + 1) * expectedScale;
      if (std::fabs(result[i] - expected) > 1e-3f) {
        _exit(1);
      }
    }
    dpg->destroyProcessGroup();
  });
  EXPECT_TRUE(ok);
}

TEST(TEST_Distributed, AllReduce_AVG) {
  SKIP_IF_NOT_ENOUGH_GPUS();

  int worldSize = 2;
  bool ok = runMultiProcess(worldSize, [](int rank, int worldSize, int port) {
    cuda::setDevice(rank);
    auto dpg = std::make_shared<DistributedProcessGroup>();
    std::string initMethod = "tcp://127.0.0.1:" + std::to_string(port);
    if (!dpg->initProcessGroup(NCCL, initMethod, rank, worldSize)) {
      _exit(1);
    }

    std::vector<float> data = {10.0f * (static_cast<float>(rank) + 1)};
    auto tensor = Tensor(data, Options({DeviceType::CUDA, static_cast<DeviceIndex>(rank)}, DType::Float32));

    std::vector<Tensor> tensors = {tensor};
    auto work = dpg->allReduce(tensors, AVG);
    if (!work || !work->wait()) {
      _exit(1);
    }

    // AVG of (10, 20) = 15
    float expectedAvg = 10.0f * static_cast<float>(worldSize + 1) / 2.0f;
    auto result = tensor.toList<float>();
    if (std::fabs(result[0] - expectedAvg) > 1e-3f) {
      _exit(1);
    }
    dpg->destroyProcessGroup();
  });
  EXPECT_TRUE(ok);
}

TEST(TEST_Distributed, Broadcast) {
  SKIP_IF_NOT_ENOUGH_GPUS();

  int worldSize = 2;
  bool ok = runMultiProcess(worldSize, [](int rank, int worldSize, int port) {
    cuda::setDevice(rank);
    auto dpg = std::make_shared<DistributedProcessGroup>();
    std::string initMethod = "tcp://127.0.0.1:" + std::to_string(port);
    if (!dpg->initProcessGroup(NCCL, initMethod, rank, worldSize)) {
      _exit(1);
    }

    // rank 0 has meaningful data, others have zeros
    Tensor tensor;
    if (rank == 0) {
      std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
      tensor = Tensor(data, Options({DeviceType::CUDA, static_cast<DeviceIndex>(rank)}, DType::Float32));
    } else {
      tensor = Tensor::zeros({4}, Options({DeviceType::CUDA, static_cast<DeviceIndex>(rank)}, DType::Float32));
    }

    std::vector<Tensor> tensors = {tensor};
    auto work = dpg->broadcast(tensors, 0);
    if (!work || !work->wait()) {
      _exit(1);
    }

    // all ranks should have rank 0's data
    auto result = tensor.toList<float>();
    for (int i = 0; i < 4; i++) {
      if (std::fabs(result[i] - static_cast<float>(i + 1)) > 1e-3f) {
        _exit(1);
      }
    }
    dpg->destroyProcessGroup();
  });
  EXPECT_TRUE(ok);
}

TEST(TEST_Distributed, AllGather) {
  SKIP_IF_NOT_ENOUGH_GPUS();

  int worldSize = 2;
  bool ok = runMultiProcess(worldSize, [](int rank, int worldSize, int port) {
    cuda::setDevice(rank);
    auto dpg = std::make_shared<DistributedProcessGroup>();
    std::string initMethod = "tcp://127.0.0.1:" + std::to_string(port);
    if (!dpg->initProcessGroup(NCCL, initMethod, rank, worldSize)) {
      _exit(1);
    }

    // each rank has a tensor with its own rank value
    std::vector<float> inputData = {static_cast<float>(rank + 1) * 10.0f, static_cast<float>(rank + 1) * 20.0f};
    auto input = Tensor(inputData, Options({DeviceType::CUDA, static_cast<DeviceIndex>(rank)}, DType::Float32));

    // prepare output tensors (one per rank)
    std::vector<Tensor> outputs;
    outputs.reserve(worldSize);
    for (int i = 0; i < worldSize; i++) {
      outputs.push_back(
          Tensor::zeros({2}, Options({DeviceType::CUDA, static_cast<DeviceIndex>(rank)}, DType::Float32)));
    }

    std::vector<std::vector<Tensor>> outputTensors = {outputs};
    std::vector<Tensor> inputTensors = {input};
    auto work = dpg->allGather(outputTensors, inputTensors);
    if (!work || !work->wait()) {
      _exit(1);
    }

    // verify: outputs[i] should contain rank i's data
    for (int i = 0; i < worldSize; i++) {
      auto result = outputTensors[0][i].toList<float>();
      float expected0 = static_cast<float>(i + 1) * 10.0f;
      float expected1 = static_cast<float>(i + 1) * 20.0f;
      if (std::fabs(result[0] - expected0) > 1e-3f || std::fabs(result[1] - expected1) > 1e-3f) {
        _exit(1);
      }
    }
    dpg->destroyProcessGroup();
  });
  EXPECT_TRUE(ok);
}

TEST(TEST_Distributed, Barrier) {
  SKIP_IF_NOT_ENOUGH_GPUS();

  int worldSize = 2;
  bool ok = runMultiProcess(worldSize, [](int rank, int worldSize, int port) {
    cuda::setDevice(rank);
    auto dpg = std::make_shared<DistributedProcessGroup>();
    std::string initMethod = "tcp://127.0.0.1:" + std::to_string(port);
    if (!dpg->initProcessGroup(NCCL, initMethod, rank, worldSize)) {
      _exit(1);
    }

    // barrier should complete without hanging
    dpg->barrier(false, {static_cast<int64_t>(rank)});
    dpg->destroyProcessGroup();
  });
  EXPECT_TRUE(ok);
}

#endif  // defined(USE_CUDA) && defined(USE_NCCL)
