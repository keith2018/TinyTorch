/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Distributed/DistributedProcessGroup.h"
#include "TinyTorch.h"
#include "Utils/CUDAUtils.h"
#include "Utils/Timer.h"

using namespace tinytorch;

namespace tinytorch::distributed {

static void demoAllReduce(int localRank, int rank, int worldSize) {
  LOGD("demoAllReduce: %d, %d, %d", localRank, rank, worldSize);

  auto dpg = DistributedProcessGroup::getInstance();

  std::string initMethod = "tcp://127.0.0.1:29500";  // master node ip & port
  bool success = dpg->initProcessGroup(NCCL, initMethod, rank, worldSize);
  if (!success) {
    LOGE("InitProcessGroup failed");
    return;
  }
  cuda::setDevice(localRank);

  std::vector<float> hostData(64, static_cast<float>(rank + 1));
  auto tensor = Tensor(hostData, Options({DeviceType::CUDA, static_cast<DeviceIndex>(localRank)}, DType::Float32));

  std::cout << "Rank " << rank << " Start All Reduce..." << std::endl;

  AllReduceOptions opts;
  opts.reduceOp = ReduceOpType::SUM;

  std::vector<Tensor> tensors = {tensor};
  auto work = dpg->allReduce(tensors);

  bool waitRet = work->wait();
  std::cout << "wait ret: " << waitRet << std::endl;

  std::vector<float> result = tensor.toList<float>();
  auto expected = worldSize * (worldSize + 1) / 2;
  bool correct = std::abs(result[0] - static_cast<float>(expected)) < 1e-5;

  std::cout << "Rank " << rank << " correct: " << (correct ? "✓" : "✗") << " (expected: " << expected
            << ", result: " << result[0] << ")" << std::endl;
}

}  // namespace tinytorch::distributed

void demo_nccl(int argc, char** argv) {
  LOGD("demo_nccl ...");
  Timer timer;
  timer.start();

  ASSERT(argc == 4);
  int localRank = std::stoi(argv[1]);
  int rank = std::stoi(argv[2]);
  int worldSize = std::stoi(argv[3]);

  int deviceCount = cuda::getDeviceCount();
  LOGD("deviceCount: %d", deviceCount);
  if (localRank >= deviceCount) {
    LOGE("Not enough GPUs available. Required: %d, Available: %d", (localRank + 1), deviceCount);
    return;
  }

  distributed::demoAllReduce(localRank, rank, worldSize);

  timer.mark();
  LOGD("Time cost: %lld ms", timer.elapseMillis());
}
