/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Data/DataLoader.h"
#include "Data/DatasetMNIST.h"
#include "TinyTorch.h"
#include "Utils/CUDAUtils.h"
#include "Utils/RandomGenerator.h"
#include "Utils/Timer.h"

using namespace tinytorch;

// https://github.com/pytorch/examples/blob/main/mnist/main.py
class Net : public nn::Module {
 public:
  Net() {
    registerModules({
        {"conv1", conv1},
        {"conv2", conv2},
        {"dropout1", dropout1},
        {"dropout2", dropout2},
        {"fc1", fc1},
        {"fc2", fc2},
    });
  }

  Tensor forward(const Tensor &input) override {
    auto x = conv1(input);
    x = function::relu(x);
    x = conv2(x);
    x = function::relu(x);
    x = function::maxPool2d(x, 2, 2);
    x = dropout1(x);
    x = function::flatten(x, 1);
    x = fc1(x);
    x = function::relu(x);
    x = dropout2(x);
    x = fc2(x);
    x = function::logSoftmax(x, 1);
    return x;
  }

 private:
  nn::Conv2D conv1{1, 32, 3, 1};
  nn::Conv2D conv2{32, 64, 3, 1};
  nn::Dropout dropout1{0.25};
  nn::Dropout dropout2{0.5};
  nn::Linear fc1{9216, 128};
  nn::Linear fc2{128, 10};
};

// Training settings
struct TrainArgs {
  // input batch size for training (default: 64)
  int32_t batchSize = 64;

  // input batch size for testing (default: 1000)
  int32_t testBatchSize = 1000;

  // number of epochs to train (default: 1)
  int32_t epochs = 1;

  // learning rate (default: 1.0)
  float lr = 1.f;

  // Learning rate step gamma (default: 0.7)
  float gamma = 0.7f;

  // disables CUDA training
  bool noCuda = false;

  // quickly check a single pass
  bool dryRun = false;

  // random seed (default: 1)
  unsigned long seed = 1;

  // how many batches to wait before logging training status
  int32_t logInterval = 10;

  // load pretrained model
  std::string pretrained;

  // for saving the current model
  bool saveModel = true;
};

static void train(TrainArgs &args, nn::Module &model, Device device, data::DataLoader &dataLoader,
                  optim::Optimizer &optimizer, int32_t epoch) {
  model.train();
  Timer timer;
  timer.start();
  for (auto [batchIdx, batch] : dataLoader) {
    auto data = batch[0].to(device);
    auto target = batch[1].to(device);
    optimizer.zeroGrad();
    auto output = model(data);
    auto loss = function::nllLoss(output, target);
    loss.backward();
    optimizer.step();

    if (batchIdx % args.logInterval == 0) {
      timer.mark();
      auto currDataCnt = batchIdx * dataLoader.batchSize();
      auto totalDataCnt = dataLoader.dataset().size();
      auto elapsed = (float)timer.elapseMillis() / 1000.f;  // seconds
      LOGD("Train Epoch: %d [%d/%d (%.0f%%)] Loss: %.6f, Elapsed: %.2fs", epoch, currDataCnt, totalDataCnt,
           100.f * currDataCnt / (float)totalDataCnt, loss.item<float>(), elapsed);

      if (args.dryRun) {
        break;
      }
    }
  }
}

static void test(nn::Module &model, Device device, data::DataLoader &dataLoader) {
  model.eval();
  Timer timer;
  timer.start();
  auto testLoss = 0.f;
  auto correct = 0;
  WithNoGrad {
    for (auto [batchIdx, batch] : dataLoader) {
      auto data = batch[0].to(device);
      auto target = batch[1].to(device);
      auto output = model(data);
      testLoss += function::nllLoss(output, target, LossReduction::SUM).item<float>();
      auto pred = output.argmax(1, true);
      correct += static_cast<int>((pred == target.view(pred.shape())).to(DType::Float32).sum().item<float>());
    }
  }
  auto total = dataLoader.dataset().size();
  testLoss /= (float)total;
  timer.mark();
  auto elapsed = (float)timer.elapseMillis() / 1000.f;  // seconds
  LOGD(
      "Test set: Average loss: %.4f, Accuracy: %d/%d (%.0f%%), Elapsed: "
      "%.2fs",
      testLoss, correct, total, 100. * correct / (float)total, elapsed);
}

void demo_mnist() {
  LOGD("demo_mnist ...");

  TrainArgs args;
  manualSeed(args.seed);

  auto useCuda = (!args.noCuda) && cuda::deviceAvailable();
  Device device = useCuda ? Device(DeviceType::CUDA, 0) : Device(DeviceType::CPU);
  LOGD("Train with device: %s", useCuda ? "CUDA" : "CPU");

  auto transform = std::make_shared<data::transforms::Compose>(data::transforms::Normalize(0.1307f, 0.3081f));

  auto dataDir = "./data/";
  auto trainDataset = std::make_shared<data::DatasetMNIST>(dataDir, data::DatasetMNIST::TRAIN, transform);
  auto testDataset = std::make_shared<data::DatasetMNIST>(dataDir, data::DatasetMNIST::TEST, transform);

  if (trainDataset->size() == 0 || testDataset->size() == 0) {
    LOGE("Dataset invalid.");
    return;
  }

  auto trainDataloader = data::DataLoader(trainDataset, args.batchSize);
  auto testDataloader = data::DataLoader(testDataset, args.testBatchSize);

  auto model = Net();
  model.to(device);

  bool loadSuccess = false;
  if (!args.pretrained.empty()) {
    LOGD("Load pretrained model: %s", args.pretrained.c_str());
    loadSuccess = load(model, args.pretrained);
  }
  if (!loadSuccess) {
    model.initParameters();
  }

  auto optimizer = optim::AdaDelta(model.parameters(), args.lr);
  auto scheduler = optim::lr_scheduler::StepLR(optimizer, 1, args.gamma);

  Timer timer;
  timer.start();

  for (auto epoch = 1; epoch < args.epochs + 1; epoch++) {
    train(args, model, device, trainDataloader, optimizer, epoch);
    test(model, device, testDataloader);
    scheduler.step();
  }

  if (args.saveModel) {
    save(model, "mnist_cnn.model");
  }

  timer.mark();
  LOGD("Total Time cost: %lld ms", timer.elapseMillis());
}
