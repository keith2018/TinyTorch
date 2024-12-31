/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Torch.h"

using namespace TinyTorch;

// https://github.com/pytorch/examples/blob/main/mnist/main.py
class Net : public nn::Module {
 public:
  Net() { registerModules({conv1, conv2, dropout1, dropout2, fc1, fc2}); }

  Tensor forward(Tensor &x) override {
    x = conv1(x);
    x = Function::relu(x);
    x = conv2(x);
    x = Function::relu(x);
    x = Function::maxPool2d(x, 2);
    x = dropout1(x);
    x = Tensor::flatten(x, 1);
    x = fc1(x);
    x = Function::relu(x);
    x = dropout2(x);
    x = fc2(x);
    x = Function::logSoftmax(x, 1);
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

void train(nn::Module &model, data::DataLoader &dataLoader,
           optim::Optimizer &optimizer, int32_t epoch) {
  model.train();
  for (auto [batchIdx, batch] : dataLoader) {
    auto &data = batch[0];
    auto &target = batch[1];
    optimizer.zeroGrad();
    auto output = model(data);
    auto loss = Function::nllloss(output, target);
    loss.backward();
    optimizer.step();

    auto currDataCnt = batchIdx * dataLoader.batchSize() + data.shape()[0];
    auto totalDataCnt = dataLoader.dataset().size();
    LOGD("Train Epoch: %d [%d/%d %.2f%%], loss: %.6f", epoch, currDataCnt,
         totalDataCnt, 100.f * currDataCnt / (float)totalDataCnt, loss.item());
  }
}

void test(nn::Module &model, data::DataLoader &dataLoader) {
  model.eval();
  auto total = 0;
  auto correct = 0;
  withNoGrad {
    for (auto [batchIdx, batch] : dataLoader) {
      auto &data = batch[0];
      auto &target = batch[1];
      auto output = model(data);
      total += target.shape()[0];
      auto pred = output.data().argmax(1, true);
      correct += (int32_t)(pred == target.data().view(pred.shape())).sum();

      auto currDataCnt = batchIdx * dataLoader.batchSize() + data.shape()[0];
      auto totalDataCnt = dataLoader.dataset().size();
      LOGD("Test [%d/%d %.2f%%], Accuracy: [%d/%d (%.2f%%)]", currDataCnt,
           totalDataCnt, 100.f * currDataCnt / (float)totalDataCnt, correct,
           total, 100. * correct / (float)total);
    }
  }
}

void demo_mnist() {
  LOGD("demo_mnist ...");
  Timer timer;
  timer.start();

  manualSeed(0);

  // config
  auto lr = 1.f;
  auto epochs = 2;
  auto batchSize = 64;

  auto transform = std::make_shared<data::transforms::Compose>(
      data::transforms::Normalize(0.1307f, 0.3081f));

  auto dataDir = "./data/";
  auto trainDataset = std::make_shared<data::DatasetMNIST>(
      dataDir, data::DatasetMNIST::TRAIN, transform);
  auto testDataset = std::make_shared<data::DatasetMNIST>(
      dataDir, data::DatasetMNIST::TEST, transform);

  if (trainDataset->size() == 0 || testDataset->size() == 0) {
    LOGE("Dataset invalid.");
    return;
  }

  LOGD("train size: %d", trainDataset->size());
  LOGD("test size: %d", testDataset->size());

  auto trainDataloader = data::DataLoader(trainDataset, batchSize, true);
  auto testDataloader = data::DataLoader(testDataset, batchSize, true);

  auto model = Net();
  auto optimizer = optim::AdaDelta(model.parameters(), lr);
  auto scheduler = optim::lr_scheduler::StepLR(optimizer, 1, 0.7f);

  for (auto epoch = 0; epoch < epochs; epoch++) {
    train(model, trainDataloader, optimizer, epoch);
    test(model, testDataloader);
    scheduler.step();

    std::ostringstream saveName;
    saveName << "mnist_cnn_epoch_" << epoch << ".model";
    save(model, saveName.str().c_str());
  }

  timer.stop();
  LOGD("Time cost: %lld ms", timer.elapseMillis());
}
