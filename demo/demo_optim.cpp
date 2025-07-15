/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "TinyTorch.h"
#include "Utils/MathUtils.h"
#include "Utils/Timer.h"

using namespace tinytorch;

// https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-optim
void demo_optim() {
  LOGD("demo_optim ...");
  Timer timer;
  timer.start();

  manualSeed(0);

  auto x = Tensor::linspace(-PI_FLT, PI_FLT, 2000);
  auto y = x.sin();

  auto p = Tensor(Array1d<float>{1, 2, 3});
  auto xx = x.unsqueeze(-1).pow(p);

  auto model = nn::Sequential(nn::Linear(3, 1), nn::Flatten(0, 1));

  auto lossFn = nn::MSELoss(LossReduction::SUM);

  constexpr float learningRate = 1e-3f;
  auto optimizer = optim::RMSprop(model.parameters(), learningRate);
  for (int t = 0; t < 2000; t++) {
    auto yPred = model(xx);
    auto loss = lossFn(yPred, y);
    if (t % 100 == 99) {
      LOGD("t: %d, loss: %f", t, loss.item<float>());
    }

    optimizer.zeroGrad();
    loss.backward();
    optimizer.step();
  }

  auto* linearLayer = dynamic_cast<nn::Linear*>(&model[0]);
  auto biasData = linearLayer->bias().toList<float>();
  auto weightData = linearLayer->weights().toList<float>();
  LOGD("Result: y = %f + %f x + %f x^2 + %f x^3", biasData[0], weightData[0], weightData[1], weightData[2]);

  timer.mark();
  LOGD("Time cost: %lld ms", timer.elapseMillis());
}
