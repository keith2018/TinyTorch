/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Torch.h"

using namespace TinyTorch;

// https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-nn
void demo_module() {
  LOGD("demo_module ...");
  Timer timer;
  timer.start();

  manualSeed(0);

  auto x = Tensor::linspace(-PI, PI, 2000);
  auto y = x.sin();

  auto p = Tensor({1, 2, 3});
  auto xx = x.unsqueeze(-1).pow(p);

  auto model = nn::Sequential(nn::Linear(3, 1), nn::Flatten(0, 1));

  auto loss_fn = nn::MSELoss(SUM);

  constexpr float learningRate = 1e-6f;
  for (int t = 0; t < 2000; t++) {
    auto yPred = model(xx);
    auto loss = loss_fn(yPred, y);
    if (t % 100 == 99) {
      LOGD("t: %d, loss: %f", t, loss.item());
    }

    model.zeroGrad();
    loss.backward();

    withNoGrad {
      for (auto& param : model.parameters()) {
        *param -= learningRate * param->getGrad();
      }
    }
  }

  auto* linearLayer = dynamic_cast<nn::Linear*>(&model[0]);
  auto& biasData = linearLayer->bias().data();
  auto& weightData = linearLayer->weights().data();
  LOGD("Result: y = %f + %f x + %f x^2 + %f x^3", biasData[0], weightData[0],
       weightData[1], weightData[2]);

  timer.mark();
  LOGD("Time cost: %lld ms", timer.elapseMillis());
}
