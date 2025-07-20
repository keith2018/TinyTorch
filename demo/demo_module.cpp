/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "TinyTorch.h"
#include "Utils/MathUtils.h"
#include "Utils/Timer.h"

using namespace tinytorch;

// https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-nn
void demo_module() {
  LOGD("demo_module ...");
  Timer timer;
  timer.start();

  manualSeed(0);

  auto x = Tensor::linspace(-PI_FLT, PI_FLT, 2000);
  auto y = x.sin();

  auto p = Tensor(Array1d<float>{1, 2, 3});
  auto xx = x.unsqueeze(-1).pow(p);

  auto model = nn::Sequential(nn::Linear(3, 1), nn::Flatten(0, 1));

  auto lossFn = nn::MSELoss(LossReduction::SUM);

  constexpr float learningRate = 1e-6f;
  for (int t = 0; t < 2000; t++) {
    auto yPred = model(xx);
    auto loss = lossFn(yPred, y);
    if (t % 100 == 99) {
      LOGD("t: %d, loss: %f", t, loss.item<float>());
    }

    model.zeroGrad();
    loss.backward();

    WithNoGrad {
      for (auto& param : model.parameters()) {
        *param -= learningRate * param->grad();
      }
    }
  }

  auto* linearLayer = dynamic_cast<nn::Linear*>(&model[0]);
  auto biasData = linearLayer->bias().toList<float>();
  auto weightData = linearLayer->weight().toList<float>();
  LOGD("Result: y = %f + %f x + %f x^2 + %f x^3", biasData[0], weightData[0], weightData[1], weightData[2]);

  timer.mark();
  LOGD("Time cost: %lld ms", timer.elapseMillis());
}
