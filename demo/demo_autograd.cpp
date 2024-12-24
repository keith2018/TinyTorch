/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Torch.h"

using namespace TinyTorch;

// https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-tensors-and-autograd
void demo_autograd() {
  LOGD("demo_autograd ...");
  Timer timer;
  timer.start();

  auto x = Tensor::linspace(-PI, PI, 2000);
  auto y = x.sin();

  auto a = Tensor::randn({}, true);
  auto b = Tensor::randn({}, true);
  auto c = Tensor::randn({}, true);
  auto d = Tensor::randn({}, true);

  constexpr float learningRate = 1e-6f;
  for (int t = 0; t < 2000; t++) {
    auto yPred = a + b * x + c * x.pow(2) + d * x.pow(3);
    auto loss = (yPred - y).pow(2).sum();

    if (t % 100 == 99) {
      LOGD("t: %d, loss: %f", t, loss.item());
    }

    loss.backward();

    withNoGrad {
      a -= learningRate * a.getGrad();
      b -= learningRate * b.getGrad();
      c -= learningRate * c.getGrad();
      d -= learningRate * d.getGrad();

      a.zeroGrad();
      b.zeroGrad();
      c.zeroGrad();
      d.zeroGrad();
    }
  }

  LOGD("Result: y = %f + %f x + %f x^2 + %f x^3", a.item(), b.item(), c.item(),
       d.item());

  timer.stop();
  LOGD("Time cost: %lld ms", timer.elapseMillis());
}
