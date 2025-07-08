/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "TinyTorch.h"
#include "Utils/MathUtils.h"
#include "Utils/Timer.h"

using namespace tinytorch;

// https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-tensors-and-autograd
void demo_autograd() {
  LOGD("demo_autograd ...");
  Timer timer;
  timer.start();

  auto x = Tensor::linspace(-PI_FLT, PI_FLT, 2000);
  auto y = x.sin();

  Options options = options::requiresGrad(true);
  auto a = Tensor::randn({}, options);
  auto b = Tensor::randn({}, options);
  auto c = Tensor::randn({}, options);
  auto d = Tensor::randn({}, options);

  constexpr float learningRate = 1e-6f;
  for (int t = 0; t < 2000; t++) {
    auto yPred = a + b * x + c * x.pow(2) + d * x.pow(3);
    auto loss = (yPred - y).pow(2).sum();

    if (t % 100 == 99) {
      LOGD("t: %d, loss: %f", t, loss.item<float>());
    }

    loss.backward();

    WithNoGrad {
      a -= learningRate * a.grad();
      b -= learningRate * b.grad();
      c -= learningRate * c.grad();
      d -= learningRate * d.grad();

      a.zeroGrad();
      b.zeroGrad();
      c.zeroGrad();
      d.zeroGrad();
    }
  }

  LOGD("Result: y = %f + %f x + %f x^2 + %f x^3", a.item<float>(), b.item<float>(), c.item<float>(), d.item<float>());

  timer.mark();
  LOGD("Time cost: %lld ms", timer.elapseMillis());
}
