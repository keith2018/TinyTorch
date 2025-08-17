/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "TinyTorch.h"
#include "test.h"

using namespace tinytorch;

TEST(TEST_Autograd, backward_01) {
  Options options = options::requiresGrad(true);
  Tensor x1(Array1d<float>{0.0140f, 0.5773f, 0.0469f}, options);
  Tensor x2(Array1d<float>{0.3232f, 0.4903f, 0.9395f}, options);

  auto y = function::sin(x1) + (x1 * x2);

  Tensor grad(Array1d<float>{0.4948f, 0.8746f, 0.7076f});
  y.backward(grad);

  auto &x1Grad = x1.grad();
  auto x1GradData = x1Grad.toList<float>();
  EXPECT_THAT(x1Grad.shape(), ElementsAre(3));
  EXPECT_FLT_NEAR(x1GradData[0], 0.654671);
  EXPECT_FLT_NEAR(x1GradData[1], 1.161678);
  EXPECT_FLT_NEAR(x1GradData[2], 1.371612);

  auto &x2Grad = x2.grad();
  auto x2GradData = x2Grad.toList<float>();
  EXPECT_THAT(x2Grad.shape(), ElementsAre(3));
  EXPECT_FLT_NEAR(x2GradData[0], 0.006927);
  EXPECT_FLT_NEAR(x2GradData[1], 0.504907);
  EXPECT_FLT_NEAR(x2GradData[2], 0.033186);
}

TEST(TEST_Autograd, backward_02) {
  Options options = options::requiresGrad(true);
  Tensor x(Array2d<float>{{1, -1}, {1, 1}}, options);
  auto y = x.pow(2).sum();
  y.backward();
  auto &grad = x.grad();
  EXPECT_THAT(grad.shape(), ElementsAre(2, 2));
  EXPECT_THAT(grad.toList<float>(), ElementsAre(2, -2, 2, 2));
}

TEST(TEST_Autograd, backward_03) {
  auto pi = static_cast<float>(M_PI);
  Options options = options::requiresGrad(true);
  auto x = Tensor::linspace(-pi, pi, 100);
  auto y = function::sin(x);

  Tensor a(Array1d<float>{1.5f}, options);
  Tensor b(Array1d<float>{2.2f}, options);

  auto yPred = a + b * x;
  auto loss = (yPred - y).pow(2).sum();
  loss.backward();

  auto &gradA = a.grad();
  EXPECT_THAT(gradA.shape(), ElementsAre(1));
  EXPECT_FLT_NEAR(gradA.item<float>(), 300.f);

  auto &gradB = b.grad();
  EXPECT_THAT(gradB.shape(), ElementsAre(1));
  EXPECT_FLT_NEAR(gradB.item<float>(), 1278.851);
}

TEST(TEST_Autograd, backward_04) {
  Options options = options::requiresGrad(true);
  Tensor a(Array1d<float>{1.5f}, options);
  Tensor x(Array1d<float>{1.f, 2.2f, 3.f});

  auto y = a * x * a;
  y.backward(Tensor::onesLike(y));

  auto grad = a.grad();
  EXPECT_THAT(grad.shape(), ElementsAre(1));
  EXPECT_THAT(grad.toList<float>(), ElementsAre(18.6));

  WithNoGrad {
    constexpr float learningRate = 0.1f;
    a -= learningRate * a.grad();
    a.zeroGrad();
  }
  y = a * x * a;
  y.backward(Tensor::onesLike(y));

  grad = a.grad();
  EXPECT_THAT(grad.shape(), ElementsAre(1));
  EXPECT_THAT(grad.toList<float>(), ElementsAre(-4.464));
}

TEST(TEST_Autograd, backward_flatten) {
  Options options = options::requiresGrad(true);
  auto x1 = Tensor(Array2d<float>{{1, 2}, {3, 4}}, options);
  auto x2 = Tensor(Array2d<float>{{1, 2}, {3, 4}}, options);
  auto x3 = x1 * x2;
  auto y = x3.flatten();
  y.backward(Tensor::onesLike(y));
  auto &grad1 = x1.grad();
  auto &grad2 = x2.grad();

  EXPECT_THAT(y.toList<float>(), ElementsAre(1, 4, 9, 16));
  EXPECT_THAT(grad1.toList<float>(), ElementsAre(1, 2, 3, 4));
  EXPECT_THAT(grad2.toList<float>(), ElementsAre(1, 2, 3, 4));
}
