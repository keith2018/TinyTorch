/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include <Torch.h>

#include "test.h"

using namespace TinyTorch;

TEST(TEST_Autograd, backward_01) {
  Tensor x1({0.0140f, 0.5773f, 0.0469f}, true);
  Tensor x2({0.3232f, 0.4903f, 0.9395f}, true);
  auto y = Function::sin(x1) + (x1 * x2);

  Tensor grad({0.4948f, 0.8746f, 0.7076f});
  y.backward(grad);

  auto &x1_grad = x1.getGrad().data();
  EXPECT_THAT(x1_grad.shape(), ElementsAre(3));
  EXPECT_FLOAT_NEAR(x1_grad[0], 0.654671);
  EXPECT_FLOAT_NEAR(x1_grad[1], 1.161678);
  EXPECT_FLOAT_NEAR(x1_grad[2], 1.371612);

  auto &x2_grad = x2.getGrad().data();
  EXPECT_THAT(x2_grad.shape(), ElementsAre(3));
  EXPECT_FLOAT_NEAR(x2_grad[0], 0.006927);
  EXPECT_FLOAT_NEAR(x2_grad[1], 0.504907);
  EXPECT_FLOAT_NEAR(x2_grad[2], 0.033186);
}

TEST(TEST_Autograd, backward_02) {
  Tensor x({{1, -1}, {1, 1}}, true);
  auto y = x.pow(2).sum();
  y.backward();
  auto &grad = x.getGrad().data();
  EXPECT_THAT(grad.shape(), ElementsAre(2, 2));
  EXPECT_THAT(grad.toArray(), ElementsAre(2, -2, 2, 2));
}

TEST(TEST_Autograd, backward_03) {
  auto x = Tensor::linspace(-PI, PI, 100);
  auto y = Function::sin(x);

  Tensor a(Array1d{1.5f}, true);
  Tensor b(Array1d{2.2f}, true);

  auto yPred = a + b * x;
  auto loss = (yPred - y).pow(2).sum();
  loss.backward();

  auto &gradA = a.getGrad().data();
  EXPECT_THAT(gradA.shape(), ElementsAre(1));
  EXPECT_FLOAT_NEAR(gradA.toArray()[0], 299.9999389);

  auto &gradB = b.getGrad().data();
  EXPECT_THAT(gradB.shape(), ElementsAre(1));
  EXPECT_FLOAT_NEAR(gradB.toArray()[0], 1278.8518);
}

TEST(TEST_Autograd, backward_04) {
  Tensor a(Array1d{1.5f}, true);
  Tensor x(Array1d{1.f, 2.2f, 3.f});

  auto y = a * x * a;
  y.backward(Tensor::onesLike(y));

  auto grad = a.getGrad().data();
  EXPECT_THAT(grad.shape(), ElementsAre(1));
  EXPECT_THAT(grad.toArray(), ElementsAre(18.6));

  const float learningRate = 0.1f;
  withNoGrad {
    a -= learningRate * a.getGrad();
    a.zeroGrad();
  }
  y = a * x * a;
  y.backward(Tensor::onesLike(y));

  grad = a.getGrad().data();
  EXPECT_THAT(grad.shape(), ElementsAre(1));
  EXPECT_THAT(grad.toArray(), ElementsAre(-4.464));
}

TEST(TEST_Autograd, backward_flatten) {
  auto x1 = Tensor({{1, 2}, {3, 4}}, true);
  auto x2 = Tensor({{1, 2}, {3, 4}}, true);
  auto x3 = x1 * x2;
  auto y = Tensor::flatten(x3);
  y.backward(Tensor::onesLike(y));
  auto &grad1 = x1.getGrad().data();
  auto &grad2 = x2.getGrad().data();

  EXPECT_THAT(y.data().toArray(), ElementsAre(1, 4, 9, 16));
  EXPECT_THAT(grad1.toArray(), ElementsAre(1, 2, 3, 4));
  EXPECT_THAT(grad2.toArray(), ElementsAre(1, 2, 3, 4));
}
