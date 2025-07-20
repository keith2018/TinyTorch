/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "TinyTorch.h"
#include "Utils/MathUtils.h"
#include "test.h"

using namespace tinytorch;

TEST(TEST_Function, func_add) {
  Options options = options::requiresGrad(true);
  Tensor a(Array1d<float>{1, 2, 3}, options);
  Tensor b(Array1d<float>{4, 5, 6}, options);
  auto y = function::add(a, b, 0.5f);
  EXPECT_THAT(y.toList<float>(), ElementsAre(3, 4.5, 6));

  y.backward(Tensor::onesLike(y));
  EXPECT_THAT(a.grad().toList<float>(), ElementsAre(1, 1, 1));
  EXPECT_THAT(b.grad().toList<float>(), ElementsAre(1, 1, 1));
}

TEST(TEST_Function, func_sub) {
  Options options = options::requiresGrad(true);
  Tensor a(Array1d<float>{1, 2, 3}, options);
  Tensor b(Array1d<float>{4, 5, 6}, options);
  auto y = function::sub(a, b);
  EXPECT_THAT(y.toList<float>(), ElementsAre(-3, -3, -3));

  y.backward(Tensor::onesLike(y));
  EXPECT_THAT(a.grad().toList<float>(), ElementsAre(1, 1, 1));
  EXPECT_THAT(b.grad().toList<float>(), ElementsAre(-1, -1, -1));
}

TEST(TEST_Function, func_mul) {
  Options options = options::requiresGrad(true);
  Tensor a(Array1d<float>{1, 2, 3}, options);
  Tensor b(Array1d<float>{4, 5, 6}, options);
  auto y = function::mul(a, b);
  EXPECT_THAT(y.toList<float>(), ElementsAre(4, 10, 18));

  y.backward(Tensor::onesLike(y));
  EXPECT_THAT(a.grad().toList<float>(), ElementsAre(4, 5, 6));
  EXPECT_THAT(b.grad().toList<float>(), ElementsAre(1, 2, 3));
}

TEST(TEST_Function, func_div) {
  Options options = options::requiresGrad(true);
  Tensor a(Array1d<float>{1, 2, 3}, options);
  Tensor b(Array1d<float>{4, 5, 6}, options);
  auto y = function::div(a, b);
  EXPECT_THAT(y.toList<float>(), ElementsAre(0.25, 0.4, 0.5));

  y.backward(Tensor::onesLike(y));
  EXPECT_THAT(a.grad().toList<float>(), ElementsAre(0.2500, 0.2000, 1.0 / 6));
  EXPECT_THAT(b.grad().toList<float>(), ElementsAre(-0.0625, -0.0800, -1.0 / 12));
}

TEST(TEST_Function, func_sin) {
  Options options = options::requiresGrad(true);
  Tensor x(Array1d<float>{0.0f, PI_FLT / 2, PI_FLT}, options);
  auto y = function::sin(x);
  EXPECT_THAT(y.toList<float>(), ElementsAre(0, 1, std::sin(PI_FLT)));

  y.backward(Tensor::onesLike(y));
  EXPECT_THAT(x.grad().toList<float>(), ElementsAre(1, std::cos(PI_FLT / 2), -1));
}

TEST(TEST_Function, func_cos) {
  Options options = options::requiresGrad(true);
  Tensor x(Array1d<float>{0.0f, PI_FLT / 2, PI_FLT}, options);
  auto y = function::cos(x);
  EXPECT_THAT(y.toList<float>(), ElementsAre(1, std::cos(PI_FLT / 2), std::cos(PI_FLT)));

  y.backward(Tensor::onesLike(y));
  EXPECT_TRUE(VectorNear(x.grad().toList<float>(), {0, -std::sin(PI_FLT / 2), 0}));
}

TEST(TEST_Function, func_pow) {
  Options options = options::requiresGrad(true);
  Tensor x1(Array1d<float>{2.0f, 3.0f, 4.0f}, options);
  Tensor x2(Array1d<float>{3.0f, 3.0f, 3.0f}, options);
  auto y = function::pow(x1, x2);
  EXPECT_THAT(y.toList<float>(), ElementsAre(8, 27, 64));

  y.backward(Tensor::onesLike(y));
  EXPECT_THAT(x1.grad().toList<float>(), ElementsAre(12, 27, 48));
  EXPECT_TRUE(VectorNear(x2.grad().toList<float>(), {5.5452, 29.6625, 88.7228}));

  // scalar
  x1 = Tensor(Array1d<float>{2.0f, 3.0f, 4.0f}, options);
  y = function::pow(x1, Tensor::scalar(3.f));
  EXPECT_THAT(y.toList<float>(), ElementsAre(8, 27, 64));

  y.backward(Tensor::onesLike(y));
  EXPECT_THAT(x1.grad().toList<float>(), ElementsAre(12, 27, 48));
}

TEST(TEST_Function, func_sum1) {
  Options options = options::requiresGrad(true);
  Tensor x(Array1d<float>{1.0f, 2.0f, 3.0f}, options);
  auto y = function::sum(x);
  EXPECT_THAT(y.toList<float>(), ElementsAre(6));

  y.backward(Tensor::onesLike(y));
  EXPECT_THAT(x.grad().toList<float>(), ElementsAre(1, 1, 1));
}

TEST(TEST_Function, func_sum2) {
  Options options = options::requiresGrad(true);
  Tensor x(Array1d<float>{1.0f, 2.0f, 3.0f}, options);
  auto y = function::sum(x);
  EXPECT_THAT(y.toList<float>(), ElementsAre(6));

  y.backward();
  EXPECT_THAT(x.grad().toList<float>(), ElementsAre(1, 1, 1));
}

TEST(TEST_Function, func_relu) {
  Options options = options::requiresGrad(true);
  Tensor x(Array2d<float>{{-1.0, 2.0}, {3.0, -4.0}}, options);
  auto y = function::relu(x);
  y.sum().backward();
  EXPECT_THAT(y.toList<float>(), ElementsAre(0, 2, 3, 0));
  EXPECT_THAT(x.grad().toList<float>(), ElementsAre(0, 1, 1, 0));
}

TEST(TEST_Function, func_gelu) {
  Tensor x(Array1d<float>{-1., -0.5, 0.5, 1.});
  auto y = function::gelu(x);
  EXPECT_TRUE(VectorNear(y.toList<float>(), {-0.1587, -0.1543, 0.3457, 0.8413}));
  // y.backward(Tensor::onesLike(y));
  // EXPECT_TRUE(VectorNear(x.grad().toList<float>(), {-0.0833, 0.1325, 0.8675, 1.0833}));
}

TEST(TEST_Function, func_silu) {
  Tensor x(Array1d<float>{-1., -0.5, 0.5, 1.});
  auto y = function::silu(x);
  EXPECT_TRUE(VectorNear(y.toList<float>(), {-0.2689, -0.1888, 0.3112, 0.7311}));
  // y.backward(Tensor::onesLike(y));
  // EXPECT_TRUE(VectorNear(x.grad().toList<float>(), {0.0723, 0.2600, 0.7400, 0.9277}));
}

TEST(TEST_Function, func_softmax) {
  Options options = options::requiresGrad(true);
  auto input = Tensor(Array1d<float>{1.1, 1.2, 1.3, 1.6}, options);
  auto output = function::softmax(input, 0);
  EXPECT_TRUE(VectorNear(output.toList<float>(), {0.2010, 0.2221, 0.2455, 0.3314}));

  output.backward(input);
  EXPECT_TRUE(VectorNear(input.grad().toList<float>(), {-0.0476, -0.0304, -0.0091, 0.0872}));
}

TEST(TEST_Function, func_logSoftmax) {
  Options options = options::requiresGrad(true);
  auto input = Tensor(Array1d<float>{1.1, 1.2, 1.3, 1.6}, options);
  auto output = function::logSoftmax(input, 0);
  EXPECT_TRUE(VectorNear(output.toList<float>(), {-1.6045, -1.5045, -1.4045, -1.1045}));
  output.backward(input);
  EXPECT_TRUE(VectorNear(input.grad().toList<float>(), {0.0548, 0.0449, 0.0234, -0.1232}));

  input = Tensor(Array2d<float>{{1, 2}, {3, 4}}, options);
  output = function::logSoftmax(input, 0);
  EXPECT_TRUE(VectorNear(output.toList<float>(), {-2.1269, -2.1269, -0.1269, -0.1269}));
  output.backward(Tensor::onesLike(output));
  EXPECT_TRUE(VectorNear(input.grad().toList<float>(), {0.7616, 0.7616, -0.7616, -0.7616}));

  input = Tensor(Array2d<float>{{1, 2}, {3, 4}}, options);
  output = function::logSoftmax(input, 1);
  EXPECT_TRUE(VectorNear(output.toList<float>(), {-1.3133, -0.3133, -1.3133, -0.3133}));
  output.backward(Tensor::onesLike(output));
  EXPECT_TRUE(VectorNear(input.grad().toList<float>(), {0.4621, -0.4621, 0.4621, -0.4621}));
}

TEST(TEST_Function, func_mseLoss_none) {
  Options options = options::requiresGrad(true);
  Tensor x(Array2d<float>{{-0.3089f, 0.5301f, -0.0245f}, {1.5852f, 0.8954f, 0.7485f}}, options);
  Tensor y(Array2d<float>{{0.8397f, 1.7990f, -0.2738f}, {-0.8910f, -0.6746f, 0.3419f}});
  auto loss = function::mseLoss(x, y, LossReduction::NONE);
  EXPECT_TRUE(
      VectorNear(loss.toList<float>(), {1.31928194, 1.6101073, 0.0621504858, 6.13156557, 2.46489978, 0.165323555}));
  loss.backward(Tensor::onesLike(loss));
  EXPECT_TRUE(VectorNear(x.grad().toList<float>(), {-2.2972, -2.5378, 0.498599976, 4.95239973, 3.13999987, 0.8132}));
}

TEST(TEST_Function, func_mseLoss_mean) {
  Options options = options::requiresGrad(true);
  Tensor x(Array2d<float>{{-0.3089f, 0.5301f, -0.0245f}, {1.5852f, 0.8954f, 0.7485f}}, options);
  Tensor y(Array2d<float>{{0.8397f, 1.7990f, -0.2738f}, {-0.8910f, -0.6746f, 0.3419f}});
  auto loss = function::mseLoss(x, y, LossReduction::MEAN);
  EXPECT_FLOAT_EQ(loss.item<float>(), 1.95888805);
  loss.backward();
  EXPECT_TRUE(VectorNear(x.grad().toList<float>(),
                         {-0.382866651, -0.422966689, 0.0831, 0.825399935, 0.523333311, 0.135533333}));
}

TEST(TEST_Function, func_mseLoss_sum) {
  Options options = options::requiresGrad(true);
  Tensor x(Array2d<float>{{-0.3089f, 0.5301f, -0.0245f}, {1.5852f, 0.8954f, 0.7485f}}, options);
  Tensor y(Array2d<float>{{0.8397f, 1.7990f, -0.2738f}, {-0.8910f, -0.6746f, 0.3419f}});
  auto loss = function::mseLoss(x, y, LossReduction::SUM);
  EXPECT_FLOAT_EQ(loss.item<float>(), 11.7533283);
  loss.backward();
  EXPECT_TRUE(VectorNear(x.grad().toList<float>(), {-2.2972, -2.5378, 0.498599976, 4.95239973, 3.13999987, 0.8132}));
}

TEST(TEST_Function, func_nllloss) {
  Options options = options::requiresGrad(true);
  auto input = Tensor(Array2d<float>{{0.1, 0.2, 0.7}, {0.3, 0.4, 0.3}}, options);
  auto target = Tensor(Array1d<int64_t>{2, 1});
  auto loss = function::nllLoss(input, target, LossReduction::NONE);
  EXPECT_TRUE(VectorNear(loss.toList<float>(), {-0.7, -0.4}));
  loss.backward(Tensor::onesLike(loss));
  EXPECT_TRUE(VectorNear(input.grad().toList<float>(), {0, 0, -1, 0, -1, 0}));

  input = Tensor(Array2d<float>{{0.1, 0.2, 0.7}, {0.3, 0.4, 0.3}}, options);
  target = Tensor(Array1d<int64_t>{2, 1});
  loss = function::nllLoss(input, target, LossReduction::MEAN);
  EXPECT_FLT_NEAR(loss.item<float>(), -0.55);
  loss.backward(Tensor::onesLike(loss));
  EXPECT_TRUE(VectorNear(input.grad().toList<float>(), {0, 0, -0.5, 0, -0.5, 0}));

  input = Tensor(Array2d<float>{{0.1, 0.2, 0.7}, {0.3, 0.4, 0.3}}, options);
  target = Tensor(Array1d<int64_t>{2, 1});
  loss = function::nllLoss(input, target, LossReduction::SUM);
  EXPECT_FLT_NEAR(loss.item<float>(), -1.1);
  loss.backward(Tensor::onesLike(loss));
  EXPECT_TRUE(VectorNear(input.grad().toList<float>(), {0, 0, -1, 0, -1, 0}));
}

TEST(TEST_Function, func_linear) {
  Options options = options::requiresGrad(true);
  Tensor x(Array2d<float>{{-0.3089f, 0.5301f, -0.0245f}, {1.5852f, 0.8954f, 0.7485f}}, options);
  Tensor w(Array2d<float>{{0.8397f, 1.7990f, -0.2738f}, {-0.8910f, -0.6746f, 0.3419f}}, options);
  Tensor b(Array1d<float>{-0.9601f, -1.4163f}, options);

  auto y = function::linear(x, w, b).sum();
  EXPECT_FLOAT_EQ(y.item<float>(), -3.1661377);

  y.backward();
  EXPECT_TRUE(
      VectorNear(x.grad().toList<float>(), {-0.0512999892, 1.1244, 0.0681000054, -0.0512999892, 1.1244, 0.0681000054}));
  EXPECT_TRUE(VectorNear(w.grad().toList<float>(), {1.2763, 1.42549992, 0.724, 1.2763, 1.42549992, 0.724}));
  EXPECT_THAT(b.grad().toList<float>(), ElementsAre(2., 2.));
}

TEST(TEST_Function, func_dropout) {
  Options options = options::requiresGrad(true);
  auto input = Tensor::ones({100, 10}, options);
  auto p = 0.3f;
  auto output = function::dropout(input, p, true);
  EXPECT_EQ(output.shape(), input.shape());
  auto zeroCnt = (output == 0).to(DType::Float32).sum().item<float>();
  EXPECT_NEAR(zeroCnt / input.numel(), p, 0.1);

  output = function::dropout(input, p, false);
  EXPECT_EQ(output.toList<float>(), input.toList<float>());

  output = function::dropout(input, p, true);
  output.sum().backward();
  zeroCnt = (input.grad() == 0).to(DType::Float32).sum().item<float>();
  EXPECT_NEAR(zeroCnt / input.numel(), p, 0.1);
}

TEST(TEST_Function, func_maxpool2d_01) {
  auto input = Tensor(Array2d<float>{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}});
  input.reshape_({1, 1, 4, 4});
  input.setRequiresGrad(true);

  auto output = function::maxPool2d(input, 2, 2);
  EXPECT_THAT(output.shape(), ElementsAre(1, 1, 2, 2));
  EXPECT_THAT(output.toList<float>(), ElementsAre(6, 8, 14, 16));

  output = function::maxPool2d(input, 3, 1);
  EXPECT_THAT(output.shape(), ElementsAre(1, 1, 2, 2));
  EXPECT_THAT(output.toList<float>(), ElementsAre(11, 12, 15, 16));

  output = function::maxPool2d(input, 3, 2);
  EXPECT_THAT(output.shape(), ElementsAre(1, 1, 1, 1));
  EXPECT_THAT(output.toList<float>(), ElementsAre(11));

  output = function::maxPool2d(input, 3, 2, 1);
  EXPECT_THAT(output.shape(), ElementsAre(1, 1, 2, 2));
  EXPECT_THAT(output.toList<float>(), ElementsAre(6, 8, 14, 16));

  output.backward(Tensor::onesLike(output));
  EXPECT_THAT(input.grad().toList<float>(), ElementsAre(0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1));
}

TEST(TEST_Function, func_maxpool2d_02) {
  auto input = Tensor(Array2d<float>{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}});
  input.reshape_({1, 1, 4, 4});
  input.setRequiresGrad(true);

  auto output = function::maxPool2d(input, 2, 2, 1);
  EXPECT_THAT(output.shape(), ElementsAre(1, 1, 3, 3));
  EXPECT_THAT(output.toList<float>(), ElementsAre(1, 3, 4, 9, 11, 12, 13, 15, 16));

  output.backward(Tensor::onesLike(output));
  EXPECT_THAT(input.grad().toList<float>(), ElementsAre(1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1));
}

TEST(TEST_Function, func_conv2d_01) {
  auto input = Tensor(Array2d<float>{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}});
  input.reshape_({1, 1, 4, 4});
  input.setRequiresGrad(true);

  auto weight = Tensor(Array2d<float>{{1, 0}, {0, -1}});
  weight.reshape_({1, 1, 2, 2});
  weight.setRequiresGrad(true);

  auto bias = Tensor(Array1d<float>{2.f});
  bias.setRequiresGrad(true);

  auto output = function::conv2d(input, weight, bias);
  EXPECT_THAT(output.shape(), ElementsAre(1, 1, 3, 3));
  EXPECT_THAT(output.toList<float>(), ElementsAre(-3, -3, -3, -3, -3, -3, -3, -3, -3));

  output.backward(Tensor::onesLike(output));
  EXPECT_THAT(input.grad().toList<float>(), ElementsAre(1, 1, 1, 0, 1, 0, 0, -1, 1, 0, 0, -1, 0, -1, -1, -1));
  EXPECT_THAT(weight.grad().toList<float>(), ElementsAre(54, 63, 90, 99));
  EXPECT_THAT(bias.grad().toList<float>(), ElementsAre(9));
}

TEST(TEST_Function, func_conv2d_02) {
  auto input = Tensor(Array1d<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  input.reshape_({2, 1, 4, 4});
  input.setRequiresGrad(true);

  auto weight = Tensor(Array2d<float>{{1, 2}, {0, -1}});
  weight.reshape_({1, 1, 2, 2});
  weight.setRequiresGrad(true);

  auto bias = Tensor(Array1d<float>{2.f});
  bias.setRequiresGrad(true);

  auto output = function::conv2d(input, weight, bias);
  EXPECT_THAT(output.shape(), ElementsAre(2, 1, 3, 3));
  EXPECT_THAT(output.toList<float>(), ElementsAre(1, 3, 5, 9, 11, 13, 17, 19, 21, 1, 3, 5, 9, 11, 13, 17, 19, 21));

  output.backward(Tensor::onesLike(output));
  EXPECT_THAT(input.grad().toList<float>(), ElementsAre(1, 3, 3, 2, 1, 2, 2, 1, 1, 2, 2, 1, 0, -1, -1, -1, 1, 3, 3, 2,
                                                        1, 2, 2, 1, 1, 2, 2, 1, 0, -1, -1, -1));
  EXPECT_THAT(weight.grad().toList<float>(), ElementsAre(108, 126, 180, 198));
  EXPECT_THAT(bias.grad().toList<float>(), ElementsAre(18));
}

TEST(TEST_Function, func_conv2d_03) {
  auto input = Tensor(Array1d<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  input.reshape_({1, 2, 4, 4});
  input.setRequiresGrad(true);

  auto weight = Tensor(Array1d<float>{1, 2, 0, -1, 1, 2, 0, -1});
  weight.reshape_({1, 2, 2, 2});
  weight.setRequiresGrad(true);

  auto bias = Tensor(Array1d<float>{2.f});
  bias.setRequiresGrad(true);

  auto output = function::conv2d(input, weight, bias);
  EXPECT_THAT(output.shape(), ElementsAre(1, 1, 3, 3));
  EXPECT_THAT(output.toList<float>(), ElementsAre(0, 4, 8, 16, 20, 24, 32, 36, 40));

  output.backward(Tensor::onesLike(output));
  EXPECT_THAT(input.grad().toList<float>(), ElementsAre(1, 3, 3, 2, 1, 2, 2, 1, 1, 2, 2, 1, 0, -1, -1, -1, 1, 3, 3, 2,
                                                        1, 2, 2, 1, 1, 2, 2, 1, 0, -1, -1, -1));
  EXPECT_THAT(weight.grad().toList<float>(), ElementsAre(54, 63, 90, 99, 54, 63, 90, 99));
  EXPECT_THAT(bias.grad().toList<float>(), ElementsAre(9));
}

TEST(TEST_Function, func_layerNorm) {
  auto input = Tensor(Array2d<float>{{1.4176, 0.1874, 0.8367}, {-0.1203, 2.5638, -1.2554}});
  auto w = Tensor(Array1d<float>{1.4072, -0.4768, -0.6006});
  auto b = Tensor(Array1d<float>{-0.1609, -0.4865, -0.6256});
  auto y = function::layerNorm(input, {input.shape().back()}, w, b);
  EXPECT_TRUE(y.shape() == input.shape());
  EXPECT_TRUE(VectorNear(y.toList<float>(), {1.5297, 0.1079, -0.6529, -0.6147, -1.1319, -0.0062}));
}
