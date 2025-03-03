/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include <Torch.h>

#include "test.h"

using namespace TinyTorch;

TEST(TEST_Function, func_add) {
  Tensor a({1, 2, 3}, true);
  Tensor b({4, 5, 6}, true);
  auto y = Function::add(a, b);
  EXPECT_THAT(y.data().toList(), ElementsAre(5, 7, 9));

  y.backward(Tensor::onesLike(y));
  EXPECT_THAT(a.getGrad().data().toList(), ElementsAre(1, 1, 1));
  EXPECT_THAT(b.getGrad().data().toList(), ElementsAre(1, 1, 1));
}

TEST(TEST_Function, func_sub) {
  Tensor a({1, 2, 3}, true);
  Tensor b({4, 5, 6}, true);
  auto y = Function::sub(a, b);
  EXPECT_THAT(y.data().toList(), ElementsAre(-3, -3, -3));

  y.backward(Tensor::onesLike(y));
  EXPECT_THAT(a.getGrad().data().toList(), ElementsAre(1, 1, 1));
  EXPECT_THAT(b.getGrad().data().toList(), ElementsAre(-1, -1, -1));
}

TEST(TEST_Function, func_mul) {
  Tensor a({1, 2, 3}, true);
  Tensor b({4, 5, 6}, true);
  auto y = Function::mul(a, b);
  EXPECT_THAT(y.data().toList(), ElementsAre(4, 10, 18));

  y.backward(Tensor::onesLike(y));
  EXPECT_THAT(a.getGrad().data().toList(), ElementsAre(4, 5, 6));
  EXPECT_THAT(b.getGrad().data().toList(), ElementsAre(1, 2, 3));
}

TEST(TEST_Function, func_div) {
  Tensor a({1, 2, 3}, true);
  Tensor b({4, 5, 6}, true);
  auto y = Function::div(a, b);
  EXPECT_THAT(y.data().toList(), ElementsAre(0.25, 0.4, 0.5));

  y.backward(Tensor::onesLike(y));
  EXPECT_THAT(a.getGrad().data().toList(),
              ElementsAre(0.2500, 0.2000, 1.0 / 6));
  EXPECT_THAT(b.getGrad().data().toList(),
              ElementsAre(-0.0625, -0.0800, -1.0 / 12));
}

TEST(TEST_Function, func_sin) {
  Tensor x({0.0f, PI / 2, PI}, true);
  auto y = Function::sin(x);
  EXPECT_THAT(y.data().toList(), ElementsAre(0, 1, std::sin(PI)));

  y.backward(Tensor::onesLike(y));
  EXPECT_THAT(x.getGrad().data().toList(),
              ElementsAre(1, std::cos(PI / 2), -1));
}

TEST(TEST_Function, func_cos) {
  Tensor x({0.0f, PI / 2, PI}, true);
  auto y = Function::cos(x);
  EXPECT_THAT(y.data().toList(),
              ElementsAre(1, std::cos(PI / 2), std::cos(PI)));

  y.backward(Tensor::onesLike(y));
  EXPECT_FLOAT_VEC_NEAR(x.getGrad().data().toList(), {0, -std::sin(PI / 2), 0});
}

TEST(TEST_Function, func_pow) {
  Tensor x({2.0f, 3.0f, 4.0f}, true);
  auto y = Function::pow(x, 3.f);
  EXPECT_THAT(y.data().toList(), ElementsAre(8, 27, 64));

  y.backward(Tensor::onesLike(y));
  EXPECT_THAT(x.getGrad().data().toList(), ElementsAre(12, 27, 48));
}

TEST(TEST_Function, func_sum1) {
  Tensor x({1.0f, 2.0f, 3.0f}, true);
  auto y = Function::sum(x);
  EXPECT_THAT(y.data().toList(), ElementsAre(6));

  y.backward(Tensor::onesLike(y));
  EXPECT_THAT(x.getGrad().data().toList(), ElementsAre(1, 1, 1));
}

TEST(TEST_Function, func_sum2) {
  Tensor x({1.0f, 2.0f, 3.0f}, true);
  auto y = Function::sum(x);
  EXPECT_THAT(y.data().toList(), ElementsAre(6));

  y.backward();
  EXPECT_THAT(x.getGrad().data().toList(), ElementsAre(1, 1, 1));
}

TEST(TEST_Function, func_linear) {
  Tensor x({{-0.3089f, 0.5301f, -0.0245f}, {1.5852f, 0.8954f, 0.7485f}}, true);
  Tensor w({{0.8397f, 1.7990f, -0.2738f}, {-0.8910f, -0.6746f, 0.3419f}}, true);
  Tensor b({-0.9601f, -1.4163f}, true);

  auto y = Function::linear(x, w, b).sum();
  EXPECT_FLOAT_EQ(y.data().item(), -3.1661377);

  y.backward();
  EXPECT_FLOAT_VEC_NEAR(x.getGrad().data().toList(),
                        {-0.0512999892, 1.1244, 0.0681000054, -0.0512999892,
                         1.1244, 0.0681000054});
  EXPECT_FLOAT_VEC_NEAR(w.getGrad().data().toList(),
                        {1.2763, 1.42549992, 0.724, 1.2763, 1.42549992, 0.724});
  EXPECT_THAT(b.getGrad().data().toList(), ElementsAre(2., 2.));
}

TEST(TEST_Function, func_relu) {
  Tensor x({{-1.0, 2.0}, {3.0, -4.0}}, true);
  auto y = Function::relu(x);
  y.sum().backward();
  EXPECT_THAT(y.data().toList(), ElementsAre(0, 2, 3, 0));
  EXPECT_THAT(x.getGrad().data().toList(), ElementsAre(0, 1, 1, 0));
}

TEST(TEST_Function, func_mseLoss_none) {
  Tensor x({{-0.3089f, 0.5301f, -0.0245f}, {1.5852f, 0.8954f, 0.7485f}}, true);
  Tensor y({{0.8397f, 1.7990f, -0.2738f}, {-0.8910f, -0.6746f, 0.3419f}}, true);
  auto loss = Function::mseLoss(x, y, NONE);
  EXPECT_FLOAT_VEC_NEAR(loss.data().toList(),
                        {1.31928194, 1.6101073, 0.0621504858, 6.13156557,
                         2.46489978, 0.165323555});
  loss.backward(Tensor::onesLike(loss));
  EXPECT_FLOAT_VEC_NEAR(
      x.getGrad().data().toList(),
      {-2.2972, -2.5378, 0.498599976, 4.95239973, 3.13999987, 0.8132});
  EXPECT_FLOAT_VEC_NEAR(
      y.getGrad().data().toList(),
      {2.2972, 2.5378, -0.498599976, -4.95239973, -3.13999987, -0.8132});
}

TEST(TEST_Function, func_mseLoss_mean) {
  Tensor x({{-0.3089f, 0.5301f, -0.0245f}, {1.5852f, 0.8954f, 0.7485f}}, true);
  Tensor y({{0.8397f, 1.7990f, -0.2738f}, {-0.8910f, -0.6746f, 0.3419f}}, true);
  auto loss = Function::mseLoss(x, y, MEAN);
  EXPECT_FLOAT_EQ(loss.data().item(), 1.95888805);
  loss.backward();
  EXPECT_FLOAT_VEC_NEAR(x.getGrad().data().toList(),
                        {-0.382866651, -0.422966689, 0.0831, 0.825399935,
                         0.523333311, 0.135533333});
  EXPECT_FLOAT_VEC_NEAR(y.getGrad().data().toList(),
                        {0.382866651, 0.422966689, -0.0831, -0.825399935,
                         -0.523333311, -0.135533333});
}

TEST(TEST_Function, func_mseLoss_sum) {
  Tensor x({{-0.3089f, 0.5301f, -0.0245f}, {1.5852f, 0.8954f, 0.7485f}}, true);
  Tensor y({{0.8397f, 1.7990f, -0.2738f}, {-0.8910f, -0.6746f, 0.3419f}}, true);
  auto loss = Function::mseLoss(x, y, SUM);
  EXPECT_FLOAT_EQ(loss.data().item(), 11.7533283);
  loss.backward();
  EXPECT_FLOAT_VEC_NEAR(
      x.getGrad().data().toList(),
      {-2.2972, -2.5378, 0.498599976, 4.95239973, 3.13999987, 0.8132});
  EXPECT_FLOAT_VEC_NEAR(
      y.getGrad().data().toList(),
      {2.2972, 2.5378, -0.498599976, -4.95239973, -3.13999987, -0.8132});
}

TEST(TEST_Function, func_dropout) {
  auto input = Tensor::randn({5, 10}, true);
  auto p = 0.3f;
  auto output = Function::dropout(input, p, true);
  EXPECT_EQ(output.shape(), input.shape());
  auto zeroCnt = (output.data() == 0).sum().item();
  auto expectZeroCnt = p * (float)input.numel();
  EXPECT_TRUE(std::abs(zeroCnt - expectZeroCnt) < expectZeroCnt * 0.7);

  output = Function::dropout(input, p, false);
  EXPECT_EQ(output.data().toList(), input.data().toList());

  output = Function::dropout(input, p, true);
  output.sum().backward();
  auto mask = output.data() != 0;
  auto grad = mask / (1.f - p);
  EXPECT_EQ(input.getGrad().data().toList(), grad.toList());
}

TEST(TEST_Function, func_softmax) {
  auto input = Tensor({1.1, 1.2, 1.3, 1.6}, true);
  auto output = Function::softmax(input, 0);
  EXPECT_FLOAT_VEC_NEAR(output.data().toList(),
                        {0.2010, 0.2221, 0.2455, 0.3314});

  output.backward(input);
  EXPECT_FLOAT_VEC_NEAR(input.getGrad().data().toList(),
                        {-0.0476, -0.0304, -0.0091, 0.0872});
}

TEST(TEST_Function, func_log_softmax) {
  auto input = Tensor({1.1, 1.2, 1.3, 1.6}, true);
  auto output = Function::logSoftmax(input, 0);
  EXPECT_FLOAT_VEC_NEAR(output.data().toList(),
                        {-1.6045, -1.5045, -1.4045, -1.1045});
  output.backward(input);
  EXPECT_FLOAT_VEC_NEAR(input.getGrad().data().toList(),
                        {0.0548, 0.0449, 0.0234, -0.1232});

  input = Tensor({{1, 2}, {3, 4}}, true);
  output = Function::logSoftmax(input, 0);
  EXPECT_FLOAT_VEC_NEAR(output.data().toList(),
                        {-2.1269, -2.1269, -0.1269, -0.1269});
  output.backward(Tensor::onesLike(output));
  EXPECT_FLOAT_VEC_NEAR(input.getGrad().data().toList(),
                        {0.7616, 0.7616, -0.7616, -0.7616});

  input = Tensor({{1, 2}, {3, 4}}, true);
  output = Function::logSoftmax(input, 1);
  EXPECT_FLOAT_VEC_NEAR(output.data().toList(),
                        {-1.3133, -0.3133, -1.3133, -0.3133});
  output.backward(Tensor::onesLike(output));
  EXPECT_FLOAT_VEC_NEAR(input.getGrad().data().toList(),
                        {0.4621, -0.4621, 0.4621, -0.4621});
}

TEST(TEST_Function, func_nllloss) {
  auto input = Tensor({{0.1, 0.2, 0.7}, {0.3, 0.4, 0.3}}, true);
  auto target = Tensor({2, 1});
  auto loss = Function::nllloss(input, target, NONE);
  EXPECT_FLOAT_VEC_NEAR(loss.data().toList(), {-0.7, -0.4});
  loss.backward(Tensor::onesLike(loss));
  EXPECT_FLOAT_VEC_NEAR(input.getGrad().data().toList(), {0, 0, -1, 0, -1, 0});

  input = Tensor({{0.1, 0.2, 0.7}, {0.3, 0.4, 0.3}}, true);
  target = Tensor({2, 1});
  loss = Function::nllloss(input, target, MEAN);
  EXPECT_FLOAT_NEAR(loss.item(), -0.55);
  loss.backward(Tensor::onesLike(loss));
  EXPECT_FLOAT_VEC_NEAR(input.getGrad().data().toList(),
                        {0, 0, -0.5, 0, -0.5, 0});

  input = Tensor({{0.1, 0.2, 0.7}, {0.3, 0.4, 0.3}}, true);
  target = Tensor({2, 1});
  loss = Function::nllloss(input, target, SUM);
  EXPECT_FLOAT_NEAR(loss.item(), -1.1);
  loss.backward(Tensor::onesLike(loss));
  EXPECT_FLOAT_VEC_NEAR(input.getGrad().data().toList(), {0, 0, -1, 0, -1, 0});
}

TEST(TEST_Function, func_maxpool2d_01) {
  auto input =
      Tensor({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}});
  input = input.reshape({1, 1, 4, 4});
  input.setRequiresGrad(true);

  auto output = Function::maxPool2d(input, 2, 2);
  EXPECT_THAT(output.shape(), ElementsAre(1, 1, 2, 2));
  EXPECT_THAT(output.data().toList(), ElementsAre(6, 8, 14, 16));

  output = Function::maxPool2d(input, 3, 1);
  EXPECT_THAT(output.shape(), ElementsAre(1, 1, 2, 2));
  EXPECT_THAT(output.data().toList(), ElementsAre(11, 12, 15, 16));

  output = Function::maxPool2d(input, 3, 2);
  EXPECT_THAT(output.shape(), ElementsAre(1, 1, 1, 1));
  EXPECT_THAT(output.data().toList(), ElementsAre(11));

  output = Function::maxPool2d(input, 3, 2, 1);
  EXPECT_THAT(output.shape(), ElementsAre(1, 1, 2, 2));
  EXPECT_THAT(output.data().toList(), ElementsAre(6, 8, 14, 16));

  output.backward(Tensor::onesLike(output));
  EXPECT_THAT(input.getGrad().data().toList(),
              ElementsAre(0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1));
}

TEST(TEST_Function, func_maxpool2d_02) {
  auto input =
      Tensor({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}});
  input = input.reshape({1, 1, 4, 4});
  input.setRequiresGrad(true);

  auto output = Function::maxPool2d(input, 2, 2, 1);
  EXPECT_THAT(output.shape(), ElementsAre(1, 1, 3, 3));
  EXPECT_THAT(output.data().toList(),
              ElementsAre(1, 3, 4, 9, 11, 12, 13, 15, 16));

  output.backward(Tensor::onesLike(output));
  EXPECT_THAT(input.getGrad().data().toList(),
              ElementsAre(1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1));
}

TEST(TEST_Function, func_conv2d_01) {
  auto input =
      Tensor({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}});
  input = input.reshape({1, 1, 4, 4});
  input.setRequiresGrad(true);

  auto weight = Tensor({{1, 0}, {0, -1}});
  weight = weight.reshape({1, 1, 2, 2});
  weight.setRequiresGrad(true);

  auto bias = Tensor(Array1d{2.f}, true);

  auto output = Function::conv2d(input, weight, bias);
  EXPECT_THAT(output.shape(), ElementsAre(1, 1, 3, 3));
  EXPECT_THAT(output.data().toList(),
              ElementsAre(-3, -3, -3, -3, -3, -3, -3, -3, -3));

  output.backward(Tensor::onesLike(output));
  EXPECT_THAT(input.getGrad().data().toList(),
              ElementsAre(1, 1, 1, 0, 1, 0, 0, -1, 1, 0, 0, -1, 0, -1, -1, -1));
  EXPECT_THAT(weight.getGrad().data().toList(), ElementsAre(54, 63, 90, 99));
  EXPECT_THAT(bias.getGrad().data().toList(), ElementsAre(9));
}

TEST(TEST_Function, func_conv2d_02) {
  auto input = Tensor({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                       1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  input = input.reshape({2, 1, 4, 4});
  input.setRequiresGrad(true);

  auto weight = Tensor({{1, 2}, {0, -1}});
  weight = weight.reshape({1, 1, 2, 2});
  weight.setRequiresGrad(true);

  auto bias = Tensor(Array1d{2.f}, true);

  auto output = Function::conv2d(input, weight, bias);
  EXPECT_THAT(output.shape(), ElementsAre(2, 1, 3, 3));
  EXPECT_THAT(output.data().toList(),
              ElementsAre(1, 3, 5, 9, 11, 13, 17, 19, 21, 1, 3, 5, 9, 11, 13,
                          17, 19, 21));

  output.backward(Tensor::onesLike(output));
  EXPECT_THAT(input.getGrad().data().toList(),
              ElementsAre(1, 3, 3, 2, 1, 2, 2, 1, 1, 2, 2, 1, 0, -1, -1, -1, 1,
                          3, 3, 2, 1, 2, 2, 1, 1, 2, 2, 1, 0, -1, -1, -1));
  EXPECT_THAT(weight.getGrad().data().toList(),
              ElementsAre(108, 126, 180, 198));
  EXPECT_THAT(bias.getGrad().data().toList(), ElementsAre(18));
}

TEST(TEST_Function, func_conv2d_03) {
  auto input = Tensor({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                       1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  input = input.reshape({1, 2, 4, 4});
  input.setRequiresGrad(true);

  auto weight = Tensor({1, 2, 0, -1, 1, 2, 0, -1});
  weight = weight.reshape({1, 2, 2, 2});
  weight.setRequiresGrad(true);

  auto bias = Tensor(Array1d{2.f}, true);

  auto output = Function::conv2d(input, weight, bias);
  EXPECT_THAT(output.shape(), ElementsAre(1, 1, 3, 3));
  EXPECT_THAT(output.data().toList(),
              ElementsAre(0, 4, 8, 16, 20, 24, 32, 36, 40));

  output.backward(Tensor::onesLike(output));
  EXPECT_THAT(input.getGrad().data().toList(),
              ElementsAre(1, 3, 3, 2, 1, 2, 2, 1, 1, 2, 2, 1, 0, -1, -1, -1, 1,
                          3, 3, 2, 1, 2, 2, 1, 1, 2, 2, 1, 0, -1, -1, -1));
  EXPECT_THAT(weight.getGrad().data().toList(),
              ElementsAre(54, 63, 90, 99, 54, 63, 90, 99));
  EXPECT_THAT(bias.getGrad().data().toList(), ElementsAre(9));
}

TEST(TEST_Function, func_batchNorm_2d) {
  auto input = Tensor::arange(1.f, 24.5f, 1.f);
  input = input.reshape({2, 3, 2, 2});
  input.setRequiresGrad(true);
  auto runningMean = Tensor::zeros({3});
  auto runningVar = Tensor::ones({3});
  auto weight = Tensor::ones({3}, true);
  auto bias = Tensor::zeros({3}, true);
  auto output = Function::batchNorm(input, runningMean, runningVar, weight,
                                    bias, true, 0.2);
  EXPECT_THAT(output.shape(), ElementsAre(2, 3, 2, 2));
  EXPECT_FLOAT_VEC_NEAR(
      output.data().toList(),
      {-1.2288, -1.0650, -0.9012, -0.7373, -1.2288, -1.0650, -0.9012, -0.7373,
       -1.2288, -1.0650, -0.9012, -0.7373, 0.7373,  0.9012,  1.0650,  1.2288,
       0.7373,  0.9012,  1.0650,  1.2288,  0.7373,  0.9012,  1.0650,  1.2288});

  EXPECT_FLOAT_VEC_NEAR(runningMean.data().toList(), {1.7000, 2.5000, 3.3000});
  EXPECT_FLOAT_VEC_NEAR(runningVar.data().toList(), {9.3143, 9.3143, 9.3143});

  output.backward(Tensor::onesLike(output));
  EXPECT_FLOAT_VEC_NEAR(input.getGrad().data().toList(),
                        TensorImpl::zeros({input.numel()}).toList());
  EXPECT_FLOAT_VEC_NEAR(weight.getGrad().data().toList(), {0., 0., 0.});
  EXPECT_FLOAT_VEC_NEAR(bias.getGrad().data().toList(), {8., 8., 8.});
}
