/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "TinyTorch.h"
#include "test.h"

using namespace tinytorch;

TEST(TEST_Module, linear) {
  auto layer = nn::Linear(4, 4, true);
  layer.weight().fill_(1.2f);
  layer.bias().fill_(0.2f);

  auto input = Tensor(Array2d<float>{{1, 2, 3, 4}, {5, 6, 7, 8}});
  auto output = layer(input);
  auto y = input.sin();
  auto loss_fn = nn::MSELoss(LossReduction::SUM);
  auto loss = loss_fn(output, y);
  loss.backward();

  EXPECT_FLOAT_EQ(loss.item<float>(), 4490.4165f);
  EXPECT_TRUE(
      VectorNear(layer.weight().grad().toList<float>(),
                 {346.306305, 433.741211, 521.176147, 608.611, 339.37558, 425.315826, 511.256042, 597.196289,
                  331.547913, 417.151703, 502.755493, 588.359314, 330.02002, 416.754913, 503.489807, 590.224731}));
  EXPECT_TRUE(VectorNear(layer.bias().grad().toList<float>(), {87.434906, 85.940239, 85.6037903, 86.7348938}));
}

TEST(TEST_Module, flatten) {
  auto layer = nn::Flatten();
  auto input = Tensor(Array2d<float>{{-1, 2, -3, 4}, {5, -6, 7, -8}});
  auto output = layer(input);
  EXPECT_THAT(output.shape(), ElementsAre(8));
}

TEST(TEST_Module, relu) {
  auto layer = nn::Relu();
  auto input = Tensor(Array2d<float>{{-1, 2, -3, 4}, {5, -6, 7, -8}});
  auto output = layer(input);
  EXPECT_THAT(output.shape(), ElementsAre(2, 4));
  EXPECT_THAT(output.toList<float>(), ElementsAre(0, 2, 0, 4, 5, 0, 7, 0));
}

TEST(TEST_Module, dropout) {
  auto layer = nn::Dropout();
  auto input = Tensor(Array2d<float>{{-1, 2, -3, 4}, {5, -6, 7, -8}});

  layer.eval();
  auto output = layer(input);
  EXPECT_THAT(output.shape(), ElementsAre(2, 4));
  EXPECT_EQ(output.toList<float>(), input.toList<float>());

  layer.train();
  output = layer(input);
  EXPECT_THAT(output.shape(), ElementsAre(2, 4));
  EXPECT_TRUE((output == 0).to(DType::Float32).sum().item<float>() > 0);
}

TEST(TEST_Module, rope) {
  auto rope = nn::RoPE(4, 3, 1000.f);
  EXPECT_THAT(rope.cache().shape(), ElementsAre(3, 4));
  EXPECT_TRUE(VectorNear(rope.cache().toList<float>(), {1.0000, 0.0000, 1.0000, 0.0000, 0.5403, 0.8415, 0.9995, 0.0316,
                                                        -0.4161, 0.9093, 0.9980, 0.0632}));

  auto x = Tensor(Array1d<float>{-0.0350, 0.7678, -0.4193, 0.3493, 2.0598, -0.8641, -0.7964, 2.1628,
                                 -0.5256, 1.0712, -1.1556, 0.1891, 0.5730, 0.8307,  0.4564,  -0.2548,
                                 -0.4788, 0.5876, -0.4538, 1.5493, 0.7187, -2.7068, -0.1677, 1.3936});
  x.reshape_({3, 2, 4});
  auto pos = Tensor(Array1d<int64_t>{0, 2, 1});
  auto y = rope(x, pos);
  EXPECT_THAT(y.shape(), ElementsAre(3, 2, 4));
  EXPECT_TRUE(VectorNear(y.toList<float>(), {-0.0350, 0.7678, -0.4193, 0.3493, 2.0598,  -0.8641, -0.7964, 2.1628,
                                             1.2695,  1.0571, 0.0030,  0.2564, -0.6535, 0.8452,  0.3311,  -0.2018,
                                             0.1232,  0.5384, -0.6481, 1.5671, 0.5295,  -2.7495, 0.5142,  1.3073}));
}
