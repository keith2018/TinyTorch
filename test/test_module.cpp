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
  EXPECT_THAT(rope.cos().shape(), ElementsAre(3, 4));
  EXPECT_THAT(rope.sin().shape(), ElementsAre(3, 4));
  EXPECT_TRUE(VectorNear(rope.cos().toList<float>(), {1.0000, 1.0000, 1.0000, 1.0000, 0.5403, 0.9995, 0.5403, 0.9995,
                                                      -0.4161, 0.9980, -0.4161, 0.9980}));
  EXPECT_TRUE(VectorNear(rope.sin().toList<float>(), {0.0000, 0.0000, 0.0000, 0.0000, 0.8415, 0.0316, 0.8415, 0.0316,
                                                      0.9093, 0.0632, 0.9093, 0.0632}));

  RopeScalingConfig scaling{2.f, 4.f, 1.f, 2};
  rope = nn::RoPE(4, 3, 10000.f, scaling);
  EXPECT_TRUE(VectorNear(rope.cos().toList<float>(), {1.0000, 1.0000, 1.0000, 1.0000, 0.8776, 1.0000, 0.8776, 1.0000,
                                                      0.5403, 0.9999, 0.5403, 0.9999}));
  EXPECT_TRUE(VectorNear(rope.sin().toList<float>(), {0.0000, 0.0000, 0.0000, 0.0000, 0.4794, 0.0050, 0.4794, 0.0050,
                                                      0.8415, 0.0100, 0.8415, 0.0100}));

  auto x = Tensor(Array1d<float>{
      0.0516,  -0.9695, -0.0861, 1.3223,  0.5351,  0.1768,  -0.0966, 0.2490,  -0.8414, -0.5110, -1.1106, -0.1058,
      0.2775,  0.4491,  -0.5324, 0.1249,  1.0637,  -1.3959, 0.4438,  2.3408,  -0.0822, 1.0439,  -0.7985, 0.1040,
      0.7335,  -2.2184, -0.6714, -0.3401, 1.5914,  -0.4572, -0.7603, -0.6322, 1.0201,  -0.5109, 0.8172,  -0.5532,
      -0.1150, 0.2467,  -0.3851, 0.5726,  -0.1837, -1.5242, 0.4929,  -0.2529, 1.2423,  -0.4946, -0.4185, 1.1320});
  x.reshape_({2, 2, 3, 4});
  auto y = rope(x);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2, 3, 4));
  EXPECT_TRUE(VectorNear(
      y.toList<float>(),
      {0.0516,  -0.9695, -0.0861, 1.3223,  0.5159,  0.1756,  0.1718, 0.2499,  0.4799,  -0.5099, -1.3081, -0.1109,
       0.2775,  0.4491,  -0.5324, 0.1249,  0.7207,  -1.4076, 0.8994, 2.3338,  0.6274,  1.0428,  -0.5006, 0.1145,
       0.7335,  -2.2184, -0.6714, -0.3401, 1.7611,  -0.4540, 0.0957, -0.6344, -0.1365, -0.5053, 1.3000,  -0.5583,
       -0.1150, 0.2467,  -0.3851, 0.5726,  -0.3975, -1.5229, 0.3445, -0.2605, 1.0234,  -0.5059, 0.8192,  1.1270}));
}
