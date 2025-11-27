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
