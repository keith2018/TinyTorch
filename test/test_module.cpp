/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include <Torch.h>

#include "test.h"

using namespace TinyTorch;

TEST(TEST_Module, linear) {
  auto layer = nn::Linear(4, 4, true);
  layer.Weights().data().fill(1.2f);
  layer.Bias().data().fill(0.2f);

  auto input = Tensor({{1, 2, 3, 4}, {5, 6, 7, 8}});
  auto output = layer(input);
  auto y = input.sin();
  auto loss_fn = nn::MSELoss(SUM);
  auto loss = loss_fn(output, y);
  loss.backward();

  EXPECT_FLOAT_EQ(loss.item(), 4490.4165f);
  EXPECT_FLOAT_VEC_NEAR(
      layer.Weights().getGrad().data().toArray(),
      {346.306305, 433.741211, 521.176147, 608.611, 339.37558, 425.315826,
       511.256042, 597.196289, 331.547913, 417.151703, 502.755493, 588.359314,
       330.02002, 416.754913, 503.489807, 590.224731});
  EXPECT_FLOAT_VEC_NEAR(layer.Bias().getGrad().data().toArray(),
                        {87.434906, 85.940239, 85.6037903, 86.7348938});
}

TEST(TEST_Module, flatten) {
  auto layer = nn::Flatten();
  auto input = Tensor({{-1, 2, -3, 4}, {5, -6, 7, -8}});
  auto output = layer(input);
  EXPECT_THAT(output.shape(), ElementsAre(8));
}

TEST(TEST_Module, relu) {
  auto layer = nn::Relu();
  auto input = Tensor({{-1, 2, -3, 4}, {5, -6, 7, -8}});
  auto output = layer(input);
  EXPECT_THAT(output.shape(), ElementsAre(2, 4));
  EXPECT_THAT(output.data().toArray(), ElementsAre(0, 2, 0, 4, 5, 0, 7, 0));
}

TEST(TEST_Module, dropout) {
  auto layer = nn::Dropout();
  auto input = Tensor({{-1, 2, -3, 4}, {5, -6, 7, -8}});

  layer.eval();
  auto output = layer(input);
  EXPECT_THAT(output.shape(), ElementsAre(2, 4));
  EXPECT_EQ(output.data().toArray(), input.data().toArray());

  layer.train();
  output = layer(input);
  EXPECT_THAT(output.shape(), ElementsAre(2, 4));
  EXPECT_TRUE((output.data() == 0).sum() > 0);
}
