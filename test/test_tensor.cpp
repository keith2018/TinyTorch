/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "TinyTorch.h"
#include "test.h"

using namespace tinytorch;

TEST(TEST_Tensor, constructor_default) {
  Tensor x;

  EXPECT_FALSE(x.defined());
}

TEST(TEST_Tensor, constructor_shape) {
  Tensor x = Tensor::empty({2, 3});

  EXPECT_TRUE(x.defined());

  EXPECT_TRUE(x.dim() == 2);
  EXPECT_TRUE(x.numel() == 6);
  EXPECT_THAT(x.shape(), ElementsAre(2, 3));
  EXPECT_THAT(x.strides(), ElementsAre(3, 1));
}

TEST(TEST_Tensor, constructor_scalar) {
  Tensor x = Tensor::scalar(2.f);

  EXPECT_TRUE(x.defined());

  EXPECT_TRUE(x.dim() == 0);
  EXPECT_TRUE(x.numel() == 1);
  EXPECT_THAT(x.toList<float>(), ElementsAre(2));
}

TEST(TEST_Tensor, constructor_ones) {
  Tensor x = Tensor::ones({2, 3});

  EXPECT_TRUE(x.dim() == 2);
  EXPECT_TRUE(x.numel() == 6);
  EXPECT_THAT(x.shape(), ElementsAre(2, 3));
  EXPECT_THAT(x.strides(), ElementsAre(3, 1));
  EXPECT_THAT(x.toList<float>(), ElementsAre(1, 1, 1, 1, 1, 1));
}

TEST(TEST_Tensor, constructor_zeros) {
  Tensor x = Tensor::zeros({2, 3});

  EXPECT_TRUE(x.dim() == 2);
  EXPECT_TRUE(x.numel() == 6);
  EXPECT_THAT(x.shape(), ElementsAre(2, 3));
  EXPECT_THAT(x.strides(), ElementsAre(3, 1));
  EXPECT_THAT(x.toList<float>(), ElementsAre(0, 0, 0, 0, 0, 0));
}

TEST(TEST_Tensor, constructor_rand) {
  Tensor x = Tensor::rand({2, 3});

  EXPECT_TRUE(x.dim() == 2);
  EXPECT_TRUE(x.numel() == 6);
  EXPECT_THAT(x.shape(), ElementsAre(2, 3));
  EXPECT_THAT(x.strides(), ElementsAre(3, 1));
}

TEST(TEST_Tensor, constructor_randn) {
  Tensor x = Tensor::randn({2, 3});

  EXPECT_TRUE(x.dim() == 2);
  EXPECT_TRUE(x.numel() == 6);
  EXPECT_THAT(x.shape(), ElementsAre(2, 3));
  EXPECT_THAT(x.strides(), ElementsAre(3, 1));
}

TEST(TEST_Tensor, constructor_bernoulli) {
  Tensor x = Tensor::bernoulli({2, 3}, 0.5);

  EXPECT_TRUE(x.dim() == 2);
  EXPECT_TRUE(x.numel() == 6);
  EXPECT_THAT(x.shape(), ElementsAre(2, 3));
  EXPECT_THAT(x.strides(), ElementsAre(3, 1));
}

TEST(TEST_Tensor, constructor_1d) {
  Tensor x(Array1d<float>{1, 2, 3});

  EXPECT_TRUE(x.dim() == 1);
  EXPECT_TRUE(x.numel() == 3);
  EXPECT_THAT(x.shape(), ElementsAre(3));
  EXPECT_THAT(x.strides(), ElementsAre(1));
  EXPECT_THAT(x.toList<float>(), ElementsAre(1, 2, 3));
}

TEST(TEST_Tensor, constructor_2d) {
  Tensor x(Array2d<float>{{1, 2}, {3, 4}, {5, 6}});

  EXPECT_TRUE(x.dim() == 2);
  EXPECT_TRUE(x.numel() == 6);
  EXPECT_THAT(x.shape(), ElementsAre(3, 2));
  EXPECT_THAT(x.strides(), ElementsAre(2, 1));
  EXPECT_THAT(x.toList<float>(), ElementsAre(1, 2, 3, 4, 5, 6));
}

TEST(TEST_Tensor, constructor_3d) {
  Tensor x(Array3d<float>{{{4, 2, 3}, {1, 0, 3}}, {{4, 2, 3}, {1, 0, 3}}});

  EXPECT_TRUE(x.dim() == 3);
  EXPECT_TRUE(x.numel() == 12);
  EXPECT_THAT(x.shape(), ElementsAre(2, 2, 3));
  EXPECT_THAT(x.strides(), ElementsAre(6, 3, 1));
  EXPECT_THAT(x.toList<float>(), ElementsAre(4, 2, 3, 1, 0, 3, 4, 2, 3, 1, 0, 3));
}
