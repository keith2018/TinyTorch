/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Torch.h"
#include "test.h"

using namespace TinyTorch;

TEST(TEST_TensorImpl, constructor_default) {
  TensorImpl x;

  EXPECT_TRUE(x.empty());
  EXPECT_FALSE(x.isScalar());
  EXPECT_TRUE(x.dim() == 0);
}

TEST(TEST_TensorImpl, constructor_shape) {
  TensorImpl x = TensorImpl::shape({2, 3});

  EXPECT_FALSE(x.empty());
  EXPECT_FALSE(x.isScalar());

  EXPECT_TRUE(x.dim() == 2);
  EXPECT_TRUE(x.size() == 6);
  EXPECT_THAT(x.shape(), ElementsAre(2, 3));
  EXPECT_THAT(x.strides(), ElementsAre(3, 1));
}

TEST(TEST_TensorImpl, constructor_scalar) {
  TensorImpl x = TensorImpl::scalar(2);

  EXPECT_FALSE(x.empty());
  EXPECT_TRUE(x.isScalar());

  EXPECT_TRUE(x.dim() == 0);
  EXPECT_TRUE(x.size() == 1);
  EXPECT_THAT(x.toArray(), ElementsAre(2));
}

TEST(TEST_TensorImpl, constructor_ones) {
  TensorImpl x = TensorImpl::ones({2, 3});

  EXPECT_TRUE(x.dim() == 2);
  EXPECT_TRUE(x.size() == 6);
  EXPECT_THAT(x.shape(), ElementsAre(2, 3));
  EXPECT_THAT(x.strides(), ElementsAre(3, 1));
  EXPECT_THAT(x.toArray(), ElementsAre(1, 1, 1, 1, 1, 1));
}

TEST(TEST_TensorImpl, constructor_zeros) {
  TensorImpl x = TensorImpl::zeros({2, 3});

  EXPECT_TRUE(x.dim() == 2);
  EXPECT_TRUE(x.size() == 6);
  EXPECT_THAT(x.shape(), ElementsAre(2, 3));
  EXPECT_THAT(x.strides(), ElementsAre(3, 1));
  EXPECT_THAT(x.toArray(), ElementsAre(0, 0, 0, 0, 0, 0));
}

TEST(TEST_TensorImpl, constructor_rand) {
  TensorImpl x = TensorImpl::rand({2, 3});

  EXPECT_TRUE(x.dim() == 2);
  EXPECT_TRUE(x.size() == 6);
  EXPECT_THAT(x.shape(), ElementsAre(2, 3));
  EXPECT_THAT(x.strides(), ElementsAre(3, 1));
}

TEST(TEST_TensorImpl, constructor_randn) {
  TensorImpl x = TensorImpl::randn({2, 3});

  EXPECT_TRUE(x.dim() == 2);
  EXPECT_TRUE(x.size() == 6);
  EXPECT_THAT(x.shape(), ElementsAre(2, 3));
  EXPECT_THAT(x.strides(), ElementsAre(3, 1));
}

TEST(TEST_TensorImpl, constructor_bernoulli) {
  TensorImpl x = TensorImpl::bernoulli({2, 3}, 0.5);

  EXPECT_TRUE(x.dim() == 2);
  EXPECT_TRUE(x.size() == 6);
  EXPECT_THAT(x.shape(), ElementsAre(2, 3));
  EXPECT_THAT(x.strides(), ElementsAre(3, 1));
}

TEST(TEST_TensorImpl, constructor_tri) {
  TensorImpl x = TensorImpl::tri(3);
  EXPECT_THAT(x.shape(), ElementsAre(3, 3));
  EXPECT_THAT(x.toArray(), ElementsAre(1, 0, 0, 1, 1, 0, 1, 1, 1));

  x = TensorImpl::tri(2, 3);
  EXPECT_THAT(x.shape(), ElementsAre(2, 3));
  EXPECT_THAT(x.toArray(), ElementsAre(1, 0, 0, 1, 1, 0));

  x = TensorImpl::tri(3, 3, 1);
  EXPECT_THAT(x.shape(), ElementsAre(3, 3));
  EXPECT_THAT(x.toArray(), ElementsAre(1, 1, 0, 1, 1, 1, 1, 1, 1));

  x = TensorImpl::tri(3, 3, -1);
  EXPECT_THAT(x.shape(), ElementsAre(3, 3));
  EXPECT_THAT(x.toArray(), ElementsAre(0, 0, 0, 1, 0, 0, 1, 1, 0));
}

TEST(TEST_TensorImpl, constructor_1d) {
  TensorImpl x({1, 2, 3});

  EXPECT_TRUE(x.dim() == 1);
  EXPECT_TRUE(x.size() == 3);
  EXPECT_THAT(x.shape(), ElementsAre(3));
  EXPECT_THAT(x.strides(), ElementsAre(1));
  EXPECT_THAT(x.toArray(), ElementsAre(1, 2, 3));
}

TEST(TEST_TensorImpl, constructor_2d) {
  TensorImpl x({{1, 2}, {3, 4}, {5, 6}});

  EXPECT_TRUE(x.dim() == 2);
  EXPECT_TRUE(x.size() == 6);
  EXPECT_THAT(x.shape(), ElementsAre(3, 2));
  EXPECT_THAT(x.strides(), ElementsAre(2, 1));
  EXPECT_THAT(x.toArray(), ElementsAre(1, 2, 3, 4, 5, 6));
}

TEST(TEST_TensorImpl, constructor_3d) {
  TensorImpl x({{{4, 2, 3}, {1, 0, 3}}, {{4, 2, 3}, {1, 0, 3}}});

  EXPECT_TRUE(x.dim() == 3);
  EXPECT_TRUE(x.size() == 12);
  EXPECT_THAT(x.shape(), ElementsAre(2, 2, 3));
  EXPECT_THAT(x.strides(), ElementsAre(6, 3, 1));
  EXPECT_THAT(x.toArray(), ElementsAre(4, 2, 3, 1, 0, 3, 4, 2, 3, 1, 0, 3));
}

TEST(TEST_TensorImpl, basic_reshape) {
  TensorImpl x({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
  auto y = x.reshape({2, 4});
  EXPECT_THAT(y.shape(), ElementsAre(2, 4));

  y = x.reshape({2, -1});
  EXPECT_THAT(y.shape(), ElementsAre(2, 4));

  y = x.reshape({-1, 2});
  EXPECT_THAT(y.shape(), ElementsAre(4, 2));
}

TEST(TEST_TensorImpl, basic_flatten) {
  TensorImpl x({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
  auto y = TensorImpl::flatten(x);
  EXPECT_THAT(y.shape(), ElementsAre(8));

  y = TensorImpl::flatten(x, 1);
  EXPECT_THAT(y.shape(), ElementsAre(2, 4));

  y = TensorImpl::flatten(x, 1, 2);
  EXPECT_THAT(y.shape(), ElementsAre(2, 4));
}

TEST(TEST_TensorImpl, basic_unflatten) {
  auto y = TensorImpl::unflatten(TensorImpl::randn({3, 4, 1}), 1, {2, 2});
  EXPECT_THAT(y.shape(), ElementsAre(3, 2, 2, 1));

  y = TensorImpl::unflatten(TensorImpl::randn({3, 4, 1}), 1, {-1, 2});
  EXPECT_THAT(y.shape(), ElementsAre(3, 2, 2, 1));

  y = TensorImpl::unflatten(TensorImpl::randn({5, 12, 3}), -2, {2, 2, 3, 1, 1});
  EXPECT_THAT(y.shape(), ElementsAre(5, 2, 2, 3, 1, 1, 3));
}

TEST(TEST_TensorImpl, basic_squeeze) {
  auto x = TensorImpl::zeros({2, 1, 2, 1, 2});
  auto y = TensorImpl::squeeze(x);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2, 2));
  y = TensorImpl::squeeze(x, 0);
  EXPECT_THAT(y.shape(), ElementsAre(2, 1, 2, 1, 2));
  y = TensorImpl::squeeze(x, 1);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2, 1, 2));
  y = TensorImpl::squeeze(x, {1, 2, 3});
  EXPECT_THAT(y.shape(), ElementsAre(2, 2, 2));
}

TEST(TEST_TensorImpl, basic_unsqueeze) {
  auto x = TensorImpl::zeros({2, 1, 2});
  auto y = TensorImpl::unsqueeze(x, 0);
  EXPECT_THAT(y.shape(), ElementsAre(1, 2, 1, 2));
  y = TensorImpl::unsqueeze(x, 1);
  EXPECT_THAT(y.shape(), ElementsAre(2, 1, 1, 2));
  y = TensorImpl::unsqueeze(x, -1);
  EXPECT_THAT(y.shape(), ElementsAre(2, 1, 2, 1));
}

TEST(TEST_TensorImpl, basic_fill) {
  auto x = TensorImpl::randn({2, 3});

  x.fill(-1);
  EXPECT_THAT(x.toArray(), ElementsAre(-1, -1, -1, -1, -1, -1));

  x.fill(2);
  EXPECT_THAT(x.toArray(), ElementsAre(2, 2, 2, 2, 2, 2));
}

TEST(TEST_TensorImpl, basic_clamp) {
  TensorImpl x;

  x = TensorImpl({1, 2, 3});
  x.clampMin(2.3f);
  EXPECT_THAT(x.toArray(), ElementsAre(2.3, 2.3, 3));

  x = TensorImpl({1, 2, 3});
  x.clampMax(2.2f);
  EXPECT_THAT(x.toArray(), ElementsAre(1, 2, 2.2));

  x = TensorImpl({1, 2, 3});
  x.clamp(1.2f, 2.2f);
  EXPECT_THAT(x.toArray(), ElementsAre(1.2, 2, 2.2));
}

TEST(TEST_TensorImpl, basic_range) {
  auto range = TensorImpl::range(3, 6);
  EXPECT_THAT(range, ElementsAre(3, 4, 5));

  range = TensorImpl::range(3, 10, 2);
  EXPECT_THAT(range, ElementsAre(3, 5, 7, 9));

  auto t = TensorImpl::arange(3, 10, 2);
  EXPECT_THAT(t.shape(), ElementsAre(4));
  EXPECT_THAT(t.toArray(), ElementsAre(3, 5, 7, 9));

  t = TensorImpl::linspace(3, 10, 5);
  EXPECT_THAT(t.shape(), ElementsAre(5));
  EXPECT_THAT(t.toArray(), ElementsAre(3, 4.75, 6.5, 8.25, 10));
}

TEST(TEST_TensorImpl, basic_indexing) {
  TensorImpl x({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});

  auto y = x.index({-1, 0});
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toArray(), ElementsAre(7, 8, 9, 1, 2, 3));

  y = x.index(std::vector<int32_t>{1});
  EXPECT_THAT(y.shape(), ElementsAre(1, 3));
  EXPECT_THAT(y.toArray(), ElementsAre(4, 5, 6));

  y = x.index(0);
  EXPECT_THAT(y.shape(), ElementsAre(3));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 2, 3));

  y = x.index(1);
  EXPECT_THAT(y.shape(), ElementsAre(3));
  EXPECT_THAT(y.toArray(), ElementsAre(4, 5, 6));

  y = x.index(0, 0);
  EXPECT_TRUE(y.isScalar());
  EXPECT_THAT(y.toArray(), ElementsAre(1));

  y = x.index(1, 1);
  EXPECT_TRUE(y.isScalar());
  EXPECT_THAT(y.toArray(), ElementsAre(5));

  y = x.indexAdvance({{0, 1}});
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 2, 3, 4, 5, 6));

  y = x.indexAdvance({{0, 1}, {2, 1}});
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toArray(), ElementsAre(3, 5));

  y = x.indexAdvance({{-1, 1}, {2, -1}});
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toArray(), ElementsAre(9, 6));

  x.indexAdvanceSet({{-1, 1}, {2, -1}}, -1);
  EXPECT_THAT(x.toArray(), ElementsAre(1, 2, 3, 4, 5, -1, 7, 8, -1));

  x.indexAdvanceSet({{-1, 1}, {2, -1}}, TensorImpl({1.2, 2.3}));
  EXPECT_THAT(x.toArray(), ElementsAre(1, 2, 3, 4, 5, 2.3, 7, 8, 1.2));
}

TEST(TEST_TensorImpl, basic_transpose) {
  TensorImpl x({1, 2, 3});
  auto y = x.transpose();
  EXPECT_TRUE(y.shape() == x.shape());
  EXPECT_TRUE(y.toArray() == x.toArray());

  x = TensorImpl({{1, 2}, {3, 4}, {5, 6}});
  y = x.transpose();
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 3, 5, 2, 4, 6));

  x = TensorImpl(Array3d{{{1, 2, 3}, {4, 5, 6}}});
  y = x.transpose();
  EXPECT_THAT(y.shape(), ElementsAre(3, 2, 1));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 4, 2, 5, 3, 6));

  y = x.transpose({1, 0, 2});
  EXPECT_THAT(y.shape(), ElementsAre(2, 1, 3));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 2, 3, 4, 5, 6));

  x = TensorImpl::arange(0, 8);
  x.reshape({1, 2, 2, 2});
  y = x.transpose({0, 3, 1, 2});
  EXPECT_THAT(y.shape(), ElementsAre(1, 2, 2, 2));
  EXPECT_THAT(y.toArray(), ElementsAre(0, 2, 4, 6, 1, 3, 5, 7));

  y = x.transpose();
  EXPECT_THAT(y.shape(), ElementsAre(2, 2, 2, 1));
  EXPECT_THAT(y.toArray(), ElementsAre(0, 4, 2, 6, 1, 5, 3, 7));
}

TEST(TEST_TensorImpl, basic_split) {
  TensorImpl x({{{4, 2, 3}, {1, 0, 3}}, {{4, 2, 3}, {1, 0, 3}}});
  auto y = x.split(2, 0);
  EXPECT_TRUE(y.size() == 2);
  EXPECT_THAT(y[0].shape(), ElementsAre(1, 2, 3));
  EXPECT_THAT(y[0].toArray(), ElementsAre(4, 2, 3, 1, 0, 3));
  EXPECT_THAT(y[1].shape(), ElementsAre(1, 2, 3));
  EXPECT_THAT(y[1].toArray(), ElementsAre(4, 2, 3, 1, 0, 3));

  y = x.split(2, 1);
  EXPECT_TRUE(y.size() == 2);
  EXPECT_THAT(y[0].shape(), ElementsAre(2, 1, 3));
  EXPECT_THAT(y[0].toArray(), ElementsAre(4, 2, 3, 4, 2, 3));
  EXPECT_THAT(y[1].shape(), ElementsAre(2, 1, 3));
  EXPECT_THAT(y[1].toArray(), ElementsAre(1, 0, 3, 1, 0, 3));

  y = x.split(3, 2);
  EXPECT_TRUE(y.size() == 3);
  EXPECT_THAT(y[0].shape(), ElementsAre(2, 2, 1));
  EXPECT_THAT(y[0].toArray(), ElementsAre(4, 1, 4, 1));
  EXPECT_THAT(y[1].shape(), ElementsAre(2, 2, 1));
  EXPECT_THAT(y[1].toArray(), ElementsAre(2, 0, 2, 0));
  EXPECT_THAT(y[2].shape(), ElementsAre(2, 2, 1));
  EXPECT_THAT(y[2].toArray(), ElementsAre(3, 3, 3, 3));

  y = x.split(std::vector<int32_t>({1}), 2);
  EXPECT_TRUE(y.size() == 2);
  EXPECT_THAT(y[0].shape(), ElementsAre(2, 2, 1));
  EXPECT_THAT(y[0].toArray(), ElementsAre(4, 1, 4, 1));
  EXPECT_THAT(y[1].shape(), ElementsAre(2, 2, 2));
  EXPECT_THAT(y[1].toArray(), ElementsAre(2, 3, 0, 3, 2, 3, 0, 3));
}

TEST(TEST_TensorImpl, basic_concatenate) {
  TensorImpl a(Array2d({{1, 2}, {3, 4}}));
  TensorImpl b(Array2d({{5, 6}}));
  TensorImpl bT = b.transpose();
  auto y = TensorImpl::concatenate({a, b}, 0);
  EXPECT_THAT(y.shape(), ElementsAre(3, 2));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 2, 3, 4, 5, 6));

  y = TensorImpl::concatenate({a, bT}, 1);
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 2, 5, 3, 4, 6));

  y = TensorImpl::concatenate({a, b});
  EXPECT_THAT(y.shape(), ElementsAre(6));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 2, 3, 4, 5, 6));
}

TEST(TEST_TensorImpl, basic_stack) {
  TensorImpl a({1, 2, 3});
  TensorImpl b({4, 5, 6});
  auto y = TensorImpl::stack({a, b});
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 2, 3, 4, 5, 6));

  y = TensorImpl::stack({a, b}, -1);
  EXPECT_THAT(y.shape(), ElementsAre(3, 2));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 4, 2, 5, 3, 6));

  y = TensorImpl::stack({a, b}, 1);
  EXPECT_THAT(y.shape(), ElementsAre(3, 2));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 4, 2, 5, 3, 6));
}

TEST(TEST_TensorImpl, basic_vstack) {
  TensorImpl a({1, 2, 3});
  TensorImpl b({4, 5, 6});
  auto y = TensorImpl::vstack({a, b});
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 2, 3, 4, 5, 6));

  a = TensorImpl(Array2d({{1}, {2}, {3}}));
  b = TensorImpl(Array2d({{4}, {5}, {6}}));
  y = TensorImpl::vstack({a, b});
  EXPECT_THAT(y.shape(), ElementsAre(6, 1));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 2, 3, 4, 5, 6));
}

TEST(TEST_TensorImpl, basic_hstack) {
  TensorImpl a({1, 2, 3});
  TensorImpl b({4, 5, 6});
  auto y = TensorImpl::hstack({a, b});
  EXPECT_THAT(y.shape(), ElementsAre(6));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 2, 3, 4, 5, 6));

  a = TensorImpl(Array2d({{1}, {2}, {3}}));
  b = TensorImpl(Array2d({{4}, {5}, {6}}));
  y = TensorImpl::hstack({a, b});
  EXPECT_THAT(y.shape(), ElementsAre(3, 2));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 4, 2, 5, 3, 6));
}

TEST(TEST_TensorImpl, basic_dstack) {
  TensorImpl a({1, 2, 3});
  TensorImpl b({2, 3, 4});
  auto y = TensorImpl::dstack({a, b});
  EXPECT_THAT(y.shape(), ElementsAre(1, 3, 2));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 2, 2, 3, 3, 4));

  a = TensorImpl(Array2d({{1}, {2}, {3}}));
  b = TensorImpl(Array2d({{2}, {3}, {4}}));
  y = TensorImpl::dstack({a, b});
  EXPECT_THAT(y.shape(), ElementsAre(3, 1, 2));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 2, 2, 3, 3, 4));
}

TEST(TEST_TensorImpl, math_compare) {
  TensorImpl x({{1, 2, 3}, {3, 4, 5}});
  TensorImpl x2({{3, 4, 5}, {1, 2, 3}});
  auto y = x > 2;
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toArray(), ElementsAre(0, 0, 1, 1, 1, 1));

  y = x < 3;
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 1, 0, 0, 0, 0));

  y = x < x2;
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 1, 1, 0, 0, 0));

  y = x == TensorImpl({{1, 1, 3}, {1, 4, 5}});
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 0, 1, 0, 1, 1));

  y = x != TensorImpl({{1, 1, 3}, {1, 4, 5}});
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toArray(), ElementsAre(0, 1, 0, 1, 0, 0));
}

TEST(TEST_TensorImpl, math_scalar) {
  TensorImpl x({{1, 2}, {3, 4}});

  // add
  TensorImpl y = 2 + x + 1.5;
  EXPECT_THAT(y.toArray(), ElementsAre(4.5, 5.5, 6.5, 7.5));
  y += 0.5;
  EXPECT_THAT(y.toArray(), ElementsAre(5, 6, 7, 8));

  // sub
  y = 2 - x - 1.5;
  EXPECT_THAT(y.toArray(), ElementsAre(-0.5, -1.5, -2.5, -3.5));
  y -= 0.5;
  EXPECT_THAT(y.toArray(), ElementsAre(-1, -2, -3, -4));

  // mul
  y = 2 * x * 1.5;
  EXPECT_THAT(y.toArray(), ElementsAre(3, 6, 9, 12));
  y *= 2;
  EXPECT_THAT(y.toArray(), ElementsAre(6, 12, 18, 24));

  // div
  y = 12 / x / 2;
  EXPECT_THAT(y.toArray(), ElementsAre(6, 3, 2, 1.5));
  y /= 0.5;
  EXPECT_THAT(y.toArray(), ElementsAre(12, 6, 4, 3));
}

TEST(TEST_TensorImpl, math_same_shape) {
  TensorImpl x1({{1, 2}, {3, 4}});
  TensorImpl x2({{2, 3}, {4, 5}});

  auto y = x1 + x2;
  EXPECT_THAT(y.toArray(), ElementsAre(3, 5, 7, 9));
  y += x1;
  EXPECT_THAT(y.toArray(), ElementsAre(4, 7, 10, 13));

  y = x1 - x2;
  EXPECT_THAT(y.toArray(), ElementsAre(-1, -1, -1, -1));
  y -= x1;
  EXPECT_THAT(y.toArray(), ElementsAre(-2, -3, -4, -5));

  y = x1 * x2;
  EXPECT_THAT(y.toArray(), ElementsAre(2, 6, 12, 20));
  y *= x1;
  EXPECT_THAT(y.toArray(), ElementsAre(2, 12, 36, 80));

  y = x1 / x2;
  EXPECT_THAT(y.toArray(), ElementsAre(0.5, 2.f / 3, 0.75, 0.8));
  y /= x1;
  EXPECT_THAT(y.toArray(), ElementsAre(0.5, 1.f / 3, 0.25, 0.2));

  x1 = TensorImpl::scalar(1.f);
  x2 = TensorImpl::scalar(2.f);
  y = x1 - x2;
  EXPECT_THAT(y.toArray(), ElementsAre(-1));
}

TEST(TEST_TensorImpl, math_min) {
  TensorImpl x({{1, 2, 3}, {4, 5, 6}});

  EXPECT_TRUE(TensorImpl::min(x) == 1);

  auto y = TensorImpl::min(x, 0);
  EXPECT_THAT(y.shape(), ElementsAre(3));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 2, 3));

  y = TensorImpl::min(x, 0, true);
  EXPECT_THAT(y.shape(), ElementsAre(1, 3));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 2, 3));

  y = TensorImpl::min(x, 1);
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 4));

  y = TensorImpl::min(x, 1, true);
  EXPECT_THAT(y.shape(), ElementsAre(2, 1));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 4));

  y = TensorImpl::min(x, -1);
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 4));

  y = TensorImpl::min(x, -1, true);
  EXPECT_THAT(y.shape(), ElementsAre(2, 1));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 4));
}

TEST(TEST_TensorImpl, math_max) {
  TensorImpl x({{1, 2, 3}, {4, 5, 6}});

  EXPECT_TRUE(TensorImpl::max(x) == 6);

  auto y = TensorImpl::max(x, 0);
  EXPECT_THAT(y.shape(), ElementsAre(3));
  EXPECT_THAT(y.toArray(), ElementsAre(4, 5, 6));

  y = TensorImpl::max(x, 1);
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toArray(), ElementsAre(3, 6));
}

TEST(TEST_TensorImpl, math_meam) {
  TensorImpl x({{1, 2, 3}, {4, 5, 6}});

  EXPECT_TRUE(TensorImpl::mean(x) == 3.5);

  auto y = TensorImpl::mean(x, 0);
  EXPECT_THAT(y.shape(), ElementsAre(3));
  EXPECT_THAT(y.toArray(), ElementsAre(2.5, 3.5, 4.5));

  y = TensorImpl::mean(x, 1);
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toArray(), ElementsAre(2, 5));
}

TEST(TEST_TensorImpl, math_sum) {
  TensorImpl x({{1, 2, 3}, {4, 5, 6}});

  EXPECT_TRUE(TensorImpl::sum(x) == 21);

  auto y = TensorImpl::sum(x, 0);
  EXPECT_THAT(y.shape(), ElementsAre(3));
  EXPECT_THAT(y.toArray(), ElementsAre(5, 7, 9));

  y = TensorImpl::sum(x, 1);
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toArray(), ElementsAre(6, 15));

  x = TensorImpl({{{4, 2, 3}, {1, 0, 3}}, {{4, 2, 3}, {1, 0, 3}}});
  EXPECT_TRUE(TensorImpl::sum(x) == 26);

  y = TensorImpl::sum(x, 2);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toArray(), ElementsAre(9, 4, 9, 4));

  y = TensorImpl::sum(x, 1);
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toArray(), ElementsAre(5, 2, 6, 5, 2, 6));

  y = TensorImpl::sum(x, 0, true);
  EXPECT_THAT(y.shape(), ElementsAre(1, 2, 3));
  EXPECT_THAT(y.toArray(), ElementsAre(8, 4, 6, 2, 0, 6));
}

TEST(TEST_TensorImpl, math_var) {
  TensorImpl x({{1, 2, 3}, {4, 5, 6}});

  EXPECT_FLOAT_NEAR(TensorImpl::var(x), 2.9166666);

  auto y = TensorImpl::var(x, 0);
  EXPECT_THAT(y.shape(), ElementsAre(3));
  EXPECT_THAT(y.toArray(), ElementsAre(2.25, 2.25, 2.25));

  y = TensorImpl::var(x, 1);
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_FLOAT_NEAR(y[0], 0.666667);
  EXPECT_FLOAT_NEAR(y[1], 0.666667);
}

TEST(TEST_TensorImpl, math_argmin) {
  TensorImpl x({{4, 2, 3}, {1, 0, 3}});

  EXPECT_TRUE(TensorImpl::argmin(x) == 4);

  auto y = TensorImpl::argmin(x, -1);
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 1));

  y = TensorImpl::argmin(x, -1, true);
  EXPECT_THAT(y.shape(), ElementsAre(2, 1));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 1));
}

TEST(TEST_TensorImpl, math_argmax) {
  TensorImpl x({{1, 2, 4}, {1, 0, 3}});

  EXPECT_TRUE(TensorImpl::argmax(x) == 2);

  auto y = TensorImpl::argmax(x, -1);
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toArray(), ElementsAre(2, 2));

  y = TensorImpl::argmax(x, -1, true);
  EXPECT_THAT(y.shape(), ElementsAre(2, 1));
  EXPECT_THAT(y.toArray(), ElementsAre(2, 2));
}

TEST(TEST_TensorImpl, math_sin) {
  TensorImpl x({{1, 2}, {3, 4}});
  auto y = TensorImpl::sin(x);
  EXPECT_THAT(y.toArray(),
              ElementsAre(std::sin(1), std::sin(2), std::sin(3), std::sin(4)));
}

TEST(TEST_TensorImpl, math_cos) {
  TensorImpl x({{1, 2}, {3, 4}});
  auto y = TensorImpl::cos(x);
  EXPECT_THAT(y.toArray(),
              ElementsAre(std::cos(1), std::cos(2), std::cos(3), std::cos(4)));
}

TEST(TEST_TensorImpl, math_sqrt) {
  TensorImpl x({{1, 2}, {3, 4}});
  auto y = TensorImpl::sqrt(x);
  EXPECT_THAT(y.toArray(), ElementsAre(std::sqrt(1), std::sqrt(2), std::sqrt(3),
                                       std::sqrt(4)));
}

TEST(TEST_TensorImpl, math_tanh) {
  TensorImpl x({{1, 2}, {3, 4}});
  auto y = TensorImpl::tanh(x);
  EXPECT_NEAR(y[0], std::tanh(1), 1e-4);
  EXPECT_NEAR(y[1], std::tanh(2), 1e-4);
  EXPECT_NEAR(y[2], std::tanh(3), 1e-4);
  EXPECT_NEAR(y[3], std::tanh(4), 1e-4);
}

TEST(TEST_TensorImpl, math_exp) {
  TensorImpl x({{1, 2}, {3, 4}});
  auto y = TensorImpl::exp(x);
  EXPECT_THAT(y.toArray(),
              ElementsAre(std::exp(1), std::exp(2), std::exp(3), std::exp(4)));
}

TEST(TEST_TensorImpl, math_log) {
  TensorImpl x({{1, 2}, {3, 4}});
  auto y = TensorImpl::log(x);
  EXPECT_THAT(y.toArray(),
              ElementsAre(std::log(1), std::log(2), std::log(3), std::log(4)));
}

TEST(TEST_TensorImpl, math_pow) {
  auto x1 = TensorImpl::arange(0, 6);
  auto y = TensorImpl::pow(x1, 3);
  EXPECT_THAT(y.toArray(), ElementsAre(0, 1, 8, 27, 64, 125));

  auto x2 = TensorImpl({1.0, 2.0, 3.0, 3.0, 2.0, 1.0});
  y = TensorImpl::pow(x1, x2);
  EXPECT_THAT(y.toArray(), ElementsAre(0, 1, 8, 27, 16, 5));

  x2 = TensorImpl({{1, 2, 3, 3, 2, 1}, {1, 2, 3, 3, 2, 3}});
  y = TensorImpl::pow(x1, x2);
  EXPECT_THAT(y.shape(), ElementsAre(2, 6));
  EXPECT_THAT(y.toArray(),
              ElementsAre(0, 1, 8, 27, 16, 5, 0, 1, 8, 27, 16, 125));
}

TEST(TEST_TensorImpl, math_dot) {
  Array2d d1 = {{1, 2}, {3, 4}};
  Array2d d2 = {{2, 3}, {4, 5}};
  auto y = TensorImpl::dot(TensorImpl(d1), TensorImpl(d2));
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toArray(), ElementsAre(10, 13, 22, 29));

  Array1d d3 = {1, 2, 3};
  y = TensorImpl::dot(TensorImpl(d3), TensorImpl(d3));
  EXPECT_TRUE(y.isScalar());
  EXPECT_THAT(y.toArray(), ElementsAre(14));

  y = TensorImpl::dot(TensorImpl(d3), 0.2f);
  EXPECT_THAT(y.shape(), ElementsAre(3));
  EXPECT_THAT(y.toArray(), ElementsAre(0.2f, 0.4f, 0.6f));

  y = TensorImpl::dot(0.2f, TensorImpl(d3));
  EXPECT_THAT(y.shape(), ElementsAre(3));
  EXPECT_THAT(y.toArray(), ElementsAre(0.2f, 0.4f, 0.6f));
}

TEST(TEST_TensorImpl, math_matmul) {
  Array2d d1 = {{1, 2}, {3, 4}};
  Array2d d2 = {{2, 3}, {4, 5}};
  auto y = TensorImpl::matmul(TensorImpl(d1), TensorImpl(d2));
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toArray(), ElementsAre(10, 13, 22, 29));

  Array2d d3 = {{1, 2, 3}, {4, 5, 6}};
  Array2d d4 = {{2, 3}, {4, 5}, {6, 7}};
  y = TensorImpl::matmul(TensorImpl(d3), TensorImpl(d4));
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toArray(), ElementsAre(28, 34, 64, 79));

  Array2d d5 = {{1, 0}, {0, 1}};
  Array1d d6 = {1, 2};
  y = TensorImpl::matmul(TensorImpl(d5), TensorImpl(d6));
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 2));

  y = TensorImpl::matmul(TensorImpl(d6), TensorImpl(d5));
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toArray(), ElementsAre(1, 2));

  Array1d d7 = {2};
  y = TensorImpl::matmul(TensorImpl(d7), TensorImpl(d7));
  EXPECT_TRUE(y.isScalar());
  EXPECT_THAT(y.toArray(), ElementsAre(4));

  // broadcast
  auto a = TensorImpl::arange(0, 2 * 2 * 4).reshape({2, 2, 4});
  auto b = TensorImpl::arange(0, 2 * 2 * 4).reshape({1, 2, 4, 2});
  auto c = TensorImpl::arange(0, 1 * 2 * 4).reshape({1, 4, 2});
  auto d = TensorImpl::matmul(a, b);
  auto e = TensorImpl::matmul(a, c);

  EXPECT_THAT(d.shape(), ElementsAre(1, 2, 2, 2));
  EXPECT_THAT(d.toArray(), ElementsAre(28, 34, 76, 98, 428, 466, 604, 658));

  EXPECT_THAT(e.shape(), ElementsAre(2, 2, 2));
  EXPECT_THAT(e.toArray(), ElementsAre(28, 34, 76, 98, 124, 162, 172, 226));
}

TEST(TEST_TensorImpl, math_matmulTrans) {
  Array2d d1 = {{1, 2}, {3, 4}};
  Array2d d2 = {{2, 3}, {4, 5}};
  auto y = TensorImpl::matmulTrans(TensorImpl(d1), TensorImpl(d2));
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toArray(), ElementsAre(8, 14, 18, 32));

  Array2d d3 = {{1, 2, 3}, {4, 5, 6}};
  Array2d d4 = {{2, 4, 6}, {3, 5, 7}};
  y = TensorImpl::matmulTrans(TensorImpl(d3), TensorImpl(d4));
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toArray(), ElementsAre(28, 34, 64, 79));
}

TEST(TEST_TensorImpl, math_broadcast) {
  Array2d d1 = {{1, 2}};
  Array2d d2 = {{2, 3}, {4, 5}};
  Array2d d3 = {{2}, {4}};
  Array1d d4 = {1, 2};
  Array1d d5 = {1};

  auto y = TensorImpl(d1) + TensorImpl(d2);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toArray(), ElementsAre(3, 5, 5, 7));

  y = TensorImpl(d2) + TensorImpl(d3);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toArray(), ElementsAre(4, 5, 8, 9));

  y = TensorImpl(d2) + TensorImpl(d4);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toArray(), ElementsAre(3, 5, 5, 7));

  y = TensorImpl(d2) + TensorImpl(d5);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toArray(), ElementsAre(3, 4, 5, 6));

  y = TensorImpl(d2) + TensorImpl::scalar(0.5);
  EXPECT_THAT(y.toArray(), ElementsAre(2.5, 3.5, 4.5, 5.5));
}

TEST(TEST_TensorImpl, basic_im2col_col2im) {
  auto input = TensorImpl(
      {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}});
  input.reshape({1, 1, 4, 4});
  auto col = input.im2col(2, 2, 0);
  EXPECT_THAT(col.shape(), ElementsAre(4, 4));
  EXPECT_THAT(col.toArray(), ElementsAre(1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14,
                                         11, 12, 15, 16));

  auto r = col.col2im(input.shape(), 2, 2, 0);
  EXPECT_EQ(r.shape(), input.shape());
  EXPECT_EQ(r.toArray(), input.toArray());

  col = input.im2col(2, 3, 0);
  EXPECT_THAT(col.shape(), ElementsAre(1, 4));
  EXPECT_THAT(col.toArray(), ElementsAre(1, 2, 5, 6));

  r = col.col2im(input.shape(), 2, 3, 0);
  EXPECT_EQ(r.shape(), input.shape());
  EXPECT_THAT(r.toArray(),
              ElementsAre(1, 2, 0, 0, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));

  col = input.im2col(3, 2, 0);
  EXPECT_THAT(col.shape(), ElementsAre(1, 9));
  EXPECT_THAT(col.toArray(), ElementsAre(1, 2, 3, 5, 6, 7, 9, 10, 11));

  r = col.col2im(input.shape(), 3, 2, 0);
  EXPECT_EQ(r.shape(), input.shape());
  EXPECT_THAT(r.toArray(),
              ElementsAre(1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 0, 0, 0, 0));
}
