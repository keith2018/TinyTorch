/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "TinyTorch.h"
#include "test.h"

using namespace tinytorch;

TEST(TEST_Operation, basic_fill) {
  auto x = Tensor::randn({2, 3});

  x.fill(-1);
  EXPECT_THAT(x.toList<float>(), ElementsAre(-1, -1, -1, -1, -1, -1));

  x.fill(2);
  EXPECT_THAT(x.toList<float>(), ElementsAre(2, 2, 2, 2, 2, 2));
}

TEST(TEST_Operation, basic_clamp) {
  Tensor x;

  x = Tensor(Array1d<float>{1, 2, 3});
  op::clampMinInplace(x, Tensor::scalar(2.3f));
  EXPECT_THAT(x.toList<float>(), ElementsAre(2.3, 2.3, 3));

  x = Tensor(Array1d<float>{1, 2, 3});
  op::clampMaxInplace(x, Tensor::scalar(2.2f));
  EXPECT_THAT(x.toList<float>(), ElementsAre(1, 2, 2.2));

  x = Tensor(Array1d<float>{1, 2, 3});
  op::clampInplace(x, Tensor::scalar(1.2f), Tensor::scalar(2.2f));
  EXPECT_THAT(x.toList<float>(), ElementsAre(1.2, 2, 2.2));
}

TEST(TEST_Operation, math_sin) {
  Tensor x(Array2d<float>{{1, 2}, {3, 4}});
  auto y = op::sin(x);
  auto arr = y.toList<float>();
  EXPECT_NEAR(arr[0], std::sin(1), 1e-4);
  EXPECT_NEAR(arr[1], std::sin(2), 1e-4);
  EXPECT_NEAR(arr[2], std::sin(3), 1e-4);
  EXPECT_NEAR(arr[3], std::sin(4), 1e-4);
}

TEST(TEST_Operation, math_cos) {
  Tensor x(Array2d<float>{{1, 2}, {3, 4}});
  auto y = op::cos(x);
  auto arr = y.toList<float>();
  EXPECT_NEAR(arr[0], std::cos(1), 1e-4);
  EXPECT_NEAR(arr[1], std::cos(2), 1e-4);
  EXPECT_NEAR(arr[2], std::cos(3), 1e-4);
  EXPECT_NEAR(arr[3], std::cos(4), 1e-4);
}

TEST(TEST_Operation, math_sqrt) {
  Tensor x(Array2d<float>{{1, 2}, {3, 4}});
  auto y = op::sqrt(x);
  auto arr = y.toList<float>();
  EXPECT_NEAR(arr[0], std::sqrt(1), 1e-4);
  EXPECT_NEAR(arr[1], std::sqrt(2), 1e-4);
  EXPECT_NEAR(arr[2], std::sqrt(3), 1e-4);
  EXPECT_NEAR(arr[3], std::sqrt(4), 1e-4);
}

TEST(TEST_Operation, math_tanh) {
  Tensor x(Array2d<float>{{1, 2}, {3, 4}});
  auto y = op::tanh(x);
  auto arr = y.toList<float>();
  EXPECT_NEAR(arr[0], std::tanh(1), 1e-4);
  EXPECT_NEAR(arr[1], std::tanh(2), 1e-4);
  EXPECT_NEAR(arr[2], std::tanh(3), 1e-4);
  EXPECT_NEAR(arr[3], std::tanh(4), 1e-4);
}

TEST(TEST_Operation, math_exp) {
  Tensor x(Array2d<float>{{1, 2}, {3, 4}});
  auto y = op::exp(x);
  auto arr = y.toList<float>();
  EXPECT_NEAR(arr[0], std::exp(1), 1e-4);
  EXPECT_NEAR(arr[1], std::exp(2), 1e-4);
  EXPECT_NEAR(arr[2], std::exp(3), 1e-4);
  EXPECT_NEAR(arr[3], std::exp(4), 1e-4);
}

TEST(TEST_Operation, math_log) {
  Tensor x(Array2d<float>{{1, 2}, {3, 4}});
  auto y = op::log(x);
  auto arr = y.toList<float>();
  EXPECT_NEAR(arr[0], std::log(1), 1e-4);
  EXPECT_NEAR(arr[1], std::log(2), 1e-4);
  EXPECT_NEAR(arr[2], std::log(3), 1e-4);
  EXPECT_NEAR(arr[3], std::log(4), 1e-4);
}

TEST(TEST_Operation, math_pow) {
  auto x1 = Tensor::arange<float>(0.f, 6.f);
  auto y = op::pow(x1, Tensor::scalar(3.f));
  EXPECT_THAT(y.toList<float>(), ElementsAre(0, 1, 8, 27, 64, 125));

  auto x2 = Tensor(Array1d<float>{1.0, 2.0, 3.0, 3.0, 2.0, 1.0});
  y = op::pow(x1, x2);
  EXPECT_THAT(y.toList<float>(), ElementsAre(0, 1, 8, 27, 16, 5));

  x2 = Tensor(Array2d<float>{{1, 2, 3, 3, 2, 1}, {1, 2, 3, 3, 2, 3}});
  y = op::pow(x1, x2);
  EXPECT_THAT(y.shape(), ElementsAre(2, 6));
  EXPECT_THAT(y.toList<float>(), ElementsAre(0, 1, 8, 27, 16, 5, 0, 1, 8, 27, 16, 125));
}

TEST(TEST_Operation, math_scalar) {
  Tensor x(Array2d<float>{{1, 2}, {3, 4}});

  // add
  Tensor y = 2.f + x + 1.5f;
  EXPECT_THAT(y.toList<float>(), ElementsAre(4.5, 5.5, 6.5, 7.5));
  y += 0.5f;
  EXPECT_THAT(y.toList<float>(), ElementsAre(5, 6, 7, 8));

  // sub
  y = 2.f - x - 1.5f;
  EXPECT_THAT(y.toList<float>(), ElementsAre(-0.5, -1.5, -2.5, -3.5));
  y -= 0.5f;
  EXPECT_THAT(y.toList<float>(), ElementsAre(-1, -2, -3, -4));

  // mul
  y = 2.f * x * 1.5f;
  EXPECT_THAT(y.toList<float>(), ElementsAre(3, 6, 9, 12));
  y *= 2.f;
  EXPECT_THAT(y.toList<float>(), ElementsAre(6, 12, 18, 24));

  // div
  y = 12.f / x / 2.f;
  EXPECT_THAT(y.toList<float>(), ElementsAre(6, 3, 2, 1.5));
  y /= 0.5f;
  EXPECT_THAT(y.toList<float>(), ElementsAre(12, 6, 4, 3));
}

TEST(TEST_Operation, math_same_shape) {
  Tensor x1(Array2d<float>{{1, 2}, {3, 4}});
  Tensor x2(Array2d<float>{{2, 3}, {4, 5}});

  auto y = x1 + x2;
  EXPECT_THAT(y.toList<float>(), ElementsAre(3, 5, 7, 9));
  y += x1;
  EXPECT_THAT(y.toList<float>(), ElementsAre(4, 7, 10, 13));

  y = x1 - x2;
  EXPECT_THAT(y.toList<float>(), ElementsAre(-1, -1, -1, -1));
  y -= x1;
  EXPECT_THAT(y.toList<float>(), ElementsAre(-2, -3, -4, -5));

  y = x1 * x2;
  EXPECT_THAT(y.toList<float>(), ElementsAre(2, 6, 12, 20));
  y *= x1;
  EXPECT_THAT(y.toList<float>(), ElementsAre(2, 12, 36, 80));

  y = x1 / x2;
  EXPECT_THAT(y.toList<float>(), ElementsAre(0.5, 2.f / 3, 0.75, 0.8));
  y /= x1;
  EXPECT_THAT(y.toList<float>(), ElementsAre(0.5, 1.f / 3, 0.25, 0.2));

  x1 = Tensor::scalar(1.f);
  x2 = Tensor::scalar(2.f);
  y = x1 - x2;
  EXPECT_THAT(y.toList<float>(), ElementsAre(-1));
}

TEST(TEST_Operation, math_broadcast) {
  Array2d<float> d1 = {{1, 2}};
  Array2d<float> d2 = {{2, 3}, {4, 5}};
  Array2d<float> d3 = {{2}, {4}};
  Array1d<float> d4 = {1, 2};
  Array1d<float> d5 = {1};

  auto y = Tensor(d1) + Tensor(d2);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toList<float>(), ElementsAre(3, 5, 5, 7));

  y = Tensor(d2) + Tensor(d3);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toList<float>(), ElementsAre(4, 5, 8, 9));

  y = Tensor(d2) + Tensor(d4);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toList<float>(), ElementsAre(3, 5, 5, 7));

  y = Tensor(d2) + Tensor(d5);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toList<float>(), ElementsAre(3, 4, 5, 6));

  y = Tensor(d2) + Tensor::scalar(0.5f);
  EXPECT_THAT(y.toList<float>(), ElementsAre(2.5, 3.5, 4.5, 5.5));
}

TEST(TEST_Operation, math_broadcast_inplace) {
  Array2d<float> d1 = {{1, 2}};
  Array2d<float> d2 = {{2, 3}, {4, 5}};
  Array2d<float> d3 = {{2}, {4}};
  Array1d<float> d4 = {1, 2};
  Array1d<float> d5 = {1};

  auto y = Tensor(d1);
  y += Tensor(d2);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toList<float>(), ElementsAre(3, 5, 5, 7));

  y = Tensor(d3);
  y -= Tensor(d2);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toList<float>(), ElementsAre(0, -1, 0, -1));

  y = Tensor(d4);
  y *= Tensor(d2);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toList<float>(), ElementsAre(2, 6, 4, 10));

  y = Tensor(d5);
  y /= Tensor(d2);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toList<float>(), ElementsAre(0.5, 1.0 / 3.0, 0.25, 0.2));

  y = Tensor(d2);
  y += Tensor::scalar(0.5f);
  EXPECT_THAT(y.toList<float>(), ElementsAre(2.5, 3.5, 4.5, 5.5));
}

TEST(TEST_Operation, math_compare) {
  Tensor x(Array2d<float>{{1, 2, 3}, {3, 4, 5}});
  Tensor x2(Array2d<float>{{3, 4, 5}, {1, 2, 3}});

  auto y = x > 2.f;
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toList<uint8_t>(), ElementsAre(0, 0, 1, 1, 1, 1));

  y = x < 3.f;
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toList<uint8_t>(), ElementsAre(1, 1, 0, 0, 0, 0));

  y = x < x2;
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toList<uint8_t>(), ElementsAre(1, 1, 1, 0, 0, 0));

  y = x == Tensor(Array2d<float>{{1, 1, 3}, {1, 4, 5}});
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toList<uint8_t>(), ElementsAre(1, 0, 1, 0, 1, 1));

  y = x != Tensor(Array2d<float>{{1, 1, 3}, {1, 4, 5}});
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toList<uint8_t>(), ElementsAre(0, 1, 0, 1, 0, 0));

  y = Tensor::maximum(x, x2);
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toList<float>(), ElementsAre(3, 4, 5, 3, 4, 5));

  y = Tensor::minimum(x, x2);
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toList<float>(), ElementsAre(1, 2, 3, 1, 2, 3));
}

TEST(TEST_Operation, math_min) {
  Tensor x(Array2d<float>{{1, 2, 3}, {4, 5, 6}});

  EXPECT_TRUE(op::min(x).item<float>() == 1);

  auto y = op::minOnDim(x, 0, false).first;
  EXPECT_THAT(y.shape(), ElementsAre(3));
  EXPECT_THAT(y.toList<float>(), ElementsAre(1, 2, 3));

  y = op::minOnDim(x, 0, true).first;
  EXPECT_THAT(y.shape(), ElementsAre(1, 3));
  EXPECT_THAT(y.toList<float>(), ElementsAre(1, 2, 3));

  y = op::minOnDim(x, 1, false).first;
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toList<float>(), ElementsAre(1, 4));

  y = op::minOnDim(x, 1, true).first;
  EXPECT_THAT(y.shape(), ElementsAre(2, 1));
  EXPECT_THAT(y.toList<float>(), ElementsAre(1, 4));

  y = op::minOnDim(x, -1, false).first;
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toList<float>(), ElementsAre(1, 4));

  y = op::minOnDim(x, -1, true).first;
  EXPECT_THAT(y.shape(), ElementsAre(2, 1));
  EXPECT_THAT(y.toList<float>(), ElementsAre(1, 4));
}

TEST(TEST_Operation, math_max_01) {
  Tensor x(Array2d<float>{{1, 2, 3}, {4, 5, 6}});

  EXPECT_TRUE(op::max(x).item<float>() == 6);

  auto y = op::maxOnDim(x, 0, false).first;
  EXPECT_THAT(y.shape(), ElementsAre(3));
  EXPECT_THAT(y.toList<float>(), ElementsAre(4, 5, 6));

  y = op::maxOnDim(x, 1, false).first;
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toList<float>(), ElementsAre(3, 6));
}

TEST(TEST_Operation, math_max_02) {
  auto x = Tensor::arange<float>(0.f, 24.f, 1.f);
  x.reshape({2, 3, 4});

  auto y = op::maxOnDim(x, 0, true).first;
  EXPECT_THAT(y.shape(), ElementsAre(1, 3, 4));
  EXPECT_THAT(y.toList<float>(), ElementsAre(12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23));

  y = op::maxOnDim(x, 1, true).first;
  EXPECT_THAT(y.shape(), ElementsAre(2, 1, 4));
  EXPECT_THAT(y.toList<float>(), ElementsAre(8, 9, 10, 11, 20, 21, 22, 23));

  y = op::maxOnDim(x, 2, true).first;
  EXPECT_THAT(y.shape(), ElementsAre(2, 3, 1));
  EXPECT_THAT(y.toList<float>(), ElementsAre(3, 7, 11, 15, 19, 23));
}

TEST(TEST_Operation, math_argmin_01) {
  Tensor x(Array2d<float>{{4, 2, 3}, {1, 0, 3}});

  EXPECT_TRUE(op::argmin(x).item<int64_t>() == 4);

  auto y = op::minOnDim(x, -1, false).second;
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toList<int64_t>(), ElementsAre(1, 1));

  y = op::minOnDim(x, -1, true).second;
  EXPECT_THAT(y.shape(), ElementsAre(2, 1));
  EXPECT_THAT(y.toList<int64_t>(), ElementsAre(1, 1));
}

TEST(TEST_Operation, math_argmin_02) {
  Tensor x(Array1d<float>{3.14, 7.89, 1.23, 4.56, 9.01, 2.34, 5.67, 8.90, 0.12, 6.78, 3.45,
                          7.12, 1.56, 4.89, 9.34, 2.67, 5.89, 8.23, 0.45, 6.12, 3.78, 7.45,
                          1.89, 4.23, 9.56, 2.12, 5.34, 8.67, 0.78, 6.45, 3.12, 7.78});
  x.reshape({2, 2, 2, 4});
  auto y = op::minOnDim(x, (2), true).second;
  EXPECT_THAT(y.shape(), ElementsAre(2, 2, 1, 4));
  EXPECT_THAT(y.toList<int64_t>(), ElementsAre(0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1));
}

TEST(TEST_Operation, math_argmax_01) {
  Tensor x(Array2d<float>{{1, 2, 4}, {1, 0, 3}});

  EXPECT_TRUE(op::argmax(x).item<int64_t>() == 2);

  auto y = op::maxOnDim(x, -1, false).second;
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toList<int64_t>(), ElementsAre(2, 2));

  y = op::maxOnDim(x, -1, true).second;
  EXPECT_THAT(y.shape(), ElementsAre(2, 1));
  EXPECT_THAT(y.toList<int64_t>(), ElementsAre(2, 2));
}

TEST(TEST_Operation, math_argmax_02) {
  Tensor x(Array1d<float>{1, 2, 3, 5, 6,  7,  9,  10, 11, 2, 3, 4, 6,  7,  8,  10, 11, 12,
                          5, 6, 7, 9, 10, 11, 13, 14, 15, 6, 7, 8, 10, 11, 12, 14, 15, 16});
  x.reshape({4, 9});

  auto y = op::maxOnDim(x, 1, false).second;
  EXPECT_THAT(y.shape(), ElementsAre(4));
  EXPECT_THAT(y.toList<int64_t>(), ElementsAre(8, 8, 8, 8));
}

TEST(TEST_Operation, math_sum) {
  Tensor x(Array2d<float>{{1, 2, 3}, {4, 5, 6}});

  EXPECT_TRUE(op::sum(x).item<float>() == 21);

  auto y = op::sumOnDim(x, 0, false);
  EXPECT_THAT(y.shape(), ElementsAre(3));
  EXPECT_THAT(y.toList<float>(), ElementsAre(5, 7, 9));

  y = op::sumOnDim(x, 1, false);
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toList<float>(), ElementsAre(6, 15));

  y = op::sumOnDims(x, IntArrayView{0, 1}, true);
  EXPECT_THAT(y.shape(), ElementsAre(1, 1));
  EXPECT_THAT(y.toList<float>(), ElementsAre(21));
  //
  x = Tensor(Array3d<float>{{{4, 2, 3}, {1, 0, 3}}, {{4, 2, 3}, {1, 0, 3}}});
  y = op::sumOnDim(x, 2, false);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toList<float>(), ElementsAre(9, 4, 9, 4));

  y = op::sumOnDim(x, 1, false);
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toList<float>(), ElementsAre(5, 2, 6, 5, 2, 6));

  y = op::sumOnDim(x, 0, true);
  EXPECT_THAT(y.shape(), ElementsAre(1, 2, 3));
  EXPECT_THAT(y.toList<float>(), ElementsAre(8, 4, 6, 2, 0, 6));
}

TEST(TEST_Operation, math_meam) {
  Tensor x(Array2d<float>{{1, 2, 3}, {4, 5, 6}});

  EXPECT_TRUE(op::mean(x).item<float>() == 3.5);

  auto y = op::meanOnDim(x, 0, false);
  EXPECT_THAT(y.shape(), ElementsAre(3));
  EXPECT_THAT(y.toList<float>(), ElementsAre(2.5, 3.5, 4.5));

  y = op::meanOnDim(x, 1, false);
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toList<float>(), ElementsAre(2, 5));

  y = op::meanOnDims(x, IntArrayView{0, 1}, true);
  EXPECT_THAT(y.shape(), ElementsAre(1, 1));
  EXPECT_THAT(y.toList<float>(), ElementsAre(3.5));

  y = op::meanOnDims(x, IntArrayView{0, 1}, false);
  EXPECT_TRUE(y.dim() == 0);
  EXPECT_THAT(y.toList<float>(), ElementsAre(3.5));
}

TEST(TEST_Operation, math_var_01) {
  Tensor x(Array2d<float>{{1, 2, 3}, {4, 5, 6}});

  EXPECT_FLT_NEAR(op::varMean(x, false).first.item<float>(), 2.9166666);
  EXPECT_FLT_NEAR(op::varMean(x, true).first.item<float>(), 3.5);

  auto y = op::varMeanOnDim(x, 0, true, true).first;
  EXPECT_THAT(y.shape(), ElementsAre(1, 3));
  EXPECT_THAT(y.toList<float>(), ElementsAre(4.5, 4.5, 4.5));

  y = op::varMeanOnDim(x, 1, true, true).first;
  EXPECT_THAT(y.shape(), ElementsAre(2, 1));
  EXPECT_TRUE(VectorNear(y.toList<float>(), {1.0, 1.0}));

  y = op::varMeanOnDims(x, IntArrayView{0, 1}, true, true).first;
  EXPECT_THAT(y.shape(), ElementsAre(1, 1));
  EXPECT_TRUE(VectorNear(y.toList<float>(), {3.5}));
}

TEST(TEST_Operation, math_var_02) {
  Tensor x(Array1d<float>{3.14, 7.89, 1.23, 4.56, 9.01, 2.34, 5.67, 8.90, 0.12, 6.78, 3.45,
                          7.12, 1.56, 4.89, 9.34, 2.67, 5.89, 8.23, 0.45, 6.12, 3.78, 7.45,
                          1.89, 4.23, 9.56, 2.12, 5.34, 8.67, 0.78, 6.45, 3.12, 7.78});
  x.reshape({2, 2, 2, 4});

  auto y = op::varMeanOnDim(x, 0, true, true).first;
  EXPECT_THAT(y.shape(), ElementsAre(1, 2, 2, 4));
  EXPECT_TRUE(VectorNear(y.toList<float>(), {3.7812, 0.0578, 0.3042, 1.2168, 13.6765, 13.0560, 7.1442, 10.9044, 44.5568,
                                             10.8578, 1.7861, 1.2013, 0.3042, 1.2168, 19.3442, 13.0561}));

  y = op::varMeanOnDim(x, 1, true, true).first;
  EXPECT_THAT(y.shape(), ElementsAre(2, 1, 2, 4));
  EXPECT_TRUE(VectorNear(y.toList<float>(), {4.5602, 0.6160, 2.4642, 3.2768, 27.7513, 3.2512, 6.7345, 19.4064, 6.7345,
                                             18.6660, 11.9561, 3.2513, 4.5000, 0.5000, 0.7564, 6.3013}));

  y = op::varMeanOnDims(x, IntArrayView{0, 1}, true, true).first;
  EXPECT_THAT(y.shape(), ElementsAre(1, 1, 2, 4));
  EXPECT_TRUE(VectorNear(y.toList<float>(), {16.1479, 7.9826, 4.9094, 2.9820, 13.7604, 4.9578, 10.8303, 8.5854}));

  y = op::varMeanOnDims(x, IntArrayView{1, 2}, true, true).first;
  EXPECT_THAT(y.shape(), ElementsAre(2, 1, 1, 4));
  EXPECT_TRUE(VectorNear(y.toList<float>(), {15.2235, 5.9019, 11.9586, 7.5621, 13.6275, 7.4389, 4.2882, 3.8282}));
}

TEST(TEST_Operation, basic_reshape) {
  Tensor x(Array3d<float>{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
  auto y = op::reshape(x, IntArrayView{2, 4});
  EXPECT_THAT(y.shape(), ElementsAre(2, 4));

  y = op::reshape(x, IntArrayView{2, -1});
  EXPECT_THAT(y.shape(), ElementsAre(2, 4));

  y = op::reshape(x, IntArrayView{-1, 2});
  EXPECT_THAT(y.shape(), ElementsAre(4, 2));
}

TEST(TEST_Operation, basic_flatten) {
  Tensor x(Array3d<float>{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
  auto y = op::flatten(x, 0, -1);
  EXPECT_THAT(y.shape(), ElementsAre(8));

  y = op::flatten(x, 1, -1);
  EXPECT_THAT(y.shape(), ElementsAre(2, 4));

  y = op::flatten(x, 1, 2);
  EXPECT_THAT(y.shape(), ElementsAre(2, 4));
}

TEST(TEST_Operation, basic_unflatten) {
  auto y = op::unflatten(Tensor::randn({3, 4, 1}), 1, IntArrayView{2, 2});
  EXPECT_THAT(y.shape(), ElementsAre(3, 2, 2, 1));

  y = op::unflatten(Tensor::randn({3, 4, 1}), 1, IntArrayView{-1, 2});
  EXPECT_THAT(y.shape(), ElementsAre(3, 2, 2, 1));

  y = op::unflatten(Tensor::randn({5, 12, 3}), -2, IntArrayView{2, 2, 3, 1, 1});
  EXPECT_THAT(y.shape(), ElementsAre(5, 2, 2, 3, 1, 1, 3));
}

TEST(TEST_Operation, basic_squeeze) {
  auto x = Tensor::randn({2, 1, 2, 1, 2});
  auto y = op::squeeze(x, IntArrayView{});
  EXPECT_THAT(y.shape(), ElementsAre(2, 2, 2));
  y = op::squeeze(x, IntArrayView{0});
  EXPECT_THAT(y.shape(), ElementsAre(2, 1, 2, 1, 2));
  y = op::squeeze(x, IntArrayView{1});
  EXPECT_THAT(y.shape(), ElementsAre(2, 2, 1, 2));
  y = op::squeeze(x, IntArrayView{1, 2, 3});
  EXPECT_THAT(y.shape(), ElementsAre(2, 2, 2));
}

TEST(TEST_Operation, basic_unsqueeze) {
  auto x = Tensor::randn({2, 1, 2});
  auto y = op::unsqueeze(x, 0);
  EXPECT_THAT(y.shape(), ElementsAre(1, 2, 1, 2));
  y = op::unsqueeze(x, 1);
  EXPECT_THAT(y.shape(), ElementsAre(2, 1, 1, 2));
  y = op::unsqueeze(x, -1);
  EXPECT_THAT(y.shape(), ElementsAre(2, 1, 2, 1));
}

TEST(TEST_Operation, basic_transpose) {
  auto x = Tensor(Array3d<float>{{{1, 2, 3}, {4, 5, 6}}});
  auto y = op::transpose(x, 0, 1);
  EXPECT_THAT(y.shape(), ElementsAre(2, 1, 3));
  EXPECT_THAT(y.toList<float>(), ElementsAre(1, 2, 3, 4, 5, 6));

  y = op::transpose(x, 1, 2);
  EXPECT_THAT(y.shape(), ElementsAre(1, 3, 2));
  EXPECT_THAT(y.toList<float>(), ElementsAre(1, 4, 2, 5, 3, 6));

  y = op::transpose(x, 0, 2);
  EXPECT_THAT(y.shape(), ElementsAre(3, 2, 1));
  EXPECT_THAT(y.toList<float>(), ElementsAre(1, 4, 2, 5, 3, 6));
}

TEST(TEST_Operation, basic_transpose2d) {
  auto x = Tensor(Array2d<float>{{1, 2, 3}, {4, 5, 6}});
  auto y = op::transpose2d(x);
  EXPECT_THAT(y.shape(), ElementsAre(3, 2));
  EXPECT_THAT(y.toList<float>(), ElementsAre(1, 4, 2, 5, 3, 6));

  x = Tensor(Array2d<float>{{1, 2}, {3, 4}, {5, 6}});
  y = op::transpose2d(x);
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toList<float>(), ElementsAre(1, 3, 5, 2, 4, 6));
}

TEST(TEST_Operation, basic_permute) {
  Tensor x(Array1d<float>{1, 2, 3});
  auto y = op::permuteAll(x);
  EXPECT_TRUE(y.shape() == x.shape());
  EXPECT_TRUE(y.toList<float>() == x.toList<float>());

  x = Tensor(Array2d<float>{{1, 2}, {3, 4}, {5, 6}});
  y = op::permuteAll(x);
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toList<float>(), ElementsAre(1, 3, 5, 2, 4, 6));

  x = Tensor(Array3d<float>{{{1, 2, 3}, {4, 5, 6}}});
  y = op::permuteAll(x);
  EXPECT_THAT(y.shape(), ElementsAre(3, 2, 1));
  EXPECT_THAT(y.toList<float>(), ElementsAre(1, 4, 2, 5, 3, 6));

  y = op::permute(x, IntArrayView{1, 0, 2});
  EXPECT_THAT(y.shape(), ElementsAre(2, 1, 3));
  EXPECT_THAT(y.toList<float>(), ElementsAre(1, 2, 3, 4, 5, 6));

  x = Tensor::arange<float>(0.f, 8.f);
  x.reshape({1, 2, 2, 2});

  y = op::permute(x, IntArrayView{0, 3, 1, 2});
  EXPECT_THAT(y.shape(), ElementsAre(1, 2, 2, 2));
  EXPECT_THAT(y.toList<float>(), ElementsAre(0, 2, 4, 6, 1, 3, 5, 7));

  y = op::permuteAll(x);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2, 2, 1));
  EXPECT_THAT(y.toList<float>(), ElementsAre(0, 4, 2, 6, 1, 5, 3, 7));
}

TEST(TEST_Operation, basic_index) {
  Tensor x(Array2d<float>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});

  auto y = op::index(x, IntArrayView{0});
  EXPECT_THAT(y.shape(), ElementsAre(3));
  EXPECT_THAT(y.toList<float>(), ElementsAre(1, 2, 3));

  y = op::index(x, IntArrayView{1});
  EXPECT_THAT(y.shape(), ElementsAre(3));
  EXPECT_THAT(y.toList<float>(), ElementsAre(4, 5, 6));

  y = op::index(x, IntArrayView{0, 0});
  EXPECT_TRUE(y.dim() == 0);
  EXPECT_THAT(y.toList<float>(), ElementsAre(1));

  y = op::index(x, IntArrayView{1, 1});
  EXPECT_TRUE(y.dim() == 0);
  EXPECT_THAT(y.toList<float>(), ElementsAre(5));

  auto idx = Tensor(Array1d<int64_t>{-1, 0});
  y = op::indexAdvance(x, ArrayView{idx});
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toList<float>(), ElementsAre(7, 8, 9, 1, 2, 3));

  idx = Tensor(Array1d<int64_t>{1});
  y = op::indexAdvance(x, ArrayView{idx});
  EXPECT_THAT(y.shape(), ElementsAre(1, 3));
  EXPECT_THAT(y.toList<float>(), ElementsAre(4, 5, 6));

  idx = Tensor(Array1d<int64_t>{0, 1});
  y = op::indexAdvance(x, ArrayView{idx});
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toList<float>(), ElementsAre(1, 2, 3, 4, 5, 6));

  auto idx1 = Tensor(Array1d<int64_t>{0, 1});
  auto idx2 = Tensor(Array1d<int64_t>{2, 1});
  y = op::indexAdvance(x, ArrayView{idx1, idx2});
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toList<float>(), ElementsAre(3, 5));

  idx1 = Tensor(Array1d<int64_t>{-1, 1});
  idx2 = Tensor(Array1d<int64_t>{2, -1});
  y = op::indexAdvance(x, ArrayView{idx1, idx2});
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toList<float>(), ElementsAre(9, 6));
}

TEST(TEST_Operation, basic_indexPut) {
  Tensor x(Array2d<float>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});

  op::indexPut(x, IntArrayView{1}, Tensor::scalar(-1));
  EXPECT_THAT(x.toList<float>(), ElementsAre(1, 2, 3, -1, -1, -1, 7, 8, 9));

  op::indexPut(x, IntArrayView{1}, Tensor(Array1d<float>{4, 5, 6}));
  EXPECT_THAT(x.toList<float>(), ElementsAre(1, 2, 3, 4, 5, 6, 7, 8, 9));

  auto idx1 = Tensor(Array1d<int64_t>{-1, 1});
  auto idx2 = Tensor(Array1d<int64_t>{2, -1});
  op::indexPutAdvance(x, ArrayView{idx1, idx2}, Tensor::scalar(-1));
  EXPECT_THAT(x.toList<float>(), ElementsAre(1, 2, 3, 4, 5, -1, 7, 8, -1));

  idx1 = Tensor(Array1d<int64_t>{-1, 1});
  idx2 = Tensor(Array1d<int64_t>{2, -1});
  op::indexPutAdvance(x, ArrayView{idx1, idx2}, Tensor(Array1d<float>{1.2, 2.3}));
  EXPECT_THAT(x.toList<float>(), ElementsAre(1, 2, 3, 4, 5, 2.3, 7, 8, 1.2));
}

TEST(TEST_Operation, basic_tril) {
  auto x = op::tril(Tensor::ones({3, 3}), 0);
  EXPECT_THAT(x.shape(), ElementsAre(3, 3));
  EXPECT_THAT(x.toList<float>(), ElementsAre(1, 0, 0, 1, 1, 0, 1, 1, 1));

  x = op::tril(Tensor::ones({2, 3}), 0);
  EXPECT_THAT(x.shape(), ElementsAre(2, 3));
  EXPECT_THAT(x.toList<float>(), ElementsAre(1, 0, 0, 1, 1, 0));

  x = op::tril(Tensor::ones({3, 3}), 1);
  EXPECT_THAT(x.shape(), ElementsAre(3, 3));
  EXPECT_THAT(x.toList<float>(), ElementsAre(1, 1, 0, 1, 1, 1, 1, 1, 1));

  x = op::tril(Tensor::ones({3, 3}), -1);
  EXPECT_THAT(x.shape(), ElementsAre(3, 3));
  EXPECT_THAT(x.toList<float>(), ElementsAre(0, 0, 0, 1, 0, 0, 1, 1, 0));
}

TEST(TEST_Operation, basic_triu) {
  auto x = op::triu(Tensor::ones({3, 3}), 0);
  EXPECT_THAT(x.shape(), ElementsAre(3, 3));
  EXPECT_THAT(x.toList<float>(), ElementsAre(1, 1, 1, 0, 1, 1, 0, 0, 1));

  x = op::triu(Tensor::ones({2, 3}), 0);
  EXPECT_THAT(x.shape(), ElementsAre(2, 3));
  EXPECT_THAT(x.toList<float>(), ElementsAre(1, 1, 1, 0, 1, 1));

  x = op::triu(Tensor::ones({3, 3}), 1);
  EXPECT_THAT(x.shape(), ElementsAre(3, 3));
  EXPECT_THAT(x.toList<float>(), ElementsAre(0, 1, 1, 0, 0, 1, 0, 0, 0));

  x = op::triu(Tensor::ones({3, 3}), -1);
  EXPECT_THAT(x.shape(), ElementsAre(3, 3));
  EXPECT_THAT(x.toList<float>(), ElementsAre(1, 1, 1, 1, 1, 1, 0, 1, 1));
}

TEST(TEST_Operation, basic_split) {
  Tensor x(Array3d<float>{{{4, 2, 3}, {1, 0, 3}}, {{4, 2, 3}, {1, 0, 3}}});
  auto y = op::split(x, 1, 0);
  EXPECT_TRUE(y.size() == 2);
  EXPECT_THAT(y[0].shape(), ElementsAre(1, 2, 3));
  EXPECT_THAT(y[0].toList<float>(), ElementsAre(4, 2, 3, 1, 0, 3));
  EXPECT_THAT(y[1].shape(), ElementsAre(1, 2, 3));
  EXPECT_THAT(y[1].toList<float>(), ElementsAre(4, 2, 3, 1, 0, 3));

  y = op::split(x, 1, 1);
  EXPECT_TRUE(y.size() == 2);
  EXPECT_THAT(y[0].shape(), ElementsAre(2, 1, 3));
  EXPECT_THAT(y[0].toList<float>(), ElementsAre(4, 2, 3, 4, 2, 3));
  EXPECT_THAT(y[1].shape(), ElementsAre(2, 1, 3));
  EXPECT_THAT(y[1].toList<float>(), ElementsAre(1, 0, 3, 1, 0, 3));

  y = op::split(x, 1, 2);
  EXPECT_TRUE(y.size() == 3);
  EXPECT_THAT(y[0].shape(), ElementsAre(2, 2, 1));
  EXPECT_THAT(y[0].toList<float>(), ElementsAre(4, 1, 4, 1));
  EXPECT_THAT(y[1].shape(), ElementsAre(2, 2, 1));
  EXPECT_THAT(y[1].toList<float>(), ElementsAre(2, 0, 2, 0));
  EXPECT_THAT(y[2].shape(), ElementsAre(2, 2, 1));
  EXPECT_THAT(y[2].toList<float>(), ElementsAre(3, 3, 3, 3));
}

TEST(TEST_Operation, basic_concat) {
  Tensor a(Array2d<float>{{1, 2}, {3, 4}});
  Tensor b(Array2d<float>{{5, 6}, {7, 8}});
  auto y = op::concat(ArrayView{a, b}, 0);
  EXPECT_THAT(y.shape(), ElementsAre(4, 2));
  EXPECT_THAT(y.toList<float>(), ElementsAre(1, 2, 3, 4, 5, 6, 7, 8));

  y = op::concat(ArrayView{a, b}, 1);
  EXPECT_THAT(y.shape(), ElementsAre(2, 4));
  EXPECT_THAT(y.toList<float>(), ElementsAre(1, 2, 5, 6, 3, 4, 7, 8));
}

TEST(TEST_Operation, basic_stack) {
  Tensor a(Array1d<float>{1, 2, 3});
  Tensor b(Array1d<float>{4, 5, 6});
  Tensor c(Array1d<float>{7, 8, 9});

  auto y = op::stack(ArrayView{a, b}, 0);
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toList<float>(), ElementsAre(1, 2, 3, 4, 5, 6));

  y = op::stack(ArrayView{a, b}, -1);
  EXPECT_THAT(y.shape(), ElementsAre(3, 2));
  EXPECT_THAT(y.toList<float>(), ElementsAre(1, 4, 2, 5, 3, 6));

  y = op::stack(ArrayView{a, b}, 1);
  EXPECT_THAT(y.shape(), ElementsAre(3, 2));
  EXPECT_THAT(y.toList<float>(), ElementsAre(1, 4, 2, 5, 3, 6));

  y = op::stack(ArrayView{a, b, c}, 1);
  EXPECT_THAT(y.shape(), ElementsAre(3, 3));
  EXPECT_THAT(y.toList<float>(), ElementsAre(1, 4, 7, 2, 5, 8, 3, 6, 9));

  Tensor t1(Array2d<float>{{1, 2}, {3, 4}});
  Tensor t2(Array2d<float>{{5, 6}, {7, 8}});

  y = op::stack(ArrayView{t1, t2}, 0);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2, 2));
  EXPECT_THAT(y.toList<float>(), ElementsAre(1, 2, 3, 4, 5, 6, 7, 8));

  y = op::stack(ArrayView{t1, t2}, 1);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2, 2));
  EXPECT_THAT(y.toList<float>(), ElementsAre(1, 2, 5, 6, 3, 4, 7, 8));
}

TEST(TEST_Operation, basic_vstack) {
  Tensor a(Array1d<float>{1, 2, 3});
  Tensor b(Array1d<float>{4, 5, 6});
  auto y = op::vstack(ArrayView{a, b});
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toList<float>(), ElementsAre(1, 2, 3, 4, 5, 6));

  a = Tensor(Array2d<float>{{1}, {2}, {3}});
  b = Tensor(Array2d<float>{{4}, {5}, {6}});
  y = op::vstack(ArrayView{a, b});
  EXPECT_THAT(y.shape(), ElementsAre(6, 1));
  EXPECT_THAT(y.toList<float>(), ElementsAre(1, 2, 3, 4, 5, 6));
}

TEST(TEST_Operation, basic_hstack) {
  Tensor a(Array1d<float>{1, 2, 3});
  Tensor b(Array1d<float>{4, 5, 6});
  auto y = op::hstack(ArrayView{a, b});
  EXPECT_THAT(y.shape(), ElementsAre(6));
  EXPECT_THAT(y.toList<float>(), ElementsAre(1, 2, 3, 4, 5, 6));

  a = Tensor(Array2d<float>{{1}, {2}, {3}});
  b = Tensor(Array2d<float>{{4}, {5}, {6}});
  y = op::hstack(ArrayView{a, b});
  EXPECT_THAT(y.shape(), ElementsAre(3, 2));
  EXPECT_THAT(y.toList<float>(), ElementsAre(1, 4, 2, 5, 3, 6));
}

TEST(TEST_Operation, math_dot) {
  Array1d<float> d1 = {1, 2, 3};
  Array1d<float> d2 = {4, 5, 6};
  auto y = op::dot(Tensor(d1), Tensor(d2));
  EXPECT_TRUE(y.dim() == 0);
  EXPECT_THAT(y.toList<float>(), ElementsAre(32));

  y = op::dot(Tensor(d1), Tensor(d1));
  EXPECT_TRUE(y.dim() == 0);
  EXPECT_THAT(y.toList<float>(), ElementsAre(14));
}

TEST(TEST_Operation, basic_im2col_col2im) {
  auto input = Tensor(Array2d<float>{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}});
  input.reshape({1, 1, 4, 4});
  auto col = op::im2col(input, 2, 2, 0);
  EXPECT_THAT(col.shape(), ElementsAre(4, 4));
  EXPECT_THAT(col.toList<float>(), ElementsAre(1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16));

  auto r = op::col2im(col, input.shape(), 2, 2, 0);
  EXPECT_EQ(r.shape(), input.shape());
  EXPECT_EQ(r.toList<float>(), input.toList<float>());

  col = op::im2col(input, 2, 3, 0);
  EXPECT_THAT(col.shape(), ElementsAre(1, 4));
  EXPECT_THAT(col.toList<float>(), ElementsAre(1, 2, 5, 6));

  r = op::col2im(col, input.shape(), 2, 3, 0);
  EXPECT_EQ(r.shape(), input.shape());
  EXPECT_THAT(r.toList<float>(), ElementsAre(1, 2, 0, 0, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));

  col = op::im2col(input, 3, 2, 0);
  EXPECT_THAT(col.shape(), ElementsAre(1, 9));
  EXPECT_THAT(col.toList<float>(), ElementsAre(1, 2, 3, 5, 6, 7, 9, 10, 11));

  r = op::col2im(col, input.shape(), 3, 2, 0);
  EXPECT_EQ(r.shape(), input.shape());
  EXPECT_THAT(r.toList<float>(), ElementsAre(1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 0, 0, 0, 0));
}

TEST(TEST_Operation, math_matmul) {
  Array2d<float> d1 = {{1, 2}, {3, 4}};
  Array2d<float> d2 = {{2, 3}, {4, 5}};
  auto y = op::matmul(Tensor(d1), Tensor(d2));
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toList<float>(), ElementsAre(10, 13, 22, 29));

  Array2d<float> d3 = {{1, 2, 3}, {4, 5, 6}};
  Array2d<float> d4 = {{2, 3}, {4, 5}, {6, 7}};
  y = op::matmul(Tensor(d3), Tensor(d4));
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toList<float>(), ElementsAre(28, 34, 64, 79));

  Array2d<float> d5 = {{1, 0}, {0, 1}};
  Array1d<float> d6 = {1, 2};
  y = op::matmul(Tensor(d5), Tensor(d6));
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toList<float>(), ElementsAre(1, 2));

  y = op::matmul(Tensor(d6), Tensor(d5));
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toList<float>(), ElementsAre(1, 2));

  Array1d<float> d7 = {2};
  y = op::matmul(Tensor(d7), Tensor(d7));
  EXPECT_TRUE(y.dim() == 0);
  EXPECT_THAT(y.toList<float>(), ElementsAre(4));

  // broadcast
  auto a = Tensor::arange<float>(0, 2 * 2 * 4);
  a.reshape({2, 2, 4});
  auto b = Tensor::arange<float>(0, 2 * 2 * 4);
  b.reshape({1, 2, 4, 2});
  auto c = Tensor::arange<float>(0, 1 * 2 * 4);
  c.reshape({1, 4, 2});
  auto d = op::matmul(a, b);
  auto e = op::matmul(a, c);

  EXPECT_THAT(d.shape(), ElementsAre(1, 2, 2, 2));
  EXPECT_THAT(d.toList<float>(), ElementsAre(28, 34, 76, 98, 428, 466, 604, 658));

  EXPECT_THAT(e.shape(), ElementsAre(2, 2, 2));
  EXPECT_THAT(e.toList<float>(), ElementsAre(28, 34, 76, 98, 124, 162, 172, 226));
}

TEST(TEST_Operation, math_matmulTrans) {
  Array2d<float> d1 = {{1, 2}, {3, 4}};
  Array2d<float> d2 = {{2, 3}, {4, 5}};
  auto y = op::matmulTrans(Tensor(d1), Tensor(d2), false, true);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toList<float>(), ElementsAre(8, 14, 18, 32));

  Array2d<float> d3 = {{1, 2, 3}, {4, 5, 6}};
  Array2d<float> d4 = {{2, 4, 6}, {3, 5, 7}};
  y = op::matmulTrans(Tensor(d3), Tensor(d4), false, true);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toList<float>(), ElementsAre(28, 34, 64, 79));
}
