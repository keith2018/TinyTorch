/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <vector>

#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"

using namespace testing;

#define FLOAT_ABS_ERROR 1e-3f

inline AssertionResult VectorNear(const std::vector<float>& v1, const std::vector<float>& v2) {
  auto absError = FLOAT_ABS_ERROR;
  if (v1.size() != v2.size()) {
    return AssertionFailure() << "Vectors have different sizes: " << v1.size() << " vs " << v2.size();
  }
  for (size_t i = 0; i < v1.size(); ++i) {
    if (std::fabs(v1[i] - v2[i]) > absError) {
      return AssertionFailure() << "Vectors differ at index " << i << ": " << v1[i] << " vs " << v2[i];
    }
  }
  return AssertionSuccess();
}

#define EXPECT_FLT_NEAR(v1, v2) EXPECT_NEAR(v1, v2, FLOAT_ABS_ERROR)
