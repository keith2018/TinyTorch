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

void checkFloatVectorNear(const std::vector<float>& v1,
                          const std::vector<float>& v2);

#define FLOAT_ABS_ERROR 1e-4
#define EXPECT_FLOAT_NEAR(v1, v2) EXPECT_NEAR(v1, v2, FLOAT_ABS_ERROR)

#define EXPECT_FLOAT_VEC_NEAR checkFloatVectorNear
