/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "test.h"

void checkFloatVectorNear(const std::vector<float>& v1,
                          const std::vector<float>& v2) {
  EXPECT_EQ(v1.size(), v2.size());
  for (size_t i = 0; i < v1.size(); i++) {
    EXPECT_FLOAT_NEAR(v1[i], v2[i]);
  }
}
