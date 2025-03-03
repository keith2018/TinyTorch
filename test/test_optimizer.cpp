/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include <Torch.h>

#include "test.h"

using namespace TinyTorch;

TEST(TEST_Optimizer, SGD) {
  auto x = Tensor({0.5, 1.5, -0.5}, true);
  x.setGrad(Tensor({0.1, -0.2, 0.3}));
  auto optimizer = optim::SGD({&x}, 0.1, 0.6, 0.7, 0.8);
  optimizer.step();
  EXPECT_FLOAT_VEC_NEAR(x.data().toList(), {0.4500, 1.4000, -0.4900});
  optimizer.step();
  EXPECT_FLOAT_VEC_NEAR(x.data().toList(), {0.4062, 1.3124, -0.4812});
}

TEST(TEST_Optimizer, Adagrad) {
  auto x = Tensor({0.5, 1.5, -0.5}, true);
  x.setGrad(Tensor({0.1, -0.2, 0.3}));
  auto optimizer = optim::Adagrad({&x}, 0.1, 0.9, 0.5, 0.1);
  optimizer.step();
  EXPECT_FLOAT_VEC_NEAR(x.data().toList(), {0.4258, 1.4133, -0.5156});
  optimizer.step();
  EXPECT_FLOAT_VEC_NEAR(x.data().toList(), {0.3967, 1.3805, -0.5225});
}

TEST(TEST_Optimizer, RMSprop) {
  auto x = Tensor({0.5, 1.5, -0.5}, true);
  x.setGrad(Tensor({0.1, -0.2, 0.3}));
  auto optimizer = optim::RMSprop({&x}, 0.1, 0.9, 0.001, 0.1, 0.9);
  optimizer.step();
  EXPECT_FLOAT_VEC_NEAR(x.data().toList(), {0.1903, 1.7974, -0.8123});
  optimizer.step();
  EXPECT_FLOAT_VEC_NEAR(x.data().toList(), {-0.2879, 2.1821, -1.3057});
}

TEST(TEST_Optimizer, AdaDelta) {
  auto x = Tensor({0.5, 1.5, -0.5}, true);
  x.setGrad(Tensor({0.1, -0.2, 0.3}));
  auto optimizer = optim::AdaDelta({&x}, 0.1, 0.9, 0.001, 0.1);
  optimizer.step();
  EXPECT_FLOAT_VEC_NEAR(x.data().toList(), {0.4917, 1.5045, -0.5093});
  optimizer.step();
  EXPECT_FLOAT_VEC_NEAR(x.data().toList(), {0.4832, 1.5089, -0.5188});
}

TEST(TEST_Optimizer, Adam) {
  auto x = Tensor({0.5, 1.5, -0.5}, true);
  x.setGrad(Tensor({0.1, -0.2, 0.3}));
  auto optimizer = optim::Adam({&x}, 0.1, {0.9, 0.8}, 0.001, 0.1);
  optimizer.step();
  EXPECT_FLOAT_VEC_NEAR(x.data().toList(), {0.4007, 1.5980, -0.5996});
  optimizer.step();
  EXPECT_FLOAT_VEC_NEAR(x.data().toList(), {0.3012, 1.6959, -0.6993});
}

TEST(TEST_Optimizer, AdamW) {
  auto x = Tensor({0.5, 1.5, -0.5}, true);
  x.setGrad(Tensor({0.1, -0.2, 0.3}));
  auto optimizer = optim::AdamW({&x}, 0.1, {0.9, 0.8}, 0.001, 0.1);
  optimizer.step();
  EXPECT_FLOAT_VEC_NEAR(x.data().toList(), {0.3960, 1.5845, -0.5947});
  optimizer.step();
  EXPECT_FLOAT_VEC_NEAR(x.data().toList(), {0.2930, 1.6682, -0.6884});
}

TEST(TEST_Scheduler, StepLR) {
  auto layer = nn::Linear(2, 3);
  auto optimizer = optim::SGD(layer.parameters(), 0.05);
  auto scheduler = optim::lr_scheduler::StepLR(optimizer, 3, 0.5);
  std::vector<float> lrList;
  for (int epoch = 0; epoch < 10; epoch++) {
    optimizer.step();
    scheduler.step();
    lrList.push_back(scheduler.getLastLr());
  }
  EXPECT_THAT(lrList, ElementsAre(0.05, 0.05, 0.025, 0.025, 0.025, 0.0125,
                                  0.0125, 0.0125, 0.00625, 0.00625));
}
