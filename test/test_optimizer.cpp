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

  auto optimizer = optim::SGD({&x}, 0.1);

  optimizer.step();
  EXPECT_FLOAT_VEC_NEAR(x.data().toArray(), {0.4900, 1.5200, -0.5300});

  optimizer.step();
  EXPECT_FLOAT_VEC_NEAR(x.data().toArray(), {0.4800, 1.5400, -0.5600});
}

TEST(TEST_Optimizer, RMSprop) {
  auto x = Tensor({0.5, 1.5, -0.5}, true);
  x.setGrad(Tensor({0.1, -0.2, 0.3}));

  auto optimizer = optim::RMSprop({&x}, 0.1, 0.8);

  optimizer.step();
  EXPECT_FLOAT_VEC_NEAR(x.data().toArray(), {0.2764, 1.7236, -0.7236});

  optimizer.step();
  EXPECT_FLOAT_VEC_NEAR(x.data().toArray(), {0.109726, 1.890273, -0.890273});
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
