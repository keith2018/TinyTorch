/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Optimizer.h"

namespace TinyTorch::optim::lr_scheduler {

class Scheduler {
 public:
  virtual ~Scheduler() = default;
  explicit Scheduler(Optimizer &optimizer, int32_t lastEpoch = -1)
      : optimizer_(optimizer),
        lastEpoch_(lastEpoch),
        stepCount_(0),
        lastLr_(0) {
    initialStep();
  }

  void step(std::optional<int32_t> epoch = std::nullopt) {
    stepCount_++;
    if (epoch.has_value()) {
      lastEpoch_ = epoch.value();
    } else {
      lastEpoch_++;
    }

    lastLr_ = getLr();
    optimizer_.setLr(lastLr_);
  }

  float getLastLr() const { return lastLr_; }
  int32_t getLastEpoch() const { return lastEpoch_; }

 protected:
  void initialStep() {
    stepCount_ = 0;
    step();
  }
  virtual float getLr() { return optimizer_.getLr(); }

 protected:
  Optimizer &optimizer_;
  int32_t lastEpoch_;
  int32_t stepCount_;
  float lastLr_;
};

class StepLR : public Scheduler {
 public:
  StepLR(Optimizer &optimizer, int32_t stepSize, float gamma = 0.1,
         int32_t lastEpoch = -1)
      : Scheduler(optimizer, lastEpoch), stepSize_(stepSize), gamma_(gamma) {}

  float getLr() override;

 private:
  int32_t stepSize_;
  float gamma_;
};

}  // namespace TinyTorch::optim::lr_scheduler
