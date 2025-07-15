/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <optional>

#include "Optimizer.h"

namespace tinytorch::optim::lr_scheduler {

class Scheduler {
 public:
  virtual ~Scheduler() = default;
  explicit Scheduler(Optimizer &optimizer, int64_t lastEpoch = -1)
      : optimizer_(optimizer), lastEpoch_(lastEpoch), stepCount_(0), lastLr_(0) {
    initialStep();
  }

  void step(std::optional<int64_t> epoch = std::nullopt) {
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
  int64_t getLastEpoch() const { return lastEpoch_; }

 protected:
  void initialStep() {
    stepCount_ = 0;
    step();
  }
  virtual float getLr() { return optimizer_.getLr(); }

  Optimizer &optimizer_;
  int64_t lastEpoch_;
  int64_t stepCount_;
  float lastLr_;
};

class StepLR : public Scheduler {
 public:
  StepLR(Optimizer &optimizer, int64_t stepSize, float gamma = 0.1, int64_t lastEpoch = -1)
      : Scheduler(optimizer, lastEpoch), stepSize_(stepSize), gamma_(gamma) {}

  float getLr() override;

 private:
  int64_t stepSize_;
  float gamma_;
};

}  // namespace tinytorch::optim::lr_scheduler
