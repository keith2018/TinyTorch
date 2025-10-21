/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <optional>
#include <random>

namespace tinytorch {

class RandomGeneratorCPU {
 public:
  static void setSeed(const unsigned long seed) {
    seed_ = seed;
    randomEngine_.seed(seed);
  }
  static std::default_random_engine& getGenerator() {
    if (!seed_.has_value()) {
      std::random_device r;
      randomEngine_.seed(r());
    }
    return randomEngine_;
  }

 private:
  static std::optional<unsigned long> seed_;
  static std::default_random_engine randomEngine_;
};

class RandomGeneratorCUDA {
 public:
  static void setSeed(unsigned long seed) { seed_ = seed; }
  static unsigned long getSeed() { return seed_; }
  static unsigned long nextSequence() { return sequence_++; }

 private:
  static unsigned long seed_;
  static unsigned long sequence_;
};

void manualSeed(unsigned long seed);

}  // namespace tinytorch
