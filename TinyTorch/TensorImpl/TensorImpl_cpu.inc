/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

namespace TinyTorch {

struct OpCpuAdd {
  float operator()(const float a, const float b) const { return a + b; }
};

struct OpCpuSub {
  float operator()(const float a, const float b) const { return a - b; }
};

struct OpCpuMul {
  float operator()(const float a, const float b) const { return a * b; }
};

struct OpCpuDiv {
  float operator()(const float a, const float b) const { return a / b; }
};

struct OpCpuPow {
  float operator()(const float a, const float b) const {
    return std::pow(a, b);
  }
};

struct OpCpuEq {
  float operator()(const float a, const float b) const {
    return a == b ? 1.f : 0.f;
  }
};

struct OpCpuNe {
  float operator()(const float a, const float b) const {
    return a != b ? 1.f : 0.f;
  }
};

struct OpCpuLt {
  float operator()(const float a, const float b) const {
    return a < b ? 1.f : 0.f;
  }
};

struct OpCpuLe {
  float operator()(const float a, const float b) const {
    return a <= b ? 1.f : 0.f;
  }
};

struct OpCpuGt {
  float operator()(const float a, const float b) const {
    return a > b ? 1.f : 0.f;
  }
};

struct OpCpuGe {
  float operator()(const float a, const float b) const {
    return a >= b ? 1.f : 0.f;
  }
};

struct OpCpuMax {
  float operator()(const float a, const float b) const {
    return std::max(a, b);
  }
};

struct OpCpuMin {
  float operator()(const float a, const float b) const {
    return std::min(a, b);
  }
};

struct OpCpuSin_ {
  void operator()(float& a) const { a = std::sin(a); }
};

struct OpCpuCos_ {
  void operator()(float& a) const { a = std::cos(a); }
};

struct OpCpuSqrt_ {
  void operator()(float& a) const { a = std::sqrt(a); }
};

struct OpCpuTanh_ {
  void operator()(float& a) const { a = std::tanh(a); }
};

struct OpCpuExp_ {
  void operator()(float& a) const { a = std::exp(a); }
};

struct OpCpuLog_ {
  void operator()(float& a) const { a = std::log(a); }
};

struct OpCpuSin {
  float operator()(const float a) const { return std::sin(a); }
};

struct OpCpuCos {
  float operator()(const float a) const { return std::cos(a); }
};

struct OpCpuSqrt {
  float operator()(const float a) const { return std::sqrt(a); }
};

struct OpCpuTanh {
  float operator()(const float a) const { return std::tanh(a); }
};

struct OpCpuExp {
  float operator()(const float a) const { return std::exp(a); }
};

struct OpCpuLog {
  float operator()(const float a) const { return std::log(a); }
};

struct OpCpuReduceMax {
  float operator()(const float a, const float b) const {
    return std::max(a, b);
  }

  static float defaultVal() { return -std::numeric_limits<float>::max(); }
};

struct OpCpuReduceMin {
  float operator()(const float a, const float b) const {
    return std::min(a, b);
  }

  static float defaultVal() { return std::numeric_limits<float>::max(); }
};

struct OpCpuReduceSum {
  float operator()(const float a, const float b) const { return a + b; }

  static float defaultVal() { return 0.f; }
};

}  // namespace TinyTorch
