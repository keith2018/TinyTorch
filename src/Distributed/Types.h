/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

namespace tinytorch::distributed {

enum ReduceOpType : uint8_t {
  SUM = 1,
  AVG = 2,
  PRODUCT = 3,
  MIN = 4,
  MAX = 5,
  UNUSED = 6,
};

struct BroadcastOptions {
  int rootRank = 0;
};

struct AllReduceOptions {
  ReduceOpType reduceOp = SUM;
};

struct ReduceOptions {
  ReduceOpType reduceOp = SUM;
  int rootRank = 0;
};

struct AllGatherOptions {};

struct ReduceScatterOptions {
  ReduceOpType reduceOp = SUM;
};

struct BarrierOptions {
  std::vector<int64_t> deviceIds;
  std::optional<Device> device;
};

}  // namespace tinytorch::distributed
