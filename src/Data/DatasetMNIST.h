/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Dataset.h"
#include "Transform.h"

namespace tinytorch::data {

class DatasetMNIST : public Dataset {
 public:
  enum MnistDataType {
    TRAIN,
    TEST,
  };
  DatasetMNIST(const std::string& dir, MnistDataType type, const std::shared_ptr<transforms::Transform>& transform);

  size_t size() const override { return size_; }

  std::vector<Tensor> getItem(size_t idx) override;

 private:
  static int32_t toInt32(const char* p);
  void loadImages(const std::string& path);
  void loadLabels(const std::string& path);

  std::vector<std::vector<float>> images_;
  std::vector<int64_t> labels_;
  int32_t height_ = 0;
  int32_t width_ = 0;
  size_t size_ = 0;

  std::shared_ptr<transforms::Transform> transform_;
};

}  // namespace tinytorch::data
