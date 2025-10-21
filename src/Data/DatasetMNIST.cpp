/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "DatasetMNIST.h"

#include <fstream>
#include <memory>

#include "Utils/Logger.h"

namespace tinytorch::data {

constexpr auto MNIST_TRAIN_IMAGES = "train-images-idx3-ubyte";
constexpr auto MNIST_TRAIN_LABELS = "train-labels-idx1-ubyte";
constexpr auto MNIST_TEST_IMAGES = "t10k-images-idx3-ubyte";
constexpr auto MNIST_TEST_LABELS = "t10k-labels-idx1-ubyte";

DatasetMNIST::DatasetMNIST(const std::string& dir, MnistDataType type,
                           const std::shared_ptr<transforms::Transform>& transform)
    : transform_(transform) {
  auto imagePath = dir + (type == TRAIN ? MNIST_TRAIN_IMAGES : MNIST_TEST_IMAGES);
  auto labelPath = dir + (type == TRAIN ? MNIST_TRAIN_LABELS : MNIST_TEST_LABELS);
  loadImages(imagePath);
  loadLabels(labelPath);
  size_ = std::min(images_.size(), labels_.size());
}

std::vector<Tensor> DatasetMNIST::getItem(size_t idx) {
  auto img = Tensor(images_[idx]);
  img.reshape_({1, height_, width_});

  auto label = Tensor::scalar(labels_[idx], options::dtype(DType::Int64));
  if (transform_) {
    img = transform_->process(img);
  }
  return {img, label};
}

int32_t DatasetMNIST::toInt32(const char* p) {
  return ((p[0] & 0xff) << 24) | ((p[1] & 0xff) << 16) | ((p[2] & 0xff) << 8) | ((p[3] & 0xff) << 0);
}

void DatasetMNIST::loadImages(const std::string& path) {
  std::ifstream ifs(path.c_str(), std::ios::in | std::ios::binary);
  if (!ifs.is_open()) {
    LOGE("failed to load images from %s", path.c_str());
    return;
  }
  char p[4];

  ifs.read(p, 4);
  auto magicNumber = toInt32(p);
  ASSERT(magicNumber == 0x803);

  ifs.read(p, 4);
  auto size = toInt32(p);
  images_.resize(size);

  ifs.read(p, 4);
  height_ = toInt32(p);

  ifs.read(p, 4);
  width_ = toInt32(p);

  char* tmp = new char[height_ * width_];
  for (int32_t i = 0; i < size; ++i) {
    images_[i].resize(height_ * width_);
    ifs.read(tmp, height_ * width_);
    float* dataPtr = &images_[i][0];
    for (int32_t j = 0; j < height_ * width_; ++j) {
      auto d = static_cast<uint8_t>(tmp[j]);
      dataPtr[j] = static_cast<float>(d) / 255.0f;
    }
  }
  delete[] tmp;

  ifs.close();
}

void DatasetMNIST::loadLabels(const std::string& path) {
  std::ifstream ifs(path.c_str(), std::ios::in | std::ios::binary);
  if (!ifs.is_open()) {
    LOGE("failed to load labels from %s", path.c_str());
    return;
  }
  char p[4];

  ifs.read(p, 4);
  auto magicNumber = toInt32(p);
  ASSERT(magicNumber == 0x801);

  ifs.read(p, 4);
  auto size = toInt32(p);
  labels_.resize(size);
  for (int32_t i = 0; i < size; ++i) {
    ifs.read(p, 1);
    labels_[i] = static_cast<unsigned char>(p[0]);
  }

  ifs.close();
}

}  // namespace tinytorch::data