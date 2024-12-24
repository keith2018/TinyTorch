/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Torch.h"

#include "TensorImpl.h"

namespace TinyTorch {

void manualSeed(unsigned int seed) { RandomGenerator::setSeed(seed); }

template <typename T>
static std::string printArray(const T* vec, int32_t size, bool restrict) {
  std::ostringstream oss;
  oss << "(";
  if (!restrict || size <= 16) {
    for (size_t i = 0; i < size; ++i) {
      oss << vec[i];
      if (i != size - 1) {
        oss << ", ";
      }
    }
  } else {
    for (size_t i = 0; i < 8; ++i) {
      oss << vec[i];
      if (i != 7) {
        oss << ", ";
      }
    }
    oss << ", ... , ";
    for (size_t i = size - 8; i < size; ++i) {
      oss << vec[i];
      if (i != size - 1) {
        oss << ", ";
      }
    }
  }
  oss << ")";
  return oss.str();
}

void print(const Tensor& tensor) {
  std::ostringstream oss;
  oss << "Tensor { shape: "
      << printArray(&tensor.shape()[0], tensor.dim(), false);
  oss << ", requiresGrad: " << (tensor.isRequiresGrad() ? "true" : "false");
  if (tensor.isRequiresGrad()) {
    oss << ", gradFunc: " << tensor.getGradFunc()->typeString();
  }
  oss << ", data: " << printArray(&tensor.data()[0], tensor.size(), true);
  LOGD("%s", oss.str().c_str());
}

void save(const Tensor& tensor, std::ofstream& ofs) {
  auto& t = tensor.data();
  // dimCount
  int32_t dimCount = t.dim();
  ofs.write((const char*)(&dimCount), sizeof(dimCount));

  // elemCount
  int32_t elemCount = t.size();
  ofs.write((const char*)(&elemCount), sizeof(elemCount));

  // shape, strides, data
  ofs.write((const char*)(t.shape().data()),
            std::streamsize(dimCount * sizeof(int32_t)));
  ofs.write((const char*)(t.strides().data()),
            std::streamsize(dimCount * sizeof(int32_t)));
  ofs.write((const char*)(&t[0]), std::streamsize(elemCount * sizeof(float)));
}

void load(Tensor& tensor, std::ifstream& ifs) {
  auto& t = tensor.data();

  // dimCount
  int32_t dimCount;
  ifs.read((char*)(&dimCount), sizeof(dimCount));

  if (dimCount != t.dim()) {
    LOGE("load failed: expect dimCount %d but got %d", t.dim(), dimCount);
    return;
  }

  // elemCount
  int32_t elemCount;
  ifs.read((char*)(&elemCount), sizeof(elemCount));
  if (elemCount != t.size()) {
    LOGE("load failed: expect elemCount %d but got %d", t.size(), elemCount);
    return;
  }

  // shape, strides, data
  ifs.read((char*)(t.shape().data()),
           std::streamsize(dimCount * sizeof(int32_t)));
  ifs.read((char*)(t.strides().data()),
           std::streamsize(dimCount * sizeof(int32_t)));
  ifs.read((char*)(&t[0]), std::streamsize(elemCount * sizeof(float)));
}

void save(nn::Module& model, const char* path) {
  std::ofstream outFile(path, std::ios::binary);
  if (!outFile) {
    LOGE("Failed to open file for writing: %s", path);
    return;
  }

  auto params = model.parameters();
  for (auto& param : params) {
    save(*param, outFile);
  }
}

void load(nn::Module& model, const char* path) {
  std::ifstream inFile(path, std::ios::binary);
  if (!inFile) {
    LOGE("Failed to open file for reading: %s", path);
    return;
  }

  auto params = model.parameters();
  for (auto& param : params) {
    load(*param, inFile);
  }
}

}  // namespace TinyTorch
