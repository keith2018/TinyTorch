/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Torch.h"

#include "TensorImpl/TensorImpl.h"
#include "TensorImpl/TensorImpl_cpu.h"
#ifdef USE_CUDA
#include "TensorImpl/TensorImpl_cuda.cuh"
#endif

namespace tinytorch {

void manualSeed(unsigned long seed) {
  RandomGeneratorCPU::setSeed(seed);
#ifdef USE_CUDA
  RandomGeneratorCUDA::setSeed(seed);
#endif
}

void setDefaultDevice(Device device) { TensorImpl::setDefaultDevice(device); }

template <typename T>
static std::string printArray(const std::vector<T>& vec, bool full) {
  auto size = vec.size();
  std::ostringstream oss;
  oss << "(";
  if (full || size <= 16) {
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

void print(const TensorImpl& tensor, bool full) {
  std::ostringstream oss;
  oss << "TensorImpl { shape: " << printArray(tensor.shape(), false);
  oss << ", data: " << printArray(tensor.toList(), full);
  oss << " }";
  LOGD("%s", oss.str().c_str());
}

void print(const Tensor& tensor, bool full) {
  std::ostringstream oss;
  oss << "Tensor { shape: " << printArray(tensor.shape(), false);
  oss << ", requiresGrad: " << (tensor.isRequiresGrad() ? "true" : "false");
  if (tensor.isRequiresGrad()) {
    oss << ", gradFunc: " << tensor.getGradFunc()->typeString();
  }
  oss << ", data: " << printArray(tensor.data().toList(), full);
  oss << " }";
  LOGD("%s", oss.str().c_str());
}

void save(const Tensor& tensor, std::ofstream& ofs) {
  auto& t = tensor.data();
  // dimCount
  int32_t dimCount = t.dim();
  ofs.write((const char*)(&dimCount), sizeof(dimCount));

  // elemCount
  int32_t elemCount = t.numel();
  ofs.write((const char*)(&elemCount), sizeof(elemCount));

  // shape, strides, data
  if (dimCount > 0) {
    ofs.write((const char*)(t.shape().data()),
              std::streamsize(dimCount * sizeof(int32_t)));
    ofs.write((const char*)(t.strides().data()),
              std::streamsize(dimCount * sizeof(int32_t)));
  }

  // data
  if (elemCount > 0) {
    auto dataBytesCnt = elemCount * sizeof(float);
    if (t.device() == Device::CPU) {
      ofs.write((const char*)t.data(), std::streamsize(dataBytesCnt));
    } else {
      std::vector<float> hostData(elemCount);
      t.ops()->copyDeviceToHost(hostData.data(), t.data(), dataBytesCnt);
      ofs.write((const char*)hostData.data(), std::streamsize(dataBytesCnt));
    }
  }
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
  if (elemCount != t.numel()) {
    LOGE("load failed: expect elemCount %d but got %d", t.numel(), elemCount);
    return;
  }

  // shape, strides, data
  if (dimCount > 0) {
    ifs.read((char*)(t.shape().data()),
             std::streamsize(dimCount * sizeof(int32_t)));
    ifs.read((char*)(t.strides().data()),
             std::streamsize(dimCount * sizeof(int32_t)));
  }

  // data
  if (elemCount > 0) {
    auto dataBytesCnt = elemCount * sizeof(float);
    if (t.device() == Device::CPU) {
      ifs.read((char*)(t.data()), std::streamsize(dataBytesCnt));
    } else {
      std::vector<float> hostData(elemCount);
      ifs.read((char*)(hostData.data()), std::streamsize(dataBytesCnt));
      t.ops()->copyHostToDevice(t.data(), hostData.data(), dataBytesCnt);
    }
  }
}

void save(nn::Module& model, const char* path) {
  std::ofstream outFile(path, std::ios::binary);
  if (!outFile) {
    LOGE("Failed to open file for writing: %s", path);
    return;
  }

  auto params = model.states();
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

  auto params = model.states();
  for (auto& param : params) {
    load(*param, inFile);
  }
}

}  // namespace tinytorch
