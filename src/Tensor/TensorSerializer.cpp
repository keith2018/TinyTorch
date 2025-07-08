/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "TensorSerializer.h"

#include <fstream>

#include "Module.h"

namespace tinytorch {

void save(const Tensor& tensor, std::ofstream& ofs) {
  auto& t = tensor.getImpl();

  // shape
  int64_t ndim = t.dim();
  ofs.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
  for (int64_t i = 0; i < ndim; i++) {
    int64_t dim_size = t.shape(i);
    ofs.write(reinterpret_cast<const char*>(&dim_size), sizeof(dim_size));
  }

  // options
  auto dtype = t.dtype();
  auto device = t.device();
  auto requiresGrad = t.requiresGrad();
  auto pinnedMemory = t.pinnedMemory();

  ofs.write(reinterpret_cast<const char*>(&dtype), sizeof(dtype));
  ofs.write(reinterpret_cast<const char*>(&device.type), sizeof(device.type));
  ofs.write(reinterpret_cast<const char*>(&device.index), sizeof(device.index));
  ofs.write(reinterpret_cast<const char*>(&requiresGrad), sizeof(requiresGrad));
  ofs.write(reinterpret_cast<const char*>(&pinnedMemory), sizeof(pinnedMemory));

  // numel
  int64_t numel = t.numel();
  ofs.write(reinterpret_cast<const char*>(&numel), sizeof(numel));

  // data
  size_t elemSize = dtypeSize(dtype);
  if (device.isCpu()) {
    ofs.write(reinterpret_cast<const char*>(t.dataPtr()), numel * elemSize);
  } else {
    std::vector<uint8_t> hostData(numel * elemSize);
    Storage::copyOnDevice(hostData.data(), DeviceType::CPU, t.dataPtr(), device, numel * elemSize);
    ofs.write(reinterpret_cast<const char*>(hostData.data()), numel * elemSize);
  }
}

void load(Tensor& tensor, std::ifstream& ifs) {
  auto& t = tensor.getImpl();

  // shape
  int64_t ndim;
  ifs.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
  std::vector<int64_t> shape(ndim);
  for (int64_t i = 0; i < ndim; i++) {
    ifs.read(reinterpret_cast<char*>(&shape[i]), sizeof(shape[i]));
  }

  // options
  DType dtype;
  DeviceType deviceType;
  DeviceIndex deviceIndex;
  bool requiresGrad;
  bool pinnedMemory;

  ifs.read(reinterpret_cast<char*>(&dtype), sizeof(dtype));
  ifs.read(reinterpret_cast<char*>(&deviceType), sizeof(deviceType));
  ifs.read(reinterpret_cast<char*>(&deviceIndex), sizeof(deviceIndex));
  ifs.read(reinterpret_cast<char*>(&requiresGrad), sizeof(requiresGrad));
  ifs.read(reinterpret_cast<char*>(&pinnedMemory), sizeof(pinnedMemory));

  Device device(deviceType, deviceIndex);
  Options options;
  options.dtype_ = dtype;
  options.device_ = device;
  options.requiresGrad_ = requiresGrad;
  options.pinnedMemory_ = pinnedMemory;

  t = TensorImpl(shape, options);

  // numel
  int64_t numel;
  ifs.read(reinterpret_cast<char*>(&numel), sizeof(numel));
  ASSERT(t.numel() == numel);

  // data
  size_t elemSize = dtypeSize(dtype);
  if (device.isCpu()) {
    ifs.read(reinterpret_cast<char*>(t.dataPtr()), numel * elemSize);
  } else {
    std::vector<uint8_t> hostData(numel * elemSize);
    ifs.read(reinterpret_cast<char*>(hostData.data()), numel * elemSize);
    Storage::copyOnDevice(t.dataPtr(), device, hostData.data(), DeviceType::CPU, numel * elemSize);
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
