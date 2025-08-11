/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <string>

namespace tinytorch {

struct MMappingResult {
  void* dataPtr = nullptr;
  size_t fileSize = 0;
#ifdef _WIN32
  void* hFile = nullptr;
  void* hMap = nullptr;
#else
  int fd = -1;
#endif
  bool success = false;
};

class MMapUtils {
 public:
  static MMappingResult mapFileForRead(const std::string& path);

  static void unmapFile(MMappingResult& mappingResult);
};

}  // namespace tinytorch