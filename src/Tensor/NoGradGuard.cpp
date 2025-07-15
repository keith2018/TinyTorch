/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "NoGradGuard.h"

namespace tinytorch {

thread_local bool NoGradGuard::gradEnabled = true;

}  // namespace tinytorch