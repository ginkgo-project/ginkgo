#include "core/base/exception.hpp"

#include "cuda_runtime.h"

static std::string get_error(cudaError_t error_code)
{
    std::string name = cudaGetErrorName(error_code);
    std::string message = cudaGetErrorString(error_code);

    return name + ": " + message;
}
