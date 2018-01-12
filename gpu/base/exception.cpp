#include "core/base/exception.hpp"


#include <cuda_runtime.h>


namespace gko {


std::string CudaError::get_error(int64 error_code)
{
    std::string name = cudaGetErrorName(static_cast<cudaError>(error_code));
    std::string message =
        cudaGetErrorString(static_cast<cudaError>(error_code));
    return name + ": " + message;
}


}  // namespace gko
