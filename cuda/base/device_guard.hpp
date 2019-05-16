
#include <cuda_runtime.h>


#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {


class device_guard {
public:
    device_guard(int device_id)
    {
        GKO_ASSERT_NO_CUDA_ERRORS(cudaGetDevice(&original_device_id));
        GKO_ASSERT_NO_CUDA_ERRORS(cudaSetDevice(device_id));
    }

    ~device_guard() noexcept(false)
    {
        /* Ignore the error during stack unwinding for this call */
        if (std::uncaught_exception()) {
            cudaSetDevice(original_device_id);
        } else {
            GKO_ASSERT_NO_CUDA_ERRORS(cudaSetDevice(original_device_id));
        }
    }

private:
    int original_device_id{};
};


}  // namespace gko
