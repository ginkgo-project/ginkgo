#include "core/base/executor.hpp"


#include "core/base/exception_helpers.hpp"


namespace gko {


void CpuExecutor::raw_copy_to(const GpuExecutor *, size_type num_bytes,
                              const void *src_ptr, void *dest_ptr) const
    NOT_COMPILED(gpu);


void GpuExecutor::free(void *ptr) const noexcept
{
    // Free must never fail, as it can be called in destructors.
    // If the nvidia module was not compiled, the library couldn't have
    // allocated the memory, so there is no need to deallocate it.
}


void *GpuExecutor::raw_alloc(size_type num_bytes) const NOT_COMPILED(nvidia);


void GpuExecutor::raw_copy_to(const CpuExecutor *, size_type num_bytes,
                              const void *src_ptr, void *dest_ptr) const
    NOT_COMPILED(gpu);


void GpuExecutor::raw_copy_to(const GpuExecutor *, size_type num_bytes,
                              const void *src_ptr, void *dest_ptr) const
    NOT_COMPILED(gpu);


void GpuExecutor::synchronize() const NOT_COMPILED(gpu);


std::string GpuExecutor::get_error(int64 error_code)
{
    return "ginkgo CUDA module is not compiled";
}


int GpuExecutor::get_num_devices() { return 0; }

}  // namespace gko
