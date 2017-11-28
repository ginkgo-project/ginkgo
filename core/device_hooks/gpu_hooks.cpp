#include "core/base/exception.hpp"
#include "core/base/exception_helpers.hpp"
#include "core/base/executor.hpp"
#include "core/matrix/dense_kernels.hpp"


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


namespace kernels {


namespace gpu {


template <typename ValueType>
GKO_DECLARE_GEMM_KERNEL(ValueType)
NOT_COMPILED(gpu);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GEMM_KERNEL);

template <typename ValueType>
GKO_DECLARE_SCAL_KERNEL(ValueType)
NOT_COMPILED(gpu);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_SCAL_KERNEL);

template <typename ValueType>
GKO_DECLARE_AXPY_KERNEL(ValueType)
NOT_COMPILED(gpu);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_AXPY_KERNEL);

template <typename ValueType>
GKO_DECLARE_DOT_KERNEL(ValueType)
NOT_COMPILED(gpu);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DOT_KERNEL);


}  // namespace gpu


}  // namespace kernels


}  // namespace gko
