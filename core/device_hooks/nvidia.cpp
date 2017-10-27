#include "core/base/executor.hpp"


namespace msparse {


#define FEATURE_NOT_COMPILED \
    { /* TODO */             \
    }


void CpuExecutor::raw_copy_to(const GpuExecutor *, size_type num_bytes,
                              const void *src_ptr,
                              void *dest_ptr) const FEATURE_NOT_COMPILED;


void GpuExecutor::free(void *ptr) const noexcept FEATURE_NOT_COMPILED;


std::shared_ptr<CpuExecutor> GpuExecutor::get_master() noexcept
{
    return master_;
}


std::shared_ptr<const CpuExecutor> GpuExecutor::get_master() const noexcept
{
    return master_;
}


void *GpuExecutor::raw_alloc(size_type num_bytes) const FEATURE_NOT_COMPILED;


void GpuExecutor::raw_copy_to(const CpuExecutor *, size_type num_bytes,
                              const void *src_ptr,
                              void *dest_ptr) const FEATURE_NOT_COMPILED;


void GpuExecutor::raw_copy_to(const GpuExecutor *, size_type num_bytes,
                              const void *src_ptr,
                              void *dest_ptr) const FEATURE_NOT_COMPILED;
}
