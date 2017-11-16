#include "core/base/executor.hpp"


#include <cstdlib>
#include <cstring>


#include "core/base/exception.hpp"
#include "core/base/exception_helpers.hpp"


namespace gko {


void CpuExecutor::free(void *ptr) const noexcept { std::free(ptr); }


std::shared_ptr<CpuExecutor> CpuExecutor::get_master() noexcept
{
    return this->shared_from_this();
}


std::shared_ptr<const CpuExecutor> CpuExecutor::get_master() const noexcept
{
    return this->shared_from_this();
}


void *CpuExecutor::raw_alloc(size_type num_bytes) const
{
    return ENSURE_ALLOCATED(std::malloc(num_bytes), "CPU", num_bytes);
}


void CpuExecutor::raw_copy_to(const CpuExecutor *, size_type num_bytes,
                              const void *src_ptr, void *dest_ptr) const
{
    std::memcpy(dest_ptr, src_ptr, num_bytes);
}


std::shared_ptr<CpuExecutor> GpuExecutor::get_master() noexcept
{
    return master_;
}


std::shared_ptr<const CpuExecutor> GpuExecutor::get_master() const noexcept
{
    return master_;
}


}  // namespace gko
