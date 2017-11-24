#include "core/base/executor.hpp"


namespace gko {


std::shared_ptr<CpuExecutor> GpuExecutor::get_master() noexcept
{
    return master_;
}


std::shared_ptr<const CpuExecutor> GpuExecutor::get_master() const noexcept
{
    return master_;
}


}  // namespace gko
