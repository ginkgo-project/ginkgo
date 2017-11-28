#include "core/base/executor.hpp"


#include "core/base/exception.hpp"
#include "core/base/exception_helpers.hpp"


namespace gko {


void Operation::run(const CpuExecutor *executor) const NOT_IMPLEMENTED;


void Operation::run(const GpuExecutor *executor) const NOT_IMPLEMENTED;


void Operation::run(const ReferenceExecutor *executor) const
{
    this->run(static_cast<const CpuExecutor *>(executor));
}


}  // namespace gko
