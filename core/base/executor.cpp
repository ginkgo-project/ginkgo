#include "core/base/executor.hpp"


namespace gko {


void Operation::run(const ReferenceExecutor *executor) const
{
    this->run(static_cast<const CpuExecutor *>(executor));
}


}  // namespace gko
