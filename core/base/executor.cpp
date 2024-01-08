// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/executor.hpp>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/name_demangling.hpp>


namespace gko {


void Operation::run(std::shared_ptr<const OmpExecutor> executor) const
    GKO_NOT_IMPLEMENTED;


void Operation::run(std::shared_ptr<const CudaExecutor> executor) const
    GKO_NOT_IMPLEMENTED;


void Operation::run(std::shared_ptr<const HipExecutor> executor) const
    GKO_NOT_IMPLEMENTED;


void Operation::run(std::shared_ptr<const DpcppExecutor> executor) const
    GKO_NOT_IMPLEMENTED;


void Operation::run(std::shared_ptr<const ReferenceExecutor> executor) const
{
    this->run(static_cast<std::shared_ptr<const OmpExecutor>>(executor));
}


const char* Operation::get_name() const noexcept
{
    static auto name = name_demangling::get_dynamic_type(*this);
    return name.c_str();
}


}  // namespace gko
