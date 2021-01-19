/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <ginkgo/core/base/executor.hpp>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {


std::shared_ptr<Executor> MpiExecutor::get_master() noexcept
{
    return this->sub_executor_->get_master();
}


std::shared_ptr<const Executor> MpiExecutor::get_master() const noexcept
{
    return this->sub_executor_->get_master();
}


std::shared_ptr<Executor> MpiExecutor::get_sub_executor() noexcept
{
    return this->sub_executor_;
}


std::shared_ptr<const Executor> MpiExecutor::get_sub_executor() const noexcept
{
    return this->sub_executor_;
}


void MpiExecutor::populate_exec_info(const MachineTopology *mach_topo) {}


void *MpiExecutor::raw_alloc(size_type num_bytes) const
{
    return this->sub_executor_->raw_alloc(num_bytes);
}


void MpiExecutor::raw_free(void *ptr) const noexcept
{
    return this->sub_executor_->raw_free(ptr);
}


void MpiExecutor::raw_copy_to(const MpiExecutor *, size_type num_bytes,
                              const void *src_ptr,
                              void *dest_ptr) const GKO_NOT_IMPLEMENTED;


void MpiExecutor::raw_copy_to(const OmpExecutor *, size_type num_bytes,
                              const void *src_ptr,
                              void *dest_ptr) const GKO_NOT_IMPLEMENTED;


void MpiExecutor::raw_copy_to(const CudaExecutor *, size_type num_bytes,
                              const void *src_ptr,
                              void *dest_ptr) const GKO_NOT_IMPLEMENTED;


void MpiExecutor::raw_copy_to(const HipExecutor *, size_type num_bytes,
                              const void *src_ptr,
                              void *dest_ptr) const GKO_NOT_IMPLEMENTED;


void MpiExecutor::raw_copy_to(const DpcppExecutor *, size_type num_bytes,
                              const void *src_ptr,
                              void *dest_ptr) const GKO_NOT_IMPLEMENTED;


}  // namespace gko
