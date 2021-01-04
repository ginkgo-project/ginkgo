/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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


#include <cstdlib>
#include <cstring>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {


void OmpExecutor::raw_free(void *ptr) const noexcept { std::free(ptr); }


std::shared_ptr<Executor> OmpExecutor::get_master() noexcept
{
    return this->shared_from_this();
}


std::shared_ptr<const Executor> OmpExecutor::get_master() const noexcept
{
    return this->shared_from_this();
}


void *OmpExecutor::raw_alloc(size_type num_bytes) const
{
    return GKO_ENSURE_ALLOCATED(std::malloc(num_bytes), "OMP", num_bytes);
}


void OmpExecutor::raw_copy_to(const OmpExecutor *, size_type num_bytes,
                              const void *src_ptr, void *dest_ptr) const
{
    if (num_bytes > 0) {
        std::memcpy(dest_ptr, src_ptr, num_bytes);
    }
}


void OmpExecutor::synchronize() const
{
    // This is a no-op for single-threaded OMP
    // TODO: change when adding support for multi-threaded OMP execution
}


}  // namespace gko
