/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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


#ifndef GKO_CORE_BASE_HANDLE_GUARD_HPP_
#define GKO_CORE_BASE_HANDLE_GUARD_HPP_


#include <ginkgo/core/base/async_handle.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/memory_space.hpp>


namespace gko {


class handle_guard {
public:
    handle_guard(std::shared_ptr<const Executor> exec,
                 std::shared_ptr<AsyncHandle> handle)
        : exec_(exec)
    {
        handle_ = exec->get_default_exec_stream();
        exec->set_default_exec_stream(handle);
    }

    ~handle_guard() noexcept
    {
        this->exec_->set_default_exec_stream(this->handle_);
    }

    handle_guard(handle_guard& other) = delete;

    handle_guard& operator=(const handle_guard& other) = delete;

    handle_guard(handle_guard&& other) = delete;

    handle_guard const& operator=(handle_guard&& other) = delete;

private:
    std::shared_ptr<AsyncHandle> handle_;
    std::shared_ptr<const Executor> exec_;
};


}  // namespace gko


#endif  // GKO_CORE_BASE_HANDLE_GUARD_HPP_
