/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#ifndef GKO_PUBLIC_CORE_LOG_PROFILER_HOOK_HPP_
#define GKO_PUBLIC_CORE_LOG_PROFILER_HOOK_HPP_


#include <ginkgo/config.hpp>
#include <ginkgo/core/log/logger.hpp>


namespace gko {
namespace log {


enum class profile_event_category {
    memory,
    operation,
    object,
    linop,
    factory,
    criterion,
};


/** The Ginkgo yellow background color as packed 24 bit RGB value. */
constexpr uint32 color_yellow_rgb = 0xFFCB05U;


std::shared_ptr<Logger> get_tau_hook(bool initialize = true);


std::shared_ptr<Logger> create_nvtx_hook(uint32 color_rgb = color_yellow_rgb);


std::shared_ptr<Logger> create_roctx_hook();


/**
 * Sets the name for an object to be profiled. Every instance of that object in
 * the profile will be replaced by the name instead of its runtime type.
 */
void set_profiled_object_name(const PolymorphicObject* obj, std::string name);


}  // namespace log
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_LOG_PROFILER_HOOK_HPP_
