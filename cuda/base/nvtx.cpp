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

#include <ginkgo/config.hpp>


#include <cuda_runtime.h>
#ifdef GKO_LEGACY_NVTX
#include <nvToolsExt.h>
#else
#include <nvtx3/nvToolsExt.h>
#endif


#include <ginkgo/core/log/profiler_hook.hpp>


namespace gko {
namespace log {


// "GKO" in ASCII to avoid collision with other application's categories
constexpr static uint32 category_magic_offset = 0x676B6FU;


void init_nvtx()
{
#define NAMED_CATEGORY(_name)                                             \
    nvtxNameCategory(static_cast<uint32>(profile_event_category::_name) + \
                         category_magic_offset,                           \
                     "gko::" #_name)
    NAMED_CATEGORY(memory);
    NAMED_CATEGORY(operation);
    NAMED_CATEGORY(object);
    NAMED_CATEGORY(linop);
    NAMED_CATEGORY(factory);
    NAMED_CATEGORY(solver);
    NAMED_CATEGORY(criterion);
    NAMED_CATEGORY(user);
    NAMED_CATEGORY(internal);
#undef NAMED_CATEGORY
}


std::function<void(const char*, profile_event_category)> begin_nvtx_fn(
    uint32_t color_argb)
{
    return [color_argb](const char* name, profile_event_category category) {
        nvtxEventAttributes_t attr{};
        attr.version = NVTX_VERSION;
        attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        attr.category = static_cast<uint32>(category) + category_magic_offset;
        attr.colorType = NVTX_COLOR_ARGB;
        attr.color = color_argb;
        attr.payloadType = NVTX_PAYLOAD_UNKNOWN;
        attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
        attr.message.ascii = name;
        nvtxRangePushEx(&attr);
    };
}


void end_nvtx(const char* name, profile_event_category) { nvtxRangePop(); }


}  // namespace log
}  // namespace gko
