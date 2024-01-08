// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cuda_runtime.h>


#include <ginkgo/config.hpp>


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
