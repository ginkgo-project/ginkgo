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

#ifndef GKO_PUBLIC_EXT_RESOURCE_MANAGER_LOG_MASK_TYPE_HPP_
#define GKO_PUBLIC_EXT_RESOURCE_MANAGER_LOG_MASK_TYPE_HPP_


#include <unordered_map>


#include <ginkgo/core/log/logger.hpp>

namespace gko {
namespace extension {
namespace resource_manager {

// clang-format off
#define GKORM_LOGGER_EVENT(_event_mask) \
    {#_event_mask, gko::log::Logger::_event_mask}
// clang-format on

// Note: before using the following command, clang ColumnLimit should use 160 or
// large enough value. Otherwise, some like criterion_check_completed will not
// be caught, so need to manually add them.
// clang-format off
// For single:
// grep -E 'GKO_LOGGER_REGISTER_EVENT\([0-9]+, [^,]*,' include/ginkgo/core/log/logger.hpp | sed -E 's/.*GKO_LOGGER_REGISTER_EVENT\([0-9]+, ([^,]*),.*/GKORM_LOGGER_EVENT(\1_mask),/g'
// For combine:
// grep -E 'mask_type +[^ ]*_mask +=' include/ginkgo/core/log/logger.hpp | sed -E 's/.*mask_type +([^ ]*_mask).*/GKORM_LOGGER_EVENT(\1),/g'
// clang-format on
std::unordered_map<std::string, gko::log::Logger::mask_type> mask_type_map{
    GKORM_LOGGER_EVENT(allocation_started_mask),
    GKORM_LOGGER_EVENT(allocation_completed_mask),
    GKORM_LOGGER_EVENT(free_started_mask),
    GKORM_LOGGER_EVENT(free_completed_mask),
    GKORM_LOGGER_EVENT(copy_started_mask),
    GKORM_LOGGER_EVENT(copy_completed_mask),
    GKORM_LOGGER_EVENT(operation_launched_mask),
    GKORM_LOGGER_EVENT(operation_completed_mask),
    GKORM_LOGGER_EVENT(polymorphic_object_create_started_mask),
    GKORM_LOGGER_EVENT(polymorphic_object_create_completed_mask),
    GKORM_LOGGER_EVENT(polymorphic_object_copy_started_mask),
    GKORM_LOGGER_EVENT(polymorphic_object_copy_completed_mask),
    GKORM_LOGGER_EVENT(polymorphic_object_deleted_mask),
    GKORM_LOGGER_EVENT(linop_apply_started_mask),
    GKORM_LOGGER_EVENT(linop_apply_completed_mask),
    GKORM_LOGGER_EVENT(linop_advanced_apply_started_mask),
    GKORM_LOGGER_EVENT(linop_advanced_apply_completed_mask),
    GKORM_LOGGER_EVENT(linop_factory_generate_started_mask),
    GKORM_LOGGER_EVENT(linop_factory_generate_completed_mask),
    GKORM_LOGGER_EVENT(criterion_check_started_mask),
    GKORM_LOGGER_EVENT(criterion_check_completed_mask),
    GKORM_LOGGER_EVENT(iteration_complete_mask),
    GKORM_LOGGER_EVENT(polymorphic_object_move_started_mask),
    GKORM_LOGGER_EVENT(polymorphic_object_move_completed_mask),
    GKORM_LOGGER_EVENT(all_events_mask),
    GKORM_LOGGER_EVENT(executor_events_mask),
    GKORM_LOGGER_EVENT(operation_events_mask),
    GKORM_LOGGER_EVENT(polymorphic_object_events_mask),
    GKORM_LOGGER_EVENT(linop_events_mask),
    GKORM_LOGGER_EVENT(linop_factory_events_mask),
    GKORM_LOGGER_EVENT(criterion_events_mask)};


}  // namespace resource_manager
}  // namespace extension
}  // namespace gko

#endif  // GKO_PUBLIC_EXT_RESOURCE_MANAGER_LOG_MASK_TYPE_HPP_
