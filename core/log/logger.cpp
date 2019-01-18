/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2019

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/


#include <ginkgo/core/log/logger.hpp>


namespace gko {
namespace log {


constexpr Logger::mask_type Logger::all_events_mask;
constexpr Logger::mask_type Logger::executor_events_mask;
constexpr Logger::mask_type Logger::operation_events_mask;
constexpr Logger::mask_type Logger::polymorphic_object_events_mask;
constexpr Logger::mask_type Logger::linop_events_mask;
constexpr Logger::mask_type Logger::linop_factory_events_mask;
constexpr Logger::mask_type Logger::criterion_events_mask;

constexpr Logger::mask_type Logger::allocation_started_mask;
constexpr Logger::mask_type Logger::allocation_completed_mask;
constexpr Logger::mask_type Logger::free_started_mask;
constexpr Logger::mask_type Logger::free_completed_mask;
constexpr Logger::mask_type Logger::copy_started_mask;
constexpr Logger::mask_type Logger::copy_completed_mask;

constexpr Logger::mask_type Logger::operation_launched_mask;
constexpr Logger::mask_type Logger::operation_completed_mask;

constexpr Logger::mask_type Logger::polymorphic_object_create_started_mask;
constexpr Logger::mask_type Logger::polymorphic_object_create_completed_mask;
constexpr Logger::mask_type Logger::polymorphic_object_copy_started_mask;
constexpr Logger::mask_type Logger::polymorphic_object_copy_completed_mask;
constexpr Logger::mask_type Logger::polymorphic_object_deleted_mask;

constexpr Logger::mask_type Logger::linop_apply_started_mask;
constexpr Logger::mask_type Logger::linop_apply_completed_mask;
constexpr Logger::mask_type Logger::linop_advanced_apply_started_mask;
constexpr Logger::mask_type Logger::linop_advanced_apply_completed_mask;

constexpr Logger::mask_type Logger::linop_factory_generate_started_mask;
constexpr Logger::mask_type Logger::linop_factory_generate_completed_mask;

constexpr Logger::mask_type Logger::criterion_check_started_mask;
constexpr Logger::mask_type Logger::criterion_check_completed_mask;

constexpr Logger::mask_type Logger::iteration_complete_mask;

}  // namespace log
}  // namespace gko
