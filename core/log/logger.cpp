// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/log/logger.hpp>


namespace gko {
namespace log {


constexpr Logger::mask_type Logger::all_events_mask;
constexpr Logger::mask_type Logger::executor_events_mask;
constexpr Logger::mask_type Logger::operation_events_mask;
constexpr Logger::mask_type Logger::polymorphic_object_events_mask;
constexpr Logger::mask_type Logger::linop_events_mask;
constexpr Logger::mask_type Logger::linop_factory_events_mask;
constexpr Logger::mask_type Logger::batch_linop_factory_events_mask;
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
constexpr Logger::mask_type Logger::polymorphic_object_move_started_mask;
constexpr Logger::mask_type Logger::polymorphic_object_move_completed_mask;
constexpr Logger::mask_type Logger::polymorphic_object_deleted_mask;

constexpr Logger::mask_type Logger::linop_apply_started_mask;
constexpr Logger::mask_type Logger::linop_apply_completed_mask;
constexpr Logger::mask_type Logger::linop_advanced_apply_started_mask;
constexpr Logger::mask_type Logger::linop_advanced_apply_completed_mask;

constexpr Logger::mask_type Logger::linop_factory_generate_started_mask;
constexpr Logger::mask_type Logger::linop_factory_generate_completed_mask;

constexpr Logger::mask_type Logger::criterion_check_started_mask;
constexpr Logger::mask_type Logger::criterion_check_completed_mask;

constexpr Logger::mask_type Logger::batch_solver_completed_mask;

constexpr Logger::mask_type Logger::iteration_complete_mask;


}  // namespace log
}  // namespace gko
