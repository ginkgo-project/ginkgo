// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GINKGO_CORE_BASE_NOOP_SCOPED_DEVICE_ID_GUARD_HPP
#define GINKGO_CORE_BASE_NOOP_SCOPED_DEVICE_ID_GUARD_HPP


#include <ginkgo/core/base/scoped_device_id_guard.hpp>


namespace gko {
namespace detail {


/**
 * An implementation of generic_scoped_device_id_guard that does nothing.
 *
 * This is used for OmpExecutor and DpcppExecutor, since they don't require
 * setting a device id.
 */
class noop_scoped_device_id_guard : public generic_scoped_device_id_guard {};


}  // namespace detail
}  // namespace gko


#endif  // GINKGO_CORE_BASE_NOOP_SCOPED_DEVICE_ID_GUARD_HPP
