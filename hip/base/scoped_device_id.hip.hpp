// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_HIP_BASE_SCOPED_DEVICE_ID_HIP_HPP_
#define GKO_HIP_BASE_SCOPED_DEVICE_ID_HIP_HPP_


#include <ginkgo/core/base/scoped_device_id_guard.hpp>


namespace gko {
namespace detail {


/**
 * A scoped device id for HIP.
 */
class hip_scoped_device_id_guard : public generic_scoped_device_id_guard {
public:
    /**
     * The constructor sets the device id to the passed in value for the
     * lifetime of the created object.
     *
     * @param device_id  Set the device id to this.
     */
    explicit hip_scoped_device_id_guard(int device_id);

    /**
     * This resets the device id. If this fails, the program is terminated.
     */
    ~hip_scoped_device_id_guard() override;

    hip_scoped_device_id_guard(hip_scoped_device_id_guard&& other) noexcept;

    hip_scoped_device_id_guard& operator=(
        hip_scoped_device_id_guard&& other) noexcept;

private:
    int original_device_id_;
    bool need_reset_;
};


}  // namespace detail
}  // namespace gko


#endif  // GKO_HIP_BASE_SCOPED_DEVICE_ID_HIP_HPP_
