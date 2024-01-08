// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CUDA_BASE_SCOPED_DEVICE_ID_HPP_
#define GKO_CUDA_BASE_SCOPED_DEVICE_ID_HPP_


#include <ginkgo/core/base/scoped_device_id_guard.hpp>


namespace gko {
namespace detail {


/**
 * A scoped device id for CUDA.
 */
class cuda_scoped_device_id_guard : public generic_scoped_device_id_guard {
public:
    /**
     * The constructor sets the device id to the passed in value for the
     * lifetime of the created object.
     *
     * @param device_id  Set the device id to this.
     */
    explicit cuda_scoped_device_id_guard(int device_id);

    /**
     * This resets the device id. If this fails, the program is terminated.
     */
    ~cuda_scoped_device_id_guard() override;

    cuda_scoped_device_id_guard(cuda_scoped_device_id_guard&& other) noexcept;

    cuda_scoped_device_id_guard& operator=(
        cuda_scoped_device_id_guard&& other) noexcept;

private:
    int original_device_id_;
    bool need_reset_;
};


}  // namespace detail
}  // namespace gko


#endif  // GKO_CUDA_BASE_SCOPED_DEVICE_ID_HPP_
