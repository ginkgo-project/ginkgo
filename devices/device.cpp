// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>
#include <mutex>

#include <ginkgo/core/base/device.hpp>
#include <ginkgo/core/base/export_device.hpp>


namespace gko {


GKO_DEVICE_EXPORT std::mutex& nvidia_device::get_mutex(int i)
{
    static std::mutex mutex[max_devices];
    return mutex[i];
}


GKO_DEVICE_EXPORT int& nvidia_device::get_num_execs(int i)
{
    static int num_execs[max_devices];
    return num_execs[i];
}


GKO_DEVICE_EXPORT std::mutex& amd_device::get_mutex(int i)
{
    static std::mutex mutex[max_devices];
    return mutex[i];
}


GKO_DEVICE_EXPORT int& amd_device::get_num_execs(int i)
{
    static int num_execs[max_devices];
    return num_execs[i];
}


}  // namespace gko
