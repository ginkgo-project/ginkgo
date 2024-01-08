// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>
#include <mutex>


#include <ginkgo/core/base/device.hpp>


namespace gko {


std::mutex& nvidia_device::get_mutex(int i)
{
    static std::mutex mutex[max_devices];
    return mutex[i];
}


int& nvidia_device::get_num_execs(int i)
{
    static int num_execs[max_devices];
    return num_execs[i];
}


std::mutex& amd_device::get_mutex(int i)
{
    static std::mutex mutex[max_devices];
    return mutex[i];
}


int& amd_device::get_num_execs(int i)
{
    static int num_execs[max_devices];
    return num_execs[i];
}


}  // namespace gko
