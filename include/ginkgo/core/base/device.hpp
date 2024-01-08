// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_DEVICE_HPP_
#define GKO_PUBLIC_CORE_BASE_DEVICE_HPP_


#include <array>
#include <cstdint>
#include <mutex>
#include <type_traits>


#include <ginkgo/config.hpp>


namespace gko {


class CudaExecutor;

class HipExecutor;


/**
 * nvidia_device handles the number of executor on Nvidia devices and have the
 * corresponding recursive_mutex.
 */
class nvidia_device {
    friend class CudaExecutor;
    friend class HipExecutor;

private:
    /**
     * get_mutex gets the static mutex reference at i.
     *
     * @param i  index of mutex
     *
     * @return recursive_mutex reference
     */
    static std::mutex& get_mutex(int i);

    /**
     * get_num_execs gets the static num_execs reference at i.
     *
     * @param i  index of num_execs
     *
     * @return int reference
     */
    static int& get_num_execs(int i);

    static constexpr int max_devices = 64;
};


/**
 * amd_device handles the number of executor on Amd devices and have the
 * corresponding recursive_mutex.
 */
class amd_device {
    friend class HipExecutor;

private:
    /**
     * get_mutex gets the static mutex reference at i.
     *
     * @param i  index of mutex
     *
     * @return recursive_mutex reference
     */
    static std::mutex& get_mutex(int i);

    /**
     * get_num_execs gets the static num_execs reference at i.
     *
     * @param i  index of num_execs
     *
     * @return int reference
     */
    static int& get_num_execs(int i);

    static constexpr int max_devices = 64;
};


}  // namespace gko

#endif  // GKO_PUBLIC_CORE_BASE_DEVICE_HPP_
