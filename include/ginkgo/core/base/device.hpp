/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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
