/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#ifndef GKO_CORE_BASE_GLOBAL_CONSTANT_HPP_
#define GKO_CORE_BASE_GLOBAL_CONSTANT_HPP_


#include <memory>
#include <mutex>
#include <type_traits>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/matrix/dense.hpp>

namespace gko {
namespace detail {
template <typename ExecutorType, typename ValueType, int max_devices,
          typename HostExecutorType>
class executor_storage {
public:
    using StorageType = matrix::Dense<ValueType>;
    executor_storage() : host(HostExecutorType::create()) {}

    std::shared_ptr<const StorageType> get(
        std::shared_ptr<const ExecutorType> exec, int device_id)
    {
        assert(device_id <= max_devices);
        assert(exec != exec->get_master());
        std::lock_guard<std::mutex> guard(mutex[device_id]);
        if (!storage[device_id]) {
            // check executor
            if (!executor[device_id]) {
                if (exec == exec->get_master()) {
                    if (typeid(ExecutorType) == typeid(HostExecutorType)) {
                        executor[device_id] = host;
                    } else {
                        // host memory executor does not need device_id;
                        executor[device_id] = ExecutorType::create();
                    }
                } else {
                    executor[device_id] = ExecutorType::create(device_id);
                }
            }
            // create the storage
            storage[device_id] =
                StorageType::create(executor[device_id], {ValueType(1)});
        }
        return storage[device_id];
    }

private:
    std::shared_ptr<const StorageType> storage[max_devices];
    std::shared_ptr<const ExecutorType> executor[max_devices];
    std::shared_ptr<const HostExecutorType> host;
    std::mutex mutex[max_devices];
};
}  // namespace detail

template <typename ValueType>
class global_constant {
public:
    using vec = matrix::Dense<ValueType>;
    static std::shared_ptr<const vec> one(std::shared_ptr<const Executor> exec)
    {
        if (auto concrete_exec =
                std::dynamic_pointer_cast<const ReferenceExecutor>(exec)) {
            ref_storage.get(concrete_exec, 0);
        } else if (auto concrete_exec =
                       std::dynamic_pointer_cast<const OmpExecutor>(exec)) {
            omp_storage.get(concrete_exec, 0);
        } else if (auto concrete_exec =
                       std::dynamic_pointer_cast<const CudaExecutor>(exec)) {
            cuda_storage.get(concrete_exec, concrete_exec->get_device_id());
        } else if (auto concrete_exec =
                       std::dynamic_pointer_cast<const HipExecutor>(exec)) {
            hip_storage.get(concrete_exec, concrete_exec->get_device_id());
        }
    }

private:
    static detail::executor_storage<CudaExecutor, ValueType, 64, OmpExecutor>
        cuda_storage;
    static detail::executor_storage<OmpExecutor, ValueType, 1, OmpExecutor>
        omp_storage;
    static detail::executor_storage<ReferenceExecutor, ValueType, 1,
                                    ReferenceExecutor>
        ref_storage;
    static detail::executor_storage<HipExecutor, ValueType, 64, OmpExecutor>
        hip_storage;
};


template class global_constant<double>;

}  // namespace gko

#endif  // GKO_CORE_BASE_GLOBAL_CONSTANT_HPP_
