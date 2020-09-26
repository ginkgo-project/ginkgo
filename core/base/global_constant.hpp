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


#include <iostream>
#include <memory>
#include <mutex>
#include <type_traits>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace detail {

template <typename ExecutorType, typename ValueType>
class host_executor_storage {
public:
    using StorageType = matrix::Dense<ValueType>;
    host_executor_storage() : executor_(ExecutorType::create()) {}
    std::shared_ptr<const StorageType> get(
        std::shared_ptr<const ExecutorType> exec)
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (!storage_) {
            // create the storage
            storage_ = initialize<StorageType>({ValueType(1)}, executor_);
        }
        return storage_;
    }

private:
    std::shared_ptr<const StorageType> storage_;
    std::shared_ptr<const ExecutorType> executor_;
    std::mutex mutex_;
};


template <typename ExecutorType, int max_devices, typename ValueType>
class device_executor_storage {
public:
    using StorageType = matrix::Dense<ValueType>;
    using HostExecutorType = OmpExecutor;
    void clear()
    {
        for (int i = 0; i < max_devices; i++) {
            std::lock_guard<std::mutex> guard(mutex_[i]);
            storage_[i] = nullptr;
            executor_[i] = nullptr;
        }
    }
    device_executor_storage() : host_(HostExecutorType::create()) {}
    std::shared_ptr<const StorageType> get(
        std::shared_ptr<const ExecutorType> exec)
    {
        auto device_id = exec->get_device_id();
        std::lock_guard<std::mutex> guard(mutex_[device_id]);
        if (!storage_[device_id]) {
            // check executor
            if (!executor_[device_id]) {
                executor_[device_id] = ExecutorType::create(device_id, host_);
            }
            // create the storage
            storage_[device_id] =
                initialize<StorageType>({ValueType(1)}, executor_[device_id]);
        }
        return storage_[device_id];
    }

private:
    std::shared_ptr<const StorageType> storage_[max_devices];
    std::shared_ptr<const ExecutorType> executor_[max_devices];
    std::shared_ptr<HostExecutorType> host_;
    std::mutex mutex_[max_devices];
};
}  // namespace detail

template <typename ValueType>
class global_constant {
public:
    // using ValueType=double;
    using StorageType = matrix::Dense<ValueType>;
    static std::shared_ptr<const StorageType> one(
        std::shared_ptr<const Executor> exec)
    {
        std::shared_ptr<const StorageType> storage;
        if (auto concrete_exec =
                std::dynamic_pointer_cast<const ReferenceExecutor>(exec)) {
            storage = ref_storage.get(concrete_exec);
        } else if (auto concrete_exec =
                       std::dynamic_pointer_cast<const OmpExecutor>(exec)) {
            storage = omp_storage.get(concrete_exec);
        } else if (auto concrete_exec =
                       std::dynamic_pointer_cast<const CudaExecutor>(exec)) {
            storage = cuda_storage.get(concrete_exec);
        } else if (auto concrete_exec =
                       std::dynamic_pointer_cast<const HipExecutor>(exec)) {
            storage = hip_storage.get(concrete_exec);
        }
        std::lock_guard<std::mutex> guard(mutex);
        if (!register_deleter) {
            // need to register the deleter in the main scope.
            // otherwise, propably lead cudaErrorCudartUnloading because the
            // static destructor is in the exit stage but CUDA could call
            // unregisterBinaryUtil before this destructor.
            std::cout << "register" << std::endl;
            std::atexit([]() {
                hip_storage.clear();
                cuda_storage.clear();
                std::cout << "clear" << std::endl;
            });
            register_deleter = true;
        }
        return storage;
    }

private:
    static detail::host_executor_storage<ReferenceExecutor, ValueType>
        ref_storage;
    static detail::host_executor_storage<OmpExecutor, ValueType> omp_storage;
    static detail::device_executor_storage<CudaExecutor, 64, ValueType>
        cuda_storage;
    static detail::device_executor_storage<HipExecutor, 64, ValueType>
        hip_storage;
    static bool register_deleter;
    static std::mutex mutex;
};


}  // namespace gko

#endif  // GKO_CORE_BASE_GLOBAL_CONSTANT_HPP_
