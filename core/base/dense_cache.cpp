// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/base/dense_cache.hpp"

#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace detail {

template <typename ValueType>
bool require_init(const std::unique_ptr<matrix::Dense<ValueType>>& vec,
                  std::shared_ptr<const Executor> exec, dim<2> size)
{
    return !vec || vec->get_size() != size || vec->get_executor() != exec;
}


template <typename ValueType>
void DenseCache<ValueType>::init(std::shared_ptr<const Executor> exec,
                                 dim<2> size) const
{
    if (require_init(vec, exec, size)) {
        vec = matrix::Dense<ValueType>::create(exec, size);
    }
}


template <typename ValueType>
void DenseCache<ValueType>::init_from(
    const matrix::Dense<ValueType>* template_vec) const
{
    if (require_init(vec, template_vec->get_executor(),
                     template_vec->get_size())) {
        vec = matrix::Dense<ValueType>::create_with_config_of(template_vec);
    }
}


#define GKO_DECLARE_DENSE_CACHE(_type) struct DenseCache<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_CACHE);


template <typename ValueType>
void DenseDualCache<ValueType>::init_from(
    const matrix::Dense<ValueType>* template_vec) const
{
    auto template_exec = template_vec->get_executor();
    auto template_size = template_vec->get_size();

    if (!require_init(vec_device, template_exec, template_size) &&
        vec_host->get_executor() == template_exec->get_master()) {
        return;
    }
    vec_device = matrix::Dense<ValueType>::create_with_config_of(template_vec);
    vec_host.vec =
        matrix::Dense<ValueType>::create(template_exec->get_master());
}


template <typename ValueType>
void DenseDualCache<ValueType>::init(std::shared_ptr<const Executor> exec,
                                     dim<2> size) const
{
    if (!require_init(vec_device, exec, size) &&
        vec_host->get_executor() == exec->get_master()) {
        return;
    }
    vec_device = matrix::Dense<ValueType>::create(exec, size);
    vec_host.vec = matrix::Dense<ValueType>::create(exec->get_master());
}


template <typename ValueType>
matrix::Dense<ValueType>* DenseDualCache<ValueType>::get(
    std::shared_ptr<const Executor> exec) const
{
    if (vec_device == nullptr || vec_host.get() == nullptr) {
        return nullptr;
    }
    if (exec != vec_device->get_executor() &&
        exec != vec_host->get_executor()) {
        std::cout << exec.get() << std::endl;
        std::cout << vec_device->get_executor().get() << std::endl;
        std::cout << vec_host->get_executor().get() << std::endl;
        GKO_NOT_SUPPORTED(exec);
    }
    if (exec == vec_device->get_executor()) {
        synchronize_device();
        device_is_active = true;
        return vec_device.get();
    } else {
        vec_host.init(vec_host->get_executor(), vec_device->get_size());
        synchronize_host();
        host_is_active = true;
        return vec_host.get();
    }
}


template <typename ValueType>
const matrix::Dense<ValueType>* DenseDualCache<ValueType>::get_const(
    std::shared_ptr<const Executor> exec) const
{
    if (vec_device == nullptr || vec_host.get() == nullptr) {
        return nullptr;
    }
    if (exec != vec_device->get_executor() &&
        exec != vec_host->get_executor()) {
        std::cout << exec.get() << std::endl;
        std::cout << vec_device->get_executor().get() << std::endl;
        std::cout << vec_host->get_executor().get() << std::endl;
        GKO_NOT_SUPPORTED(exec);
    }
    if (exec == vec_device->get_executor()) {
        synchronize_device();
        return vec_device.get();
    } else {
        std::cout << vec_host->get_size() << " " << vec_device->get_size()
                  << std::endl;
        vec_host.init(vec_host->get_executor(), vec_device->get_size());
        synchronize_host();
        return vec_host.get();
    }
}


template <typename ValueType>
void DenseDualCache<ValueType>::synchronize_host() const
{
    if (device_is_active &&
        vec_device->get_executor() != vec_host->get_executor()) {
        vec_host->copy_from(vec_device);
    }
    device_is_active = false;
}


template <typename ValueType>
void DenseDualCache<ValueType>::synchronize_device() const
{
    if (host_is_active &&
        vec_device->get_executor() != vec_host->get_executor()) {
        vec_device->copy_from(vec_host.get());
    }
    host_is_active = false;
}


#define GKO_DECLARE_DENSE_DUAL_CACHE(_type) struct DenseDualCache<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_DUAL_CACHE);


}  // namespace detail
}  // namespace gko
