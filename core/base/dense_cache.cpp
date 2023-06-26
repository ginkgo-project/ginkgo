// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/dense_cache.hpp>


#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace detail {


template <typename ValueType>
void DenseCache<ValueType>::init(std::shared_ptr<const Executor> exec,
                                 dim<2> size, gko::array<char>* storage) const
{
    if (vec && vec->get_size() == size && vec->get_executor() == exec) {
        return;
    }
    if (storage) {
        auto num_stored_elems = size[0] * size[1];
        storage->set_executor(exec);
        storage->resize_and_reset(sizeof(ValueType) * num_stored_elems);
        array<ValueType> value_storage(
            exec, num_stored_elems,
            reinterpret_cast<ValueType*>(storage->get_data()));
        vec = matrix::Dense<ValueType>::create(
            exec, size, std::move(value_storage), size[1]);
    } else {
        vec = matrix::Dense<ValueType>::create(exec, size);
    }
}


template <typename ValueType>
void DenseCache<ValueType>::init_from(
    const matrix::Dense<ValueType>* template_vec,
    gko::array<char>* storage) const
{
    if (vec && vec->get_size() == template_vec->get_size() &&
        vec->get_executor() == template_vec->get_executor()) {
        return;
    }
    if (storage) {
        auto exec = template_vec->get_executor();
        auto num_stored_elems = template_vec->get_num_stored_elements();
        storage->set_executor(exec);
        storage->resize_and_reset(sizeof(ValueType) * num_stored_elems);
        array<ValueType> value_storage(
            exec, num_stored_elems,
            reinterpret_cast<ValueType*>(storage->get_data()));
        vec = matrix::Dense<ValueType>::create(exec, template_vec->get_size(),
                                               std::move(value_storage),
                                               template_vec->get_stride());

    } else {
        vec = matrix::Dense<ValueType>::create_with_config_of(template_vec);
    }
}


#define GKO_DECLARE_DENSE_CACHE(_type) class DenseCache<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_CACHE);


}  // namespace detail
}  // namespace gko
