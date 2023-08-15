// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_BATCH_UTILITIES_HPP_
#define GKO_CORE_BASE_BATCH_UTILITIES_HPP_


#include <algorithm>
#include <type_traits>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/base/utils_helper.hpp>


namespace gko {
namespace batch {
namespace multivector {


template <typename ValueType>
std::unique_ptr<batch::MultiVector<ValueType>> duplicate(
    std::shared_ptr<const Executor> exec, size_type num_duplications,
    const batch::MultiVector<ValueType>* input)
{
    auto num_batch_items = input->get_num_batch_items();
    auto tmp = batch::MultiVector<ValueType>::create(
        exec, batch_dim<2>(num_batch_items * num_duplications,
                           input->get_common_size()));

    for (size_type i = 0; i < num_duplications; ++i) {
        for (size_type b = 0; b < num_batch_items; ++b) {
            tmp->create_view_for_item(i * num_batch_items + b)
                ->copy_from(input->create_const_view_for_item(b).get());
        }
    }

    return std::move(tmp);
}


template <typename ValueType>
std::unique_ptr<batch::MultiVector<ValueType>> create_from_dense(
    std::shared_ptr<const Executor> exec, const size_type num_duplications,
    const matrix::Dense<ValueType>* input)
{
    auto num_batch_items = num_duplications;
    auto tmp = batch::MultiVector<ValueType>::create(
        exec, batch_dim<2>(num_batch_items, input->get_size()));

    for (size_type b = 0; b < num_batch_items; ++b) {
        tmp->create_view_for_item(b)->copy_from(input);
    }

    return std::move(tmp);
}


template <typename ValueType>
std::unique_ptr<batch::MultiVector<ValueType>> create_from_dense(
    std::shared_ptr<const Executor> exec,
    const std::vector<matrix::Dense<ValueType>*>& input)
{
    auto num_batch_items = input.size();
    auto tmp = batch::MultiVector<ValueType>::create(
        exec, batch_dim<2>(num_batch_items, input[0]->get_size()));

    for (size_type b = 0; b < num_batch_items; ++b) {
        tmp->create_view_for_item(b)->copy_from(input[b]);
    }

    return std::move(tmp);
}


template <typename ValueType>
std::vector<std::unique_ptr<matrix::Dense<ValueType>>> unbatch(
    const batch::MultiVector<ValueType>* batch_multivec)
{
    auto exec = batch_multivec->get_executor();
    auto unbatched_mats =
        std::vector<std::unique_ptr<matrix::Dense<ValueType>>>{};
    for (size_type b = 0; b < batch_multivec->get_num_batch_items(); ++b) {
        unbatched_mats.emplace_back(
            batch_multivec->create_const_view_for_item(b)->clone());
    }
    return unbatched_mats;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<MultiVector<ValueType>> read(
    std::shared_ptr<const Executor> exec,
    const std::vector<gko::matrix_data<ValueType, IndexType>>& data)
{
    auto num_batch_items = data.size();
    auto tmp = MultiVector<ValueType>::create(
        exec, batch_dim<2>(num_batch_items, data[0].size));

    for (size_type b = 0; b < num_batch_items; ++b) {
        tmp->create_view_for_item(b)->read(data[b]);
    }

    return std::move(tmp);
}


template <typename ValueType, typename IndexType>
std::vector<gko::matrix_data<ValueType, IndexType>> write(
    const MultiVector<ValueType>* mvec)
{
    auto data = std::vector<gko::matrix_data<ValueType, IndexType>>(
        mvec->get_num_batch_items());

    for (size_type b = 0; b < mvec->get_num_batch_items(); ++b) {
        data[b] = {mvec->get_common_size(), {}};
        mvec->create_const_view_for_item(b)->write(data[b]);
    }

    return data;
}


}  // namespace multivector
}  // namespace batch
}  // namespace gko


#endif  // GKO_CORE_BASE_BATCH_UTILITIES_HPP_
