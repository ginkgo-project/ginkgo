// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
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
#include <ginkgo/core/matrix/batch_dense.hpp>


namespace gko {
namespace batch {


/**
 * Duplicate a given input batch object.
 */
template <typename OutputType, typename... TArgs>
std::unique_ptr<OutputType> duplicate(std::shared_ptr<const Executor> exec,
                                      size_type num_duplications,
                                      const OutputType* input,
                                      TArgs&&... create_args)
{
    auto num_batch_items = input->get_num_batch_items();
    auto tmp =
        OutputType::create(exec,
                           batch_dim<2>(num_batch_items * num_duplications,
                                        input->get_common_size()),
                           std::forward<TArgs>(create_args)...);

    for (size_type i = 0; i < num_duplications; ++i) {
        for (size_type b = 0; b < num_batch_items; ++b) {
            tmp->create_view_for_item(i * num_batch_items + b)
                ->copy_from(input->create_const_view_for_item(b).get());
        }
    }

    return std::move(tmp);
}


/**
 * Duplicate a monolithic matrix and create a batch object.
 */
template <typename OutputType, typename... TArgs>
std::unique_ptr<OutputType> create_from_item(
    std::shared_ptr<const Executor> exec, const size_type num_duplications,
    const typename OutputType::unbatch_type* input, TArgs&&... create_args)
{
    auto num_batch_items = num_duplications;
    auto tmp = OutputType::create(
        exec, batch_dim<2>(num_batch_items, input->get_size()),
        std::forward<TArgs>(create_args)...);

    for (size_type b = 0; b < num_batch_items; ++b) {
        tmp->create_view_for_item(b)->copy_from(input);
    }

    return std::move(tmp);
}


/**
 * Create a batch object from a vector of monolithic object that share the same
 * sparsity pattern.
 *
 * @note The sparsity of the elements in the input vector of matrices needs to
 * be the same. TODO: Check for same sparsity among the different input items
 */
template <typename OutputType, typename... TArgs>
std::unique_ptr<OutputType> create_from_item(
    std::shared_ptr<const Executor> exec,
    const std::vector<typename OutputType::unbatch_type*>& input,
    TArgs&&... create_args)
{
    auto num_batch_items = input.size();
    auto tmp = OutputType::create(
        exec, batch_dim<2>(num_batch_items, input[0]->get_size()),
        std::forward<TArgs>(create_args)...);

    for (size_type b = 0; b < num_batch_items; ++b) {
        tmp->create_view_for_item(b)->copy_from(input[b]);
    }

    return std::move(tmp);
}


/**
 * Unbatch a batched object into a vector of items of its unbatch_type.
 */
template <typename InputType>
auto unbatch(const InputType* batch_object)
{
    auto unbatched_mats =
        std::vector<std::unique_ptr<typename InputType::unbatch_type>>{};
    for (size_type b = 0; b < batch_object->get_num_batch_items(); ++b) {
        unbatched_mats.emplace_back(
            batch_object->create_const_view_for_item(b)->clone());
    }
    return unbatched_mats;
}


namespace detail {


template <typename ValueType, typename IndexType>
void assert_same_sparsity_in_batched_data(
    const std::vector<gko::matrix_data<ValueType, IndexType>>& data)
{
    if (data.empty()) {
        return;
    }
    auto num_nnz = data.at(0).nonzeros.size();
    auto base_data = data.at(0);
    base_data.sort_row_major();
    for (int b = 1; b < data.size(); ++b) {
        if (data[b].nonzeros.size() != num_nnz) {
            GKO_NOT_IMPLEMENTED;
        }
        auto temp_data = data.at(b);
        temp_data.sort_row_major();
        for (int nnz = 0; nnz < num_nnz; ++nnz) {
            if (temp_data.nonzeros.at(nnz).row !=
                    base_data.nonzeros.at(nnz).row ||
                temp_data.nonzeros.at(nnz).column !=
                    base_data.nonzeros.at(nnz).column) {
                GKO_NOT_IMPLEMENTED;
            }
        }
    }
}


}  // namespace detail


/**
 * Create a batch object from a vector of gko::matrix_data objects. Each item of
 * the vector needs to store the same sparsity pattern.
 */
template <typename ValueType, typename IndexType, typename OutputType,
          typename... TArgs>
std::unique_ptr<OutputType> read(
    std::shared_ptr<const Executor> exec,
    const std::vector<gko::matrix_data<ValueType, IndexType>>& data,
    TArgs&&... create_args)
{
    auto num_batch_items = data.size();
    // Throw if all the items in the batch dont have same sparsity.
    if (!std::is_same<OutputType,
                      gko::batch::matrix::Dense<ValueType>>::value &&
        !std::is_same<OutputType, gko::batch::MultiVector<ValueType>>::value) {
        detail::assert_same_sparsity_in_batched_data(data);
    }
    auto tmp =
        OutputType::create(exec, batch_dim<2>(num_batch_items, data.at(0).size),
                           std::forward<TArgs>(create_args)...);

    for (size_type b = 0; b < num_batch_items; ++b) {
        if (data.at(b).size != data.at(0).size) {
            GKO_INVALID_STATE("Incorrect data passed in");
        }
        tmp->create_view_for_item(b)->read(data[b]);
    }

    return std::move(tmp);
}


/**
 * Write a vector of matrix data objects from an input batch object.
 */
template <typename ValueType, typename IndexType, typename OutputType>
std::vector<gko::matrix_data<ValueType, IndexType>> write(
    const OutputType* mvec)
{
    auto data = std::vector<gko::matrix_data<ValueType, IndexType>>(
        mvec->get_num_batch_items());

    for (size_type b = 0; b < mvec->get_num_batch_items(); ++b) {
        data[b] = {mvec->get_common_size(), {}};
        mvec->create_const_view_for_item(b)->write(data[b]);
    }

    return data;
}


/**
 * Creates and initializes a batch of the specified Matrix type from a series of
 * single column-vectors.
 *
 * @tparam Matrix  matrix type to initialize (It has to implement the
 *                 read<Matrix> function)
 * @tparam TArgs  argument types for Matrix::create method
 *                (not including the implied Executor as the first argument)
 *
 * @param vals  values used to initialize the batch vector
 * @param exec  Executor associated to the vector
 * @param create_args  additional arguments passed to Matrix::create, not
 *                     including the Executor, which is passed as the first
 *                     argument
 *
 * @ingroup mat_formats
 */
template <typename Matrix, typename... TArgs>
std::unique_ptr<Matrix> initialize(
    std::initializer_list<std::initializer_list<typename Matrix::value_type>>
        vals,
    std::shared_ptr<const Executor> exec, TArgs&&... create_args)
{
    using value_type = typename Matrix::value_type;
    using index_type = typename Matrix::index_type;
    using mat_data = gko::matrix_data<value_type, index_type>;
    size_type num_batch_items = vals.size();
    GKO_THROW_IF_INVALID(num_batch_items > 0, "Input data is empty");
    auto vals_begin = begin(vals);
    size_type common_num_rows = vals_begin ? vals_begin->size() : 0;
    auto common_size = dim<2>(common_num_rows, 1);
    for (auto& val : vals) {
        GKO_ASSERT_EQ(common_num_rows, val.size());
    }
    auto b_size = batch_dim<2>(num_batch_items, common_size);
    size_type batch = 0;
    std::vector<mat_data> input_mat_data(num_batch_items, common_size);
    for (const auto& b : vals) {
        input_mat_data[batch].nonzeros.reserve(b.size());
        size_type idx = 0;
        for (const auto& elem : b) {
            if (elem != zero<value_type>()) {
                input_mat_data[batch].nonzeros.emplace_back(idx, 0, elem);
            }
            ++idx;
        }
        ++batch;
    }
    return read<value_type, index_type, Matrix>(
        exec, input_mat_data, std::forward<TArgs>(create_args)...);
}


/**
 * Creates and initializes a batch of matrices.
 *
 * @tparam Matrix  matrix type to initialize (It has to implement the
 *                 read<Matrix> function)
 * @tparam TArgs  argument types for Matrix::create method
 *                (not including the implied Executor as the first argument)
 *
 * @param vals  values used to initialize the matrix
 * @param exec  Executor associated with the matrix
 * @param create_args  additional arguments passed to Matrix::create, not
 *                     including the Executor, which is passed as the first
 *                     argument
 *
 * @ingroup mat_formats
 */
template <typename Matrix, typename... TArgs>
std::unique_ptr<Matrix> initialize(
    std::initializer_list<std::initializer_list<
        std::initializer_list<typename Matrix::value_type>>>
        vals,
    std::shared_ptr<const Executor> exec, TArgs&&... create_args)
{
    using value_type = typename Matrix::value_type;
    using index_type = typename Matrix::index_type;
    using mat_data = gko::matrix_data<value_type, index_type>;
    size_type num_batch_items = vals.size();
    GKO_THROW_IF_INVALID(num_batch_items > 0, "Input data is empty");
    auto vals_begin = begin(vals);
    size_type common_num_rows = vals_begin ? vals_begin->size() : 0;
    size_type common_num_cols =
        vals_begin->begin() ? vals_begin->begin()->size() : 0;
    auto common_size = dim<2>(common_num_rows, common_num_cols);
    for (const auto& b : vals) {
        auto num_rows = b.size();
        auto num_cols = begin(b)->size();
        auto b_size = dim<2>(num_rows, num_cols);
        GKO_ASSERT_EQUAL_DIMENSIONS(b_size, common_size);
    }

    auto b_size = batch_dim<2>(num_batch_items, common_size);
    size_type batch = 0;
    std::vector<mat_data> input_mat_data(num_batch_items, common_size);
    for (const auto& b : vals) {
        size_type ridx = 0;
        for (const auto& row : b) {
            size_type cidx = 0;
            for (const auto& elem : row) {
                if (elem != zero<value_type>()) {
                    input_mat_data[batch].nonzeros.emplace_back(ridx, cidx,
                                                                elem);
                }
                ++cidx;
            }
            ++ridx;
        }
        ++batch;
    }
    return read<value_type, index_type, Matrix>(
        exec, input_mat_data, std::forward<TArgs>(create_args)...);
}


/**
 * Creates and initializes a batch of specified Matrix type with a single
 * column-vector by making copies of the single input column vector.
 *
 * @tparam Matrix  matrix type to initialize (It has to implement the
 *                 read<Matrix> function)
 * @tparam TArgs  argument types for Matrix::create method
 *                (not including the implied Executor as the first argument)
 *
 * @param num_batch_items  The number of times the input vector is to be
 *                         duplicated
 * @param vals  values used to initialize each vector in the temp. batch
 * @param exec  Executor associated with the matrix
 * @param create_args  additional arguments passed to Matrix::create, not
 *                     including the Executor, which is passed as the first
 *                     argument
 *
 * @ingroup mat_formats
 */
template <typename Matrix, typename... TArgs>
std::unique_ptr<Matrix> initialize(
    const size_type num_batch_items,
    std::initializer_list<typename Matrix::value_type> vals,
    std::shared_ptr<const Executor> exec, TArgs&&... create_args)
{
    using value_type = typename Matrix::value_type;
    using index_type = typename Matrix::index_type;
    using mat_data = gko::matrix_data<value_type, index_type>;
    GKO_THROW_IF_INVALID(num_batch_items > 0 && vals.size() > 0,
                         "Input data is empty");
    auto num_rows = begin(vals) ? vals.size() : 0;
    auto common_size = dim<2>(num_rows, 1);
    auto b_size = batch_dim<2>(num_batch_items, common_size);
    mat_data single_mat_data(common_size);
    single_mat_data.nonzeros.reserve(num_rows);
    size_type idx = 0;
    for (const auto& elem : vals) {
        if (elem != zero<value_type>()) {
            single_mat_data.nonzeros.emplace_back(idx, 0, elem);
        }
        ++idx;
    }
    std::vector<mat_data> input_mat_data(num_batch_items, single_mat_data);
    return read<value_type, index_type, Matrix>(
        exec, input_mat_data, std::forward<TArgs>(create_args)...);
}


/**
 * Creates and initializes a matrix from copies of a given matrix.
 *
 * @tparam Matrix  matrix type to initialize (It has to implement the
 *                 read<Matrix> function)
 * @tparam TArgs  argument types for Matrix::create method
 *                (not including the implied Executor as the first argument)
 *
 * @param num_batch_items  The number of times the input matrix is duplicated
 * @param vals  values used to initialize each matrix in the temp. batch
 * @param exec  Executor associated to the matrix
 * @param create_args  additional arguments passed to Matrix::create, not
 *                     including the Executor, which is passed as the first
 *                     argument
 *
 * @ingroup mat_formats
 */
template <typename Matrix, typename... TArgs>
std::unique_ptr<Matrix> initialize(
    const size_type num_batch_items,
    std::initializer_list<std::initializer_list<typename Matrix::value_type>>
        vals,
    std::shared_ptr<const Executor> exec, TArgs&&... create_args)
{
    using value_type = typename Matrix::value_type;
    using index_type = typename Matrix::index_type;
    using mat_data = gko::matrix_data<value_type, index_type>;
    GKO_THROW_IF_INVALID(num_batch_items > 0 && vals.size() > 0,
                         "Input data is empty");
    auto common_size = dim<2>(begin(vals) ? vals.size() : 0,
                              begin(vals) ? begin(vals)->size() : 0);
    batch_dim<2> b_size(num_batch_items, common_size);
    mat_data single_mat_data(common_size);
    size_type ridx = 0;
    for (const auto& row : vals) {
        size_type cidx = 0;
        for (const auto& elem : row) {
            if (elem != zero<value_type>()) {
                single_mat_data.nonzeros.emplace_back(ridx, cidx, elem);
            }
            ++cidx;
        }
        ++ridx;
    }
    std::vector<mat_data> input_mat_data(num_batch_items, single_mat_data);
    return read<value_type, index_type, Matrix>(
        exec, input_mat_data, std::forward<TArgs>(create_args)...);
}


}  // namespace batch
}  // namespace gko


#endif  // GKO_CORE_BASE_BATCH_UTILITIES_HPP_
