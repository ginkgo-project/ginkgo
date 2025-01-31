// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <variant>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/extensions/kokkos.hpp>

#include "examples/batched-matrix-free-templated/batched/kernel_tags.hpp"
#include "ginkgo/core/matrix/identity.hpp"

namespace tensor {

using value_type = double;


void convert_tensor(gko::ptr_param<const gko::matrix::Dense<value_type>> A,
                    gko::ptr_param<const gko::matrix::Dense<value_type>> B,
                    gko::ptr_param<gko::matrix::Dense<value_type>> result)
{
    auto expected_dims = gko::dim<2>{A->get_size()[0] * B->get_size()[0],
                                     A->get_size()[1] * B->get_size()[1]};
    GKO_ASSERT_EQUAL_DIMENSIONS(result, expected_dims);
    auto exec = result->get_executor();
    auto host_result = gko::make_temporary_clone(exec->get_master(), result);
    auto host_a = gko::make_temporary_clone(exec->get_master(), A);
    auto host_b = gko::make_temporary_clone(exec->get_master(), B);

    for (gko::size_type ai = 0; ai < A->get_size()[0]; ++ai) {
        for (gko::size_type aj = 0; aj < A->get_size()[1]; ++aj) {
            for (gko::size_type bi = 0; bi < B->get_size()[0]; ++bi) {
                for (gko::size_type bj = 0; bj < B->get_size()[1]; ++bj) {
                    auto i = ai * A->get_size()[0] + bi;
                    auto j = aj * A->get_size()[1] + bj;
                    host_result->at(i, j) =
                        host_a->at(ai, aj) * host_b->at(bi, bj);
                }
            }
        }
    }
}

void convert_tensor(gko::ptr_param<const gko::matrix::Dense<value_type>> A,
                    gko::ptr_param<const gko::matrix::Identity<value_type>> B,
                    gko::ptr_param<gko::matrix::Dense<value_type>> result)
{
    auto expected_dims = gko::dim<2>{A->get_size()[0] * B->get_size()[0],
                                     A->get_size()[1] * B->get_size()[1]};
    GKO_ASSERT_EQUAL_DIMENSIONS(result, expected_dims);

    result->fill(0.0);

    auto exec = result->get_executor();
    auto host_result = gko::make_temporary_clone(exec->get_master(), result);
    auto host_a = gko::make_temporary_clone(exec->get_master(), A);

    for (gko::size_type ai = 0; ai < A->get_size()[0]; ++ai) {
        for (gko::size_type aj = 0; aj < A->get_size()[1]; ++aj) {
            for (gko::size_type bi = 0; bi < B->get_size()[0]; ++bi) {
                auto i = ai * B->get_size()[0] + bi;
                auto j = aj * B->get_size()[1] + bi;
                host_result->at(i, j) = host_a->at(ai, aj);
            }
        }
    }
}

void convert_tensor(gko::ptr_param<const gko::matrix::Identity<value_type>> A,
                    gko::ptr_param<const gko::matrix::Dense<value_type>> B,
                    gko::ptr_param<gko::matrix::Dense<value_type>> result)
{
    auto expected_dims = gko::dim<2>{A->get_size()[0] * B->get_size()[0],
                                     A->get_size()[1] * B->get_size()[1]};
    GKO_ASSERT_EQUAL_DIMENSIONS(result, expected_dims);

    result->fill(0.0);

    auto exec = result->get_executor();
    auto host_result = gko::make_temporary_clone(exec->get_master(), result);
    auto host_b = gko::make_temporary_clone(exec->get_master(), B);

    for (gko::size_type ai = 0; ai < A->get_size()[0]; ++ai) {
        for (gko::size_type bi = 0; bi < B->get_size()[0]; ++bi) {
            for (gko::size_type bj = 0; bj < B->get_size()[1]; ++bj) {
                auto i = ai * B->get_size()[0] + bi;
                auto j = ai * B->get_size()[1] + bj;
                host_result->at(i, j) = host_b->at(bi, bj);
            }
        }
    }
}


struct tensor_left_view {
    gko::size_type num_batch_items;
    gko::int32 stride;
    gko::int32 size_1d;
    const value_type* data;
};

struct tensor_left_item {
    gko::int32 stride;
    gko::int32 size_1d;
    const value_type* data;
};

constexpr tensor_left_item extract_batch_item(tensor_left_view op,
                                              gko::size_type batch_id)
{
    return {op.stride, op.size_1d, op.data + batch_id * op.size_1d * op.stride};
}

class TensorLeft : public gko::EnablePolymorphicObject<TensorLeft> {
public:
    struct const_item {};

    explicit TensorLeft(std::shared_ptr<const gko::Executor> exec,
                        gko::size_type num_batch_items = 0,
                        gko::size_type num_rows_1d = 0)
        : EnablePolymorphicObject(exec),
          size_(num_batch_items,
                gko::dim<2>{num_rows_1d * num_rows_1d * num_rows_1d}),
          data_(gko::batch::matrix::Dense<value_type>::create(
              exec,
              gko::batch_dim<2>{num_batch_items, gko::dim<2>{num_rows_1d}}))
    {}

    explicit TensorLeft(
        std::unique_ptr<gko::batch::matrix::Dense<value_type>> data)
        : EnablePolymorphicObject(data->get_executor()),
          size_(data->get_num_batch_items(),
                gko::dim<2>{data->get_common_size()[0] *
                            data->get_common_size()[0] *
                            data->get_common_size()[0]}),
          data_(std::move(data))
    {}

    [[nodiscard]] tensor_left_view create_view() const
    {
        return {this->get_num_batch_items(),
                static_cast<gko::int32>(data_->get_common_size()[0]),
                static_cast<gko::int32>(data_->get_common_size()[1]),
                data_->get_const_values()};
    }

    [[nodiscard]] constexpr gko::batch_dim<2> get_size() const { return size_; }

    [[nodiscard]] constexpr gko::dim<2> get_common_size() const
    {
        return size_.get_common_size();
    }

    [[nodiscard]] constexpr gko::size_type get_num_batch_items() const
    {
        return size_.get_num_batch_items();
    }

    [[nodiscard]] const gko::batch::matrix::Dense<value_type>* get_data() const
    {
        return data_.get();
    }

private:
    gko::batch_dim<2> size_;
    std::unique_ptr<gko::batch::matrix::Dense<value_type>> data_;
};


constexpr void advanced_apply(
    double alpha, tensor_left_item a,
    gko::batch::multi_vector::batch_item<const double> b, double beta,
    gko::batch::multi_vector::batch_item<double> x,
    [[maybe_unused]] std::variant<gko::reference_kernel, gko::omp_kernel>)
{
    for (gko::int32 k = 0; k < a.size_1d; ++k) {
        for (gko::int32 j = 0; j < a.size_1d; ++j) {
            for (gko::int32 i = 0; i < a.size_1d; ++i) {
                auto vector_start = k * a.size_1d * a.size_1d + i;

                value_type acc = 0;
                for (gko::size_type q = 0; q < a.size_1d; q++) {
                    auto vector_index = vector_start + q * a.size_1d;
                    acc = a.data[j * a.size_1d + q] * b.values[vector_index] +
                          acc;
                }
                auto row = k * a.size_1d * a.size_1d + j * a.size_1d + i;
                x.values[row] = alpha * acc + beta * x.values[row];
            }
        }
    }
}

constexpr void simple_apply(
    const tensor_left_item& a,
    const gko::batch::multi_vector::batch_item<const double>& b,
    const gko::batch::multi_vector::batch_item<double>& x,
    std::variant<gko::reference_kernel, gko::omp_kernel> tag)
{
    advanced_apply(1.0, a, b, 0.0, x, tag);
}


std::unique_ptr<gko::batch::matrix::Dense<value_type>> convert(
    gko::ptr_param<const TensorLeft> tensor)
{
    auto result = gko::batch::matrix::Dense<value_type>::create(
        tensor->get_executor(), tensor->get_size());

    auto size_1d = tensor->get_data()->get_common_size()[0];
    auto id = gko::matrix::Identity<value_type>::create(tensor->get_executor(),
                                                        size_1d);
    auto intermediate = gko::matrix::Dense<value_type>::create(
        tensor->get_executor(), gko::dim<2>{size_1d * size_1d});
    for (gko::size_type batch = 0; batch < tensor->get_num_batch_items();
         ++batch) {
        convert_tensor(tensor->get_data()->create_const_view_for_item(batch),
                       id, intermediate);
        convert_tensor(id, intermediate, result->create_view_for_item(batch));
    }

    return result;
}


}  // namespace tensor
