// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <variant>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/extensions/kokkos.hpp>

#include "batched/batch_user_linop.hpp"
#include "batched/kernel_tags.hpp"
#include "ginkgo/core/matrix/identity.hpp"

namespace tensor {

using ValueType = double;


void convert_tensor(gko::ptr_param<const gko::matrix::Dense<ValueType>> A,
                    gko::ptr_param<const gko::matrix::Dense<ValueType>> B,
                    gko::ptr_param<gko::matrix::Dense<ValueType>> result)
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

void convert_tensor(gko::ptr_param<const gko::matrix::Dense<ValueType>> A,
                    gko::ptr_param<const gko::matrix::Identity<ValueType>> B,
                    gko::ptr_param<gko::matrix::Dense<ValueType>> result)
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

void convert_tensor(gko::ptr_param<const gko::matrix::Identity<ValueType>> A,
                    gko::ptr_param<const gko::matrix::Dense<ValueType>> B,
                    gko::ptr_param<gko::matrix::Dense<ValueType>> result)
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
    const ValueType* data;
};

struct tensor_left_item {
    gko::int32 stride;
    gko::int32 size_1d;
    const ValueType* data;
};

constexpr tensor_left_item extract_batch_item(tensor_left_view op,
                                              gko::size_type batch_id)
{
    return {op.stride, op.size_1d, op.data + batch_id * op.size_1d * op.stride};
}

class TensorLeft
    : public gko::batch_template::EnableBatchUserLinOp<ValueType, TensorLeft> {
public:
    using value_type = ValueType;
    struct const_item {};

    explicit TensorLeft(std::shared_ptr<const gko::Executor> exec,
                        gko::size_type num_batch_items = 0,
                        gko::size_type num_rows_1d = 0)
        : EnableBatchUserLinOp(
              exec, gko::batch_dim<2>{num_batch_items,
                                      gko::dim<2>{num_rows_1d * num_rows_1d *
                                                  num_rows_1d}}),
          data_(gko::batch::matrix::Dense<value_type>::create(
              exec,
              gko::batch_dim<2>{num_batch_items, gko::dim<2>{num_rows_1d}}))
    {}

    explicit TensorLeft(
        std::unique_ptr<gko::batch::matrix::Dense<value_type>> data)
        : EnableBatchUserLinOp(
              data->get_executor(),
              gko::batch_dim<2>{data->get_num_batch_items(),
                                gko::dim<2>{data->get_common_size()[0] *
                                            data->get_common_size()[0] *
                                            data->get_common_size()[0]}}),
          data_(std::move(data))
    {}

    [[nodiscard]] tensor_left_view create_view() const
    {
        return {this->get_num_batch_items(),
                static_cast<gko::int32>(data_->get_common_size()[0]),
                static_cast<gko::int32>(data_->get_common_size()[1]),
                data_->get_const_values()};
    }

    [[nodiscard]] const gko::batch::matrix::Dense<value_type>* get_data() const
    {
        return data_.get();
    }

private:
    std::shared_ptr<gko::batch::matrix::Dense<value_type>> data_;
};


constexpr void advanced_apply(
    double alpha, tensor_left_item a,
    gko::batch::multi_vector::batch_item<const double> b, double beta,
    gko::batch::multi_vector::batch_item<double> x,
    [[maybe_unused]] gko::cpu_kernel)
{
    for (gko::int32 k = 0; k < a.size_1d; ++k) {
        for (gko::int32 j = 0; j < a.size_1d; ++j) {
            for (gko::int32 i = 0; i < a.size_1d; ++i) {
                auto vector_start = k * a.size_1d * a.size_1d + i;

                ValueType acc = 0;
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
    const gko::batch::multi_vector::batch_item<double>& x, gko::cpu_kernel tag)
{
    advanced_apply(1.0, a, b, 0.0, x, tag);
}

#if defined(GINKGO_BUILD_CUDA) || defined(GINKGO_BUILD_HIP)


__device__ void advanced_apply(
    double alpha, tensor_left_item a,
    gko::batch::multi_vector::batch_item<const double> b, double beta,
    gko::batch::multi_vector::batch_item<double> x,
    [[maybe_unused]] gko::cuda_hip_kernel)
{
    auto row =
        static_cast<gko::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
    auto n = a.size_1d;
    auto num_rows = n * n * n;

    if (row >= num_rows) {
        return;
    }

    auto k = row / (n * n);
    auto j = (row - k * n * n) / n;
    auto i = (row - k * n * n) % n;
    auto vector_start = k * n * n + i;

    ValueType acc = 0;
    for (gko::size_type q = 0; q < n; q++) {
        auto vector_index = vector_start + q * n;
        acc = a.data[j * n + q] * b.values[vector_index] + acc;
    }
    x.values[row] = alpha * acc + beta * x.values[row];
}

__device__ void simple_apply(
    const tensor_left_item& a,
    const gko::batch::multi_vector::batch_item<const double>& b,
    const gko::batch::multi_vector::batch_item<double>& x,
    gko::cuda_hip_kernel tag)
{
    advanced_apply(1.0, a, b, 0.0, x, tag);
}

#endif


std::unique_ptr<gko::batch::matrix::Dense<ValueType>> convert(
    gko::ptr_param<const TensorLeft> tensor)
{
    auto result = gko::batch::matrix::Dense<ValueType>::create(
        tensor->get_executor(), tensor->get_size());

    auto size_1d = tensor->get_data()->get_common_size()[0];
    auto id = gko::matrix::Identity<ValueType>::create(tensor->get_executor(),
                                                       size_1d);
    auto intermediate = gko::matrix::Dense<ValueType>::create(
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
