// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/extensions/kokkos.hpp>

namespace tensor {

using value_type = double;

struct tensor_left_view {
    gko::size_type num_batch_items;
    gko::int32 num_rows;
    gko::int32 num_cols;
    gko::int32 stride;
    const value_type* data;
};

struct tensor_left_item {
    gko::int32 num_rows;
    gko::int32 num_cols;
    gko::int32 stride;
    const value_type* data;
};

constexpr tensor_left_item extract_batch_item(tensor_left_view op,
                                              gko::size_type batch_id)
{
    return {op.num_rows, op.num_cols, op.stride,
            op.data + batch_id * op.num_rows * op.stride};
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

    [[nodiscard]] constexpr tensor_left_view create_view() const
    {
        return {this->get_num_batch_items(),
                static_cast<gko::int32>(this->get_common_size()[0]),
                static_cast<gko::int32>(this->get_common_size()[1]),
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
    std::unique_ptr<gko::batch::matrix::Dense<value_type>> convert_to_dense()
        const;

private:
    gko::batch_dim<2> size_;
    std::unique_ptr<gko::batch::matrix::Dense<value_type>> data_;
};

inline std::unique_ptr<gko::batch::matrix::Dense<value_type>>
TensorLeft::convert_to_dense() const
{
    gko::array<value_type> result_array{this->get_executor(),
                                        this->get_num_batch_items() *
                                            this->get_common_size()[0] *
                                            this->get_common_size()[1]};
    result_array.fill(gko::zero<value_type>());

    auto size_1d = data_->get_common_size()[0];

    for (gko::size_type outer_block = 0; outer_block < size_1d; ++outer_block) {
        for (gko::size_type inner_block_row = 0; inner_block_row < size_1d;
             ++inner_block_row) {
            for (gko::size_type inner_block_col = 0; inner_block_col < size_1d;
                 ++inner_block_col) {
                for (gko::size_type i = 0; i < size_1d; ++i) {
                    auto row = outer_block * size_1d * size_1d +
                               inner_block_row * size_1d + i;
                    auto col = outer_block * size_1d * size_1d +
                               inner_block_col * size_1d + i;
                    result_array
                        .get_data()[row * (size_1d * size_1d * size_1d) + col] =
                        data_->at(outer_block, inner_block_row,
                                  inner_block_col);
                }
            }
        }
    }

    //
    // struct functor {
    //
    //     KOKKOS_INLINE_FUNCTION void operator()(int row) const
    //     {
    //         auto outer_block_id = row / (size_1d * size_1d);
    //         auto inner_block_col = (row % (size_1d * size_1d)) % size_1d;
    //         auto inner_block_row = row % size_1d;
    //         auto val = block[outer_block_id * size_1d * size_1d +
    //         inner_block_row * size_1d + inner_block_col]; for (gko::size_type
    //         j = 0; j < size_1d ; ++j) {
    //             auto col = outer_block_id * size_1d * size_1d + j * size_1d +
    //             inner_block_row; dense[col + row * size_1d * size_1d *
    //             size_1d] = block[outer_block_id * size_1d * size_1d]
    //         }
    //     }
    //
    //
    //     gko::size_type size_1d;
    //     value_type* dense;
    //     const value_type* block;
    // };


    return gko::batch::matrix::Dense<value_type>::create(
        this->get_executor(), this->get_size(), std::move(result_array));
}


}  // namespace tensor
