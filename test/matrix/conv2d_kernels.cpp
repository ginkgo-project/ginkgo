// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/matrix/conv.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/matrix/conv_kernels.hpp"
#include "core/test/utils.hpp"
#include "test/utils/common_fixture.hpp"


class Conv2d : public CommonTestFixture {
protected:
    using ValueType = value_type;
    using Mtx = gko::matrix::Conv2d<ValueType>;
    using Dense = gko::matrix::Dense<ValueType>;

    Conv2d() : mtx_size(152, 152), kernel_size(7, 7), rand_engine(42) {}

    template <typename MtxType>
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols, int stride)
    {
        return gko::test::generate_random_matrix<MtxType>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<value_type>(0.0, 100.0), rand_engine, ref,
            gko::dim<2>{static_cast<gko::size_type>(num_rows),
                        static_cast<gko::size_type>(num_cols)},
            stride);
    }

    std::unique_ptr<Mtx> gen_conv(gko::dim<2> size)
    {
        int size_x = size[0];
        int size_y = size[1];
        std::vector<ValueType> data(size_x * size_y);
        auto value_dist = std::normal_distribution<value_type>(0.0, 1.0);
        for (int i = 0; i < size_x * size_y; i++) {
            data.at(i) = gko::test::detail::get_rand_value<ValueType>(
                value_dist, rand_engine);
        }

        auto host_array_2d = gko::share(gko::matrix::Dense<ValueType>::create(
            exec, gko::dim<2>{size_x, size_y}));
        std::copy(data.begin(), data.end(), host_array_2d->get_values());

        return Mtx::create(ref, host_array_2d);
    }

    void set_up_apply_data(gko::size_type in_stride = 1,
                           gko::size_type out_stride = 1)
    {
        conv = gen_conv(kernel_size);
        dconv = gko::clone(exec, conv);
        // only support padding = 0 stride = 1 currently
        int padding = 0;
        int stride = 1;
        auto expect_size_x =
            (mtx_size[0] + 2 * padding - kernel_size[0]) / stride + 1;

        auto expect_size_y =
            (mtx_size[1] + 2 * padding - kernel_size[1]) / stride + 1;

        // if (in_stride < mtx_size[1]) in_stride = mtx_size[1];
        // if (out_stride < expect_size_y) out_stride = expect_size_y;

        input = gen_mtx<Dense>(mtx_size[0], mtx_size[1], in_stride);
        output = gen_mtx<Dense>(expect_size_x, expect_size_y, out_stride);
        dinput = Dense::create(exec, input->get_size(), input->get_stride());
        doutput = Dense::create(exec, output->get_size(), output->get_stride());
        dinput->copy_from(input);
        doutput->copy_from(output);
    }

    const gko::dim<2> mtx_size;
    const gko::dim<2> kernel_size;
    std::default_random_engine rand_engine;

    std::unique_ptr<Mtx> conv;
    std::unique_ptr<Mtx> dconv;

    std::unique_ptr<Dense> input;
    std::unique_ptr<Dense> dinput;
    std::unique_ptr<Dense> output;
    std::unique_ptr<Dense> doutput;
};

/*
TEST_F(Conv2d, ApplyToDenseIsEquivalentToRef)
{
    set_up_apply_data();

    conv->apply(input, output);
    dconv->apply(dinput, doutput);

    GKO_ASSERT_MTX_NEAR(doutput, output, r<value_type>::value);
}

*/
TEST_F(Conv2d, ApplyToDenseWithStrideIsEquivalentToRef)
{
    set_up_apply_data(154, 155);

    conv->apply(input, output);
    dconv->apply(dinput, doutput);

    GKO_ASSERT_MTX_NEAR(doutput, output, r<value_type>::value);
}
