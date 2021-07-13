/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include "common/base/kernel_launch.hpp"


#include <memory>
#include <type_traits>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "common/base/kernel_launch_reduction.hpp"
#include "common/base/kernel_launch_solver.hpp"
#include "core/test/utils.hpp"


namespace {


using gko::dim;
using gko::int64;
using gko::size_type;
using std::is_same;


class KernelLaunch : public ::testing::Test {
protected:
    KernelLaunch()
        : exec(gko::OmpExecutor::create()),
          zero_array(exec, 16),
          iota_array(exec, 16),
          iota_transp_array(exec, 16),
          iota_dense(gko::matrix::Dense<>::create(exec, dim<2>{4, 4})),
          zero_dense(gko::matrix::Dense<>::create(exec, dim<2>{4, 4}, 6)),
          zero_dense2(gko::matrix::Dense<>::create(exec, dim<2>{4, 4}, 5)),
          vec_dense(gko::matrix::Dense<>::create(exec, dim<2>{1, 4}))
    {
        for (int i = 0; i < 16; i++) {
            zero_array.get_data()[i] = 0;
            iota_array.get_data()[i] = i;
            iota_transp_array.get_data()[i] = (i % 4 * 4) + i / 4;
            iota_dense->at(i / 4, i % 4) = i;
            zero_dense->at(i / 4, i % 4) = 0.0;
            zero_dense2->at(i / 4, i % 4) = 0.0;
        }
    }

    std::shared_ptr<gko::OmpExecutor> exec;
    gko::Array<int> zero_array;
    gko::Array<int> iota_array;
    gko::Array<int> iota_transp_array;
    std::unique_ptr<gko::matrix::Dense<>> iota_dense;
    std::unique_ptr<gko::matrix::Dense<>> zero_dense;
    std::unique_ptr<gko::matrix::Dense<>> zero_dense2;
    std::unique_ptr<gko::matrix::Dense<>> vec_dense;
};


TEST_F(KernelLaunch, Runs1D)
{
    gko::kernels::omp::run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto d) {
            static_assert(is_same<decltype(i), int64>::value, "index");
            static_assert(is_same<decltype(d), int *>::value, "type");
            d[i] = i;
        },
        zero_array.get_num_elems(), zero_array.get_data());

    GKO_ASSERT_ARRAY_EQ(zero_array, iota_array);
}


TEST_F(KernelLaunch, Runs1DArray)
{
    gko::kernels::omp::run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto d, auto d_ptr) {
            static_assert(is_same<decltype(i), int64>::value, "index");
            static_assert(is_same<decltype(d), int *>::value, "type");
            static_assert(is_same<decltype(d_ptr), const int *>::value, "type");
            if (d == d_ptr) {
                d[i] = i;
            } else {
                d[i] = 0;
            }
        },
        zero_array.get_num_elems(), zero_array, zero_array.get_const_data());

    GKO_ASSERT_ARRAY_EQ(zero_array, iota_array);
}


TEST_F(KernelLaunch, Runs1DDense)
{
    gko::kernels::omp::run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto d, auto d2, auto d_ptr) {
            static_assert(is_same<decltype(i), int64>::value, "index");
            static_assert(is_same<decltype(d(0, 0)), double &>::value, "type");
            static_assert(is_same<decltype(d2(0, 0)), const double &>::value,
                          "type");
            static_assert(is_same<decltype(d_ptr), const double *>::value,
                          "type");
            bool pointers_correct = d.data == d_ptr && d2.data == d_ptr;
            bool strides_correct = d.stride == 5 && d2.stride == 5;
            bool accessors_2d_correct =
                &d(0, 0) == d_ptr && &d(1, 0) == d_ptr + d.stride &&
                &d2(0, 0) == d_ptr && &d2(1, 0) == d_ptr + d.stride;
            bool accessors_1d_correct = &d[0] == d_ptr && &d2[0] == d_ptr;
            if (pointers_correct && strides_correct && accessors_2d_correct &&
                accessors_1d_correct) {
                d(i / 4, i % 4) = i;
            } else {
                d(i / 4, i % 4) = 0;
            }
        },
        16, zero_dense2.get(),
        static_cast<const gko::matrix::Dense<> *>(zero_dense2.get()),
        zero_dense2->get_const_values());

    GKO_ASSERT_MTX_NEAR(zero_dense2, iota_dense, 0.0);
}


TEST_F(KernelLaunch, Runs2D)
{
    gko::kernels::omp::run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto j, auto d) {
            static_assert(is_same<decltype(i), int64>::value, "index");
            static_assert(is_same<decltype(j), int64>::value, "index");
            static_assert(is_same<decltype(d), int *>::value, "type");
            d[i + 4 * j] = 4 * i + j;
        },
        dim<2>{4, 4}, zero_array.get_data());

    GKO_ASSERT_ARRAY_EQ(zero_array, iota_transp_array);
}


TEST_F(KernelLaunch, Runs2DArray)
{
    gko::kernels::omp::run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto j, auto d, auto d_ptr) {
            static_assert(is_same<decltype(i), int64>::value, "index");
            static_assert(is_same<decltype(j), int64>::value, "index");
            static_assert(is_same<decltype(d), int *>::value, "type");
            static_assert(is_same<decltype(d_ptr), const int *>::value, "type");
            if (d == d_ptr) {
                d[i + 4 * j] = 4 * i + j;
            } else {
                d[i + 4 * j] = 0;
            }
        },
        dim<2>{4, 4}, zero_array, zero_array.get_const_data());

    GKO_ASSERT_ARRAY_EQ(zero_array, iota_transp_array);
}


TEST_F(KernelLaunch, Runs2DDense)
{
    gko::kernels::omp::run_kernel_solver(
        exec,
        [] GKO_KERNEL(auto i, auto j, auto d, auto d2, auto d_ptr, auto d3,
                      auto d4, auto d2_ptr, auto d3_ptr) {
            static_assert(is_same<decltype(i), int64>::value, "index");
            static_assert(is_same<decltype(d(0, 0)), double &>::value, "type");
            static_assert(is_same<decltype(d2(0, 0)), const double &>::value,
                          "type");
            static_assert(is_same<decltype(d_ptr), const double *>::value,
                          "type");
            static_assert(is_same<decltype(d3(0, 0)), double &>::value, "type");
            static_assert(is_same<decltype(d4), double *>::value, "type");
            static_assert(is_same<decltype(d2_ptr), double *>::value, "type");
            static_assert(is_same<decltype(d3_ptr), double *>::value, "type");
            bool pointers_correct = d.data == d_ptr && d2.data == d_ptr &&
                                    d3.data == d2_ptr && d4 == d3_ptr;
            bool strides_correct =
                d.stride == 5 && d2.stride == 5 && d3.stride == 6;
            bool accessors_2d_correct =
                &d(0, 0) == d_ptr && &d(1, 0) == d_ptr + d.stride &&
                &d2(0, 0) == d_ptr && &d2(1, 0) == d_ptr + d2.stride &&
                &d3(0, 0) == d2_ptr && &d3(1, 0) == d2_ptr + d3.stride;
            bool accessors_1d_correct =
                &d[0] == d_ptr && &d2[0] == d_ptr && &d3[0] == d2_ptr;
            if (pointers_correct && strides_correct && accessors_2d_correct &&
                accessors_1d_correct) {
                d(i, j) = 4 * i + j;
            } else {
                d(i, j) = 0;
            }
        },
        dim<2>{4, 4}, zero_dense->get_stride(), zero_dense2.get(),
        static_cast<const gko::matrix::Dense<> *>(zero_dense2.get()),
        zero_dense2->get_const_values(),
        gko::kernels::omp::default_stride(zero_dense.get()),
        gko::kernels::omp::row_vector(vec_dense.get()),
        zero_dense->get_values(), vec_dense->get_values());

    GKO_ASSERT_MTX_NEAR(zero_dense2, iota_dense, 0.0);
}

TEST_F(KernelLaunch, Reduction1D)
{
    gko::Array<int64> output{exec, 1};
    gko::kernels::omp::run_kernel_reduction(
        exec,
        [] GKO_KERNEL(auto i) {
            static_assert(is_same<decltype(i), int64>::value, "index");
            return i + 1;
        },
        [] GKO_KERNEL(auto i, auto j) { return i + j; },
        [] GKO_KERNEL(auto j) { return j * 2; }, int64{}, output.get_data(),
        size_type{100000});
    ASSERT_EQ(*output.get_const_data(), 10000100000ll);

    gko::kernels::omp::run_kernel_reduction(
        exec,
        [] GKO_KERNEL(auto i) {
            static_assert(is_same<decltype(i), int64>::value, "index");
            return i + 1;
        },
        [] GKO_KERNEL(auto i, auto j) { return i + j; },
        [] GKO_KERNEL(auto j) { return j * 2; }, int64{}, output.get_data(),
        size_type{10});
    ASSERT_EQ(*output.get_const_data(), 110ll);
}


TEST_F(KernelLaunch, Reduction2DSmallRows)
{
    gko::Array<int64> output{exec, 1};
    for (int cols = 0; cols < 17; cols++) {
        gko::kernels::omp::run_kernel_reduction(
            exec,
            [] GKO_KERNEL(auto i, auto j) {
                static_assert(is_same<decltype(i), int64>::value, "index");
                return (i + 1) * (j + 1);
            },
            [] GKO_KERNEL(auto i, auto j) { return i + j; },
            [] GKO_KERNEL(auto j) { return j * 4; }, int64{}, output.get_data(),
            gko::dim<2>{10, cols});
        ASSERT_EQ(*output.get_const_data(), 110ll * cols * (cols + 1));
    }
}


TEST_F(KernelLaunch, Reduction2DLargeRows)
{
    gko::Array<int64> output{exec, 1};
    for (int cols = 0; cols < 17; cols++) {
        gko::kernels::omp::run_kernel_reduction(
            exec,
            [] GKO_KERNEL(auto i, auto j) {
                static_assert(is_same<decltype(i), int64>::value, "index");
                return (i + 1) * (j + 1);
            },
            [] GKO_KERNEL(auto i, auto j) { return i + j; },
            [] GKO_KERNEL(auto j) { return j * 4; }, int64{}, output.get_data(),
            gko::dim<2>{1000, cols});
        ASSERT_EQ(*output.get_const_data(), 1001000ll * cols * (cols + 1));
    }
}


TEST_F(KernelLaunch, Reduction2D)
{
    gko::Array<int64> output{exec, 1};
    gko::kernels::omp::run_kernel_reduction(
        exec,
        [] GKO_KERNEL(auto i, auto j) {
            static_assert(is_same<decltype(i), int64>::value, "index");
            return (i + 1) * (j + 1);
        },
        [] GKO_KERNEL(auto i, auto j) { return i + j; },
        [] GKO_KERNEL(auto j) { return j * 4; }, int64{}, output.get_data(),
        gko::dim<2>{1000, 100});
    ASSERT_EQ(*output.get_const_data(), 10110100000ll);
}


}  // namespace
