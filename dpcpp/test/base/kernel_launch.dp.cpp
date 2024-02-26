// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "common/unified/base/kernel_launch.hpp"


#include <memory>
#include <type_traits>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "common/unified/base/kernel_launch_reduction.hpp"
#include "common/unified/base/kernel_launch_solver.hpp"
#include "core/base/array_access.hpp"
#include "core/test/utils.hpp"


using gko::dim;
using gko::int64;
using gko::size_type;
using std::is_same;


struct move_only_type {
    move_only_type() {}

    move_only_type(move_only_type&&) {}

    move_only_type(const move_only_type&) = delete;
};


move_only_type move_only_val{};


namespace gko {
namespace kernels {
namespace dpcpp {


template <>
struct to_device_type_impl<move_only_type&> {
    using type = int64;

    static type map_to_device(move_only_type&) { return 0; }
};


}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


class KernelLaunch : public ::testing::Test {
protected:
#if GINKGO_DPCPP_SINGLE_MODE
    using value_type = float;
#else
    using value_type = double;
#endif
    using Mtx = gko::matrix::Dense<value_type>;

    KernelLaunch()
        : exec(gko::DpcppExecutor::create(
              0, gko::ReferenceExecutor::create(),
              gko::DpcppExecutor::get_num_devices("gpu") > 0 ? "gpu" : "cpu")),
          zero_array(exec->get_master(), 16),
          iota_array(exec->get_master(), 16),
          iota_transp_array(exec->get_master(), 16),
          iota_dense(Mtx::create(exec, dim<2>{4, 4})),
          zero_dense(Mtx::create(exec, dim<2>{4, 4}, 6)),
          zero_dense2(Mtx::create(exec, dim<2>{4, 4}, 5)),
          vec_dense(Mtx::create(exec, dim<2>{1, 4}))
    {
        auto ref_iota_dense = Mtx::create(exec->get_master(), dim<2>{4, 4});
        for (int i = 0; i < 16; i++) {
            zero_array.get_data()[i] = 0;
            iota_array.get_data()[i] = i;
            iota_transp_array.get_data()[i] = (i % 4 * 4) + i / 4;
            ref_iota_dense->at(i / 4, i % 4) = i;
        }
        zero_dense->fill(0.0);
        zero_dense2->fill(0.0);
        iota_dense->copy_from(ref_iota_dense);
        zero_array.set_executor(exec);
        iota_array.set_executor(exec);
        iota_transp_array.set_executor(exec);
    }

    std::shared_ptr<gko::DpcppExecutor> exec;
    gko::array<int> zero_array;
    gko::array<int> iota_array;
    gko::array<int> iota_transp_array;
    std::unique_ptr<Mtx> iota_dense;
    std::unique_ptr<Mtx> zero_dense;
    std::unique_ptr<Mtx> zero_dense2;
    std::unique_ptr<Mtx> vec_dense;
};


TEST_F(KernelLaunch, Runs1D)
{
    gko::kernels::dpcpp::run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto d, auto dummy) {
            static_assert(is_same<decltype(i), int64>::value, "index");
            static_assert(is_same<decltype(d), int*>::value, "type");
            static_assert(is_same<decltype(dummy), int64>::value, "dummy");
            d[i] = i;
        },
        zero_array.get_size(), zero_array.get_data(), move_only_val);

    GKO_ASSERT_ARRAY_EQ(zero_array, iota_array);
}


TEST_F(KernelLaunch, Runs1DArray)
{
    gko::kernels::dpcpp::run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto d, auto d_ptr, auto dummy) {
            static_assert(is_same<decltype(i), int64>::value, "index");
            static_assert(is_same<decltype(d), int*>::value, "type");
            static_assert(is_same<decltype(d_ptr), const int*>::value, "type");
            static_assert(is_same<decltype(dummy), int64>::value, "dummy");
            if (d == d_ptr) {
                d[i] = i;
            } else {
                d[i] = 0;
            }
        },
        zero_array.get_size(), zero_array, zero_array.get_const_data(),
        move_only_val);

    GKO_ASSERT_ARRAY_EQ(zero_array, iota_array);
}


TEST_F(KernelLaunch, Runs1DDense)
{
    gko::kernels::dpcpp::run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto d, auto d2, auto d_ptr, auto dummy) {
            static_assert(is_same<decltype(i), int64>::value, "index");
            static_assert(is_same<decltype(d(0, 0)), value_type&>::value,
                          "type");
            static_assert(is_same<decltype(d2(0, 0)), const value_type&>::value,
                          "type");
            static_assert(is_same<decltype(d_ptr), const value_type*>::value,
                          "type");
            static_assert(is_same<decltype(dummy), int64>::value, "dummy");
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
        16, zero_dense2.get(), static_cast<const Mtx*>(zero_dense2.get()),
        zero_dense2->get_const_values(), move_only_val);

    GKO_ASSERT_MTX_NEAR(zero_dense2, iota_dense, 0.0);
}


TEST_F(KernelLaunch, Runs2D)
{
    gko::kernels::dpcpp::run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto j, auto d, auto dummy) {
            static_assert(is_same<decltype(i), int64>::value, "index");
            static_assert(is_same<decltype(j), int64>::value, "index");
            static_assert(is_same<decltype(d), int*>::value, "type");
            static_assert(is_same<decltype(dummy), int64>::value, "dummy");
            d[i + 4 * j] = 4 * i + j;
        },
        dim<2>{4, 4}, zero_array.get_data(), move_only_val);

    GKO_ASSERT_ARRAY_EQ(zero_array, iota_transp_array);
}


TEST_F(KernelLaunch, Runs2DArray)
{
    gko::kernels::dpcpp::run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto j, auto d, auto d_ptr, auto dummy) {
            static_assert(is_same<decltype(i), int64>::value, "index");
            static_assert(is_same<decltype(j), int64>::value, "index");
            static_assert(is_same<decltype(d), int*>::value, "type");
            static_assert(is_same<decltype(d_ptr), const int*>::value, "type");
            static_assert(is_same<decltype(dummy), int64>::value, "dummy");
            if (d == d_ptr) {
                d[i + 4 * j] = 4 * i + j;
            } else {
                d[i + 4 * j] = 0;
            }
        },
        dim<2>{4, 4}, zero_array, zero_array.get_const_data(), move_only_val);

    GKO_ASSERT_ARRAY_EQ(zero_array, iota_transp_array);
}


TEST_F(KernelLaunch, Runs2DDense)
{
    gko::kernels::dpcpp::run_kernel_solver(
        exec,
        [] GKO_KERNEL(auto i, auto j, auto d, auto d2, auto d_ptr, auto d3,
                      auto d4, auto d2_ptr, auto d3_ptr, auto dummy) {
            static_assert(is_same<decltype(i), int64>::value, "index");
            static_assert(is_same<decltype(d(0, 0)), value_type&>::value,
                          "type");
            static_assert(is_same<decltype(d_ptr), const value_type*>::value,
                          "type");
            static_assert(is_same<decltype(d3(0, 0)), value_type&>::value,
                          "type");
            static_assert(is_same<decltype(d4), value_type*>::value, "type");
            static_assert(is_same<decltype(d2_ptr), value_type*>::value,
                          "type");
            static_assert(is_same<decltype(d3_ptr), value_type*>::value,
                          "type");
            static_assert(is_same<decltype(dummy), int64>::value, "dummy");
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
        static_cast<const Mtx*>(zero_dense2.get()),
        zero_dense2->get_const_values(),
        gko::kernels::dpcpp::default_stride(zero_dense.get()),
        gko::kernels::dpcpp::row_vector(vec_dense.get()),
        zero_dense->get_values(), vec_dense->get_values(), move_only_val);

    GKO_ASSERT_MTX_NEAR(zero_dense2, iota_dense, 0.0);
}


TEST_F(KernelLaunch, Reduction1D)
{
    gko::array<int64> output{exec, 1};

    gko::kernels::dpcpp::run_kernel_reduction(
        exec,
        [] GKO_KERNEL(auto i, auto a, auto dummy) {
            static_assert(is_same<decltype(i), int64>::value, "index");
            static_assert(is_same<decltype(a), int64*>::value, "value");
            static_assert(is_same<decltype(dummy), int64>::value, "dummy");
            return i + 1;
        },
        [] GKO_KERNEL(auto i, auto j) {
            static_assert(is_same<decltype(i), int64>::value, "i");
            static_assert(is_same<decltype(j), int64>::value, "j");
            return i + j;
        },
        [] GKO_KERNEL(auto j) {
            static_assert(is_same<decltype(j), int64>::value, "j");
            return j * 2;
        },
        int64{}, output.get_data(), size_type{100000}, output, move_only_val);

    // 2 * sum i=0...99999 (i+1)
    EXPECT_EQ(get_element(output, 0), 10000100000LL);

    gko::kernels::dpcpp::run_kernel_reduction(
        exec,
        [] GKO_KERNEL(auto i, auto a, auto dummy) {
            static_assert(is_same<decltype(i), int64>::value, "index");
            static_assert(is_same<decltype(a), int64*>::value, "value");
            static_assert(is_same<decltype(dummy), int64>::value, "dummy");
            return i + 1;
        },
        [] GKO_KERNEL(auto i, auto j) {
            static_assert(is_same<decltype(i), int64>::value, "i");
            static_assert(is_same<decltype(j), int64>::value, "j");
            return i + j;
        },
        [] GKO_KERNEL(auto j) {
            static_assert(is_same<decltype(j), int64>::value, "j");
            return j * 2;
        },
        int64{}, output.get_data(), size_type{100}, output, move_only_val);

    // 2 * sum i=0...99 (i+1)
    EXPECT_EQ(get_element(output, 0), 10100LL);
}


TEST_F(KernelLaunch, Reduction2D)
{
    gko::array<int64> output{exec, 1};

    gko::kernels::dpcpp::run_kernel_reduction(
        exec,
        [] GKO_KERNEL(auto i, auto j, auto a, auto dummy) {
            static_assert(is_same<decltype(i), int64>::value, "index");
            static_assert(is_same<decltype(j), int64>::value, "index");
            static_assert(is_same<decltype(a), int64*>::value, "value");
            static_assert(is_same<decltype(dummy), int64>::value, "dummy");
            return (i + 1) * (j + 1);
        },
        [] GKO_KERNEL(auto i, auto j) {
            static_assert(is_same<decltype(i), int64>::value, "i");
            static_assert(is_same<decltype(j), int64>::value, "j");
            return i + j;
        },
        [] GKO_KERNEL(auto j) {
            static_assert(is_same<decltype(j), int64>::value, "j");
            return j * 4;
        },
        int64{}, output.get_data(), gko::dim<2>{1000, 100}, output,
        move_only_val);

    // 4 * sum i=0...999 sum j=0...99 of (i+1)*(j+1)
    EXPECT_EQ(get_element(output, 0), 10110100000LL);

    gko::kernels::dpcpp::run_kernel_reduction(
        exec,
        [] GKO_KERNEL(auto i, auto j, auto a, auto dummy) {
            static_assert(is_same<decltype(i), int64>::value, "index");
            static_assert(is_same<decltype(j), int64>::value, "index");
            static_assert(is_same<decltype(a), int64*>::value, "value");
            static_assert(is_same<decltype(dummy), int64>::value, "dummy");
            return (i + 1) * (j + 1);
        },
        [] GKO_KERNEL(auto i, auto j) {
            static_assert(is_same<decltype(i), int64>::value, "i");
            static_assert(is_same<decltype(j), int64>::value, "j");
            return i + j;
        },
        [] GKO_KERNEL(auto j) {
            static_assert(is_same<decltype(j), int64>::value, "j");
            return j * 4;
        },
        int64{}, output.get_data(), gko::dim<2>{10, 10}, output, move_only_val);

    // 4 * sum i=0...9 sum j=0...9 of (i+1)*(j+1)
    ASSERT_EQ(get_element(output, 0), 12100LL);
}


TEST_F(KernelLaunch, ReductionRow2D)
{
    for (auto num_rows : {0, 1, 10, 100, 1000, 10000}) {
        for (auto num_cols : {0, 1, 10, 100, 1000, 10000}) {
            SCOPED_TRACE(std::to_string(num_rows) + " rows, " +
                         std::to_string(num_cols) + " cols");
            gko::array<int64> host_ref{exec->get_master(),
                                       static_cast<size_type>(2 * num_rows)};
            std::fill_n(host_ref.get_data(), 2 * num_rows, 1234);
            gko::array<int64> output{exec, host_ref};
            for (int i = 0; i < num_rows; i++) {
                // we are computing 2 * sum {j=0, j<cols} (i+1)*(j+1) for each
                // row i and storing it with stride 2
                host_ref.get_data()[2 * i] =
                    static_cast<int64>(num_cols) * (num_cols + 1) * (i + 1);
            }

            gko::kernels::dpcpp::run_kernel_row_reduction(
                exec,
                [] GKO_KERNEL(auto i, auto j, auto a, auto dummy) {
                    static_assert(is_same<decltype(i), int64>::value, "index");
                    static_assert(is_same<decltype(j), int64>::value, "index");
                    static_assert(is_same<decltype(a), int64*>::value, "value");
                    static_assert(is_same<decltype(dummy), int64>::value,
                                  "dummy");
                    return (i + 1) * (j + 1);
                },
                [] GKO_KERNEL(auto i, auto j) { return i + j; },
                [] GKO_KERNEL(auto j) { return 2 * j; }, int64{},
                output.get_data(), 2,
                gko::dim<2>{static_cast<size_type>(num_rows),
                            static_cast<size_type>(num_cols)},
                output, move_only_val);

            GKO_ASSERT_ARRAY_EQ(host_ref, output);
        }
    }
}


TEST_F(KernelLaunch, ReductionCol2D)
{
    for (int num_rows : {0, 1, 10, 100, 1000, 10000}) {
        for (int num_cols :
             {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 40, 100, 1000}) {
            SCOPED_TRACE(std::to_string(num_rows) + " rows, " +
                         std::to_string(num_cols) + " cols");
            gko::array<int64> host_ref{exec->get_master(),
                                       static_cast<size_type>(num_cols)};
            gko::array<int64> output{exec, static_cast<size_type>(num_cols)};
            for (int i = 0; i < num_cols; i++) {
                // we are computing 2 * sum {j=0, j<row} (i+1)*(j+1) for each
                // column i
                host_ref.get_data()[i] =
                    static_cast<int64>(num_rows) * (num_rows + 1) * (i + 1);
            }

            gko::kernels::dpcpp::run_kernel_col_reduction(
                exec,
                [] GKO_KERNEL(auto i, auto j, auto a, auto dummy) {
                    static_assert(is_same<decltype(i), int64>::value, "index");
                    static_assert(is_same<decltype(j), int64>::value, "index");
                    static_assert(is_same<decltype(a), int64*>::value, "value");
                    static_assert(is_same<decltype(dummy), int64>::value,
                                  "dummy");
                    return (i + 1) * (j + 1);
                },
                [] GKO_KERNEL(auto i, auto j) { return i + j; },
                [] GKO_KERNEL(auto j) { return j * 2; }, int64{},
                output.get_data(),
                gko::dim<2>{static_cast<size_type>(num_rows),
                            static_cast<size_type>(num_cols)},
                output, move_only_val);

            GKO_ASSERT_ARRAY_EQ(host_ref, output);
        }
    }
}
