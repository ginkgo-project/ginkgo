/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <core/preconditioner/jacobi.hpp>


#include <algorithm>


#include <gtest/gtest.h>


#include <core/base/extended_float.hpp>
#include <core/matrix/csr.hpp>
#include <core/matrix/dense.hpp>
#include <core/preconditioner/jacobi_utils.hpp>


namespace {


class Jacobi : public ::testing::Test {
protected:
    using Bj = gko::preconditioner::Jacobi<>;
    using Mtx = gko::matrix::Csr<>;
    using Vec = gko::matrix::Dense<>;

    Jacobi()
        : exec(gko::ReferenceExecutor::create()),
          bj_factory(Bj::build().with_max_block_size(3u).on(exec)),
          block_pointers(exec, 3),
          block_precisions(exec, 2),
          mtx(gko::matrix::Csr<>::create(exec, gko::dim<2>{5}, 13))
    {
        block_pointers.get_data()[0] = 0;
        block_pointers.get_data()[1] = 2;
        block_pointers.get_data()[2] = 5;
        block_precisions.get_data()[0] = gko::precision_reduction(0, 1);
        block_precisions.get_data()[1] = gko::precision_reduction(0, 0);
        /* test matrix:
            4  -2 |        -2
           -1   4 |
           -------+----------
                  | 4  -2
                  |-1   4  -2
           -1     |    -1   4
         */
        init_array(mtx->get_row_ptrs(), {0, 3, 5, 7, 10, 13});
        init_array(mtx->get_col_idxs(),
                   {0, 1, 4, 0, 1, 2, 3, 2, 3, 4, 0, 3, 4});
        init_array(mtx->get_values(), {4.0, -2.0, -2.0, -1.0, 4.0, 4.0, -2.0,
                                       -1.0, 4.0, -2.0, -1.0, -1.0, 4.0});
        bj_factory = Bj::build()
                         .with_max_block_size(3u)
                         .with_block_pointers(block_pointers)
                         .on(exec);
        adaptive_bj_factory = Bj::build()
                                  .with_max_block_size(3u)
                                  .with_block_pointers(block_pointers)
                                  .with_storage_optimization(block_precisions)
                                  .on(exec);

        bj = bj_factory->generate(mtx);
        adaptive_bj = adaptive_bj_factory->generate(mtx);
    }

    template <typename T>
    void init_array(T *arr, std::initializer_list<T> vals)
    {
        for (auto elem : vals) {
            *(arr++) = elem;
        }
    }

    template <typename ValueType>
    void assert_same_block(gko::size_type block_size, const ValueType *ptr_a,
                           gko::size_type stride_a, const ValueType *ptr_b,
                           gko::size_type stride_b)
    {
        for (int i = 0; i < block_size; ++i) {
            for (int j = 0; j < block_size; ++j) {
                EXPECT_EQ(ptr_a[i * stride_a + j], ptr_b[i * stride_b + j])
                    << "Mismatch at position (" << i << ", " << j << ")";
            }
        }
    }

    void assert_same_precond(const Bj *a, const Bj *b)
    {
        ASSERT_EQ(a->get_size()[0], b->get_size()[0]);
        ASSERT_EQ(a->get_size()[1], b->get_size()[1]);
        ASSERT_EQ(a->get_num_blocks(), b->get_num_blocks());
        ASSERT_EQ(a->get_parameters().max_block_size,
                  b->get_parameters().max_block_size);
        const auto b_ptr_a =
            a->get_parameters().block_pointers.get_const_data();
        const auto b_ptr_b =
            b->get_parameters().block_pointers.get_const_data();
        const auto b_prec_a =
            a->get_parameters()
                .storage_optimization.block_wise.get_const_data();
        const auto b_prec_b =
            b->get_parameters()
                .storage_optimization.block_wise.get_const_data();
        ASSERT_EQ(b_ptr_a[0], b_ptr_b[0]);
        for (int i = 0; i < a->get_num_blocks(); ++i) {
            ASSERT_EQ(b_ptr_a[i + 1], b_ptr_b[i + 1]);
            const auto prec_a =
                b_prec_a ? b_prec_a[i] : gko::precision_reduction();
            const auto prec_b =
                b_prec_b ? b_prec_b[i] : gko::precision_reduction();
            ASSERT_EQ(prec_a, prec_b);
            auto scheme = a->get_storage_scheme();
            GKO_PRECONDITIONER_JACOBI_RESOLVE_PRECISION(
                Bj::value_type, prec_a,
                assert_same_block(
                    b_ptr_a[i + 1] - b_ptr_a[i],
                    a->get_blocks() + scheme.get_global_block_offset(i),
                    scheme.get_stride(),
                    b->get_blocks() + scheme.get_global_block_offset(i),
                    scheme.get_stride()));
        }
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Bj::Factory> bj_factory;
    std::unique_ptr<Bj::Factory> adaptive_bj_factory;
    gko::Array<gko::int32> block_pointers;
    gko::Array<gko::precision_reduction> block_precisions;
    std::shared_ptr<gko::matrix::Csr<>> mtx;
    std::unique_ptr<Bj> bj;
    std::unique_ptr<Bj> adaptive_bj;
};


TEST_F(Jacobi, GeneratesCorrectStorageScheme)
{
    auto scheme = bj->get_storage_scheme();

    ASSERT_EQ(scheme.group_power, 3);  // 8 3-by-3 blocks fit into 32-wide group
    ASSERT_EQ(scheme.block_offset, 3);
    ASSERT_EQ(scheme.group_offset, 8 * 3 * 3);
}


TEST_F(Jacobi, CanBeCloned)
{
    auto bj_clone = clone(bj);

    assert_same_precond(lend(bj_clone), lend(bj));
}


TEST_F(Jacobi, CanBeClonedWithAdaptvePrecision)
{
    auto bj_clone = clone(adaptive_bj);
    assert_same_precond(lend(bj_clone), lend(adaptive_bj));
}


TEST_F(Jacobi, CanBeCopied)
{
    gko::Array<gko::int32> empty(exec, 1);
    empty.get_data()[0] = 0;
    auto copy = Bj::build().with_block_pointers(empty).on(exec)->generate(
        Mtx::create(exec));

    copy->copy_from(lend(bj));

    assert_same_precond(lend(copy), lend(bj));
}


TEST_F(Jacobi, CanBeCopiedWithAdaptivePrecision)
{
    gko::Array<gko::int32> empty(exec, 1);
    empty.get_data()[0] = 0;
    auto copy = Bj::build().with_block_pointers(empty).on(exec)->generate(
        Mtx::create(exec));

    copy->copy_from(lend(adaptive_bj));

    assert_same_precond(lend(copy), lend(adaptive_bj));
}


TEST_F(Jacobi, CanBeMoved)
{
    auto tmp = clone(bj);
    gko::Array<gko::int32> empty(exec, 1);
    empty.get_data()[0] = 0;
    auto copy = Bj::build().with_block_pointers(empty).on(exec)->generate(
        Mtx::create(exec));

    copy->copy_from(give(bj));

    assert_same_precond(lend(copy), lend(tmp));
}


TEST_F(Jacobi, CanBeMovedWithAdaptivePrecision)
{
    auto tmp = clone(adaptive_bj);
    gko::Array<gko::int32> empty(exec, 1);
    empty.get_data()[0] = 0;
    auto copy = Bj::build().with_block_pointers(empty).on(exec)->generate(
        Mtx::create(exec));

    copy->copy_from(give(adaptive_bj));

    assert_same_precond(lend(copy), lend(tmp));
}


TEST_F(Jacobi, CanBeCleared)
{
    bj->clear();

    ASSERT_EQ(bj->get_size(), gko::dim<2>(0, 0));
    ASSERT_EQ(bj->get_num_stored_elements(), 0);
    ASSERT_EQ(bj->get_parameters().max_block_size, 32);
    ASSERT_EQ(bj->get_parameters().block_pointers.get_const_data(), nullptr);
    ASSERT_EQ(bj->get_blocks(), nullptr);
}


TEST_F(Jacobi, CanBeClearedWithAdaptivePrecision)
{
    adaptive_bj->clear();

    ASSERT_EQ(adaptive_bj->get_size(), gko::dim<2>(0, 0));
    ASSERT_EQ(adaptive_bj->get_num_stored_elements(), 0);
    ASSERT_EQ(adaptive_bj->get_parameters().max_block_size, 32);
    ASSERT_EQ(adaptive_bj->get_parameters().block_pointers.get_const_data(),
              nullptr);
    ASSERT_EQ(adaptive_bj->get_parameters()
                  .storage_optimization.block_wise.get_const_data(),
              nullptr);
    ASSERT_EQ(adaptive_bj->get_blocks(), nullptr);
}


#define EXPECT_NONZERO_NEAR(first, second, tol) \
    EXPECT_EQ(first.row, second.row);           \
    EXPECT_EQ(first.column, second.column);     \
    EXPECT_NEAR(first.value, second.value, tol)


TEST_F(Jacobi, GeneratesCorrectMatrixData)
{
    using tpl = gko::matrix_data<>::nonzero_type;
    gko::matrix_data<> data;

    bj->write(data);

    ASSERT_EQ(data.size, gko::dim<2>{5});
    ASSERT_EQ(data.nonzeros.size(), 13);
    EXPECT_NONZERO_NEAR(data.nonzeros[0], tpl(0, 0, 4.0 / 14), 1e-14);
    EXPECT_NONZERO_NEAR(data.nonzeros[1], tpl(0, 1, 2.0 / 14), 1e-14);
    EXPECT_NONZERO_NEAR(data.nonzeros[2], tpl(1, 0, 1.0 / 14), 1e-14);
    EXPECT_NONZERO_NEAR(data.nonzeros[3], tpl(1, 1, 4.0 / 14), 1e-14);
    EXPECT_NONZERO_NEAR(data.nonzeros[4], tpl(2, 2, 14.0 / 48), 1e-14);
    EXPECT_NONZERO_NEAR(data.nonzeros[5], tpl(2, 3, 8.0 / 48), 1e-14);
    EXPECT_NONZERO_NEAR(data.nonzeros[6], tpl(2, 4, 4.0 / 48), 1e-14);
    EXPECT_NONZERO_NEAR(data.nonzeros[7], tpl(3, 2, 4.0 / 48), 1e-14);
    EXPECT_NONZERO_NEAR(data.nonzeros[8], tpl(3, 3, 16.0 / 48), 1e-14);
    EXPECT_NONZERO_NEAR(data.nonzeros[9], tpl(3, 4, 8.0 / 48), 1e-14);
    EXPECT_NONZERO_NEAR(data.nonzeros[10], tpl(4, 2, 1.0 / 48), 1e-14);
    EXPECT_NONZERO_NEAR(data.nonzeros[11], tpl(4, 3, 4.0 / 48), 1e-14);
    EXPECT_NONZERO_NEAR(data.nonzeros[12], tpl(4, 4, 14.0 / 48), 1e-14);
}


TEST_F(Jacobi, GeneratesCorrectMatrixDataWithAdaptivePrecision)
{
    using tpl = gko::matrix_data<>::nonzero_type;
    gko::matrix_data<> data;

    adaptive_bj->write(data);

    ASSERT_EQ(data.size, gko::dim<2>{5});
    ASSERT_EQ(data.nonzeros.size(), 13);
    EXPECT_NONZERO_NEAR(data.nonzeros[0], tpl(0, 0, 4.0 / 14), 1e-7);
    EXPECT_NONZERO_NEAR(data.nonzeros[1], tpl(0, 1, 2.0 / 14), 1e-7);
    EXPECT_NONZERO_NEAR(data.nonzeros[2], tpl(1, 0, 1.0 / 14), 1e-7);
    EXPECT_NONZERO_NEAR(data.nonzeros[3], tpl(1, 1, 4.0 / 14), 1e-7);
    EXPECT_NONZERO_NEAR(data.nonzeros[4], tpl(2, 2, 14.0 / 48), 1e-14);
    EXPECT_NONZERO_NEAR(data.nonzeros[5], tpl(2, 3, 8.0 / 48), 1e-14);
    EXPECT_NONZERO_NEAR(data.nonzeros[6], tpl(2, 4, 4.0 / 48), 1e-14);
    EXPECT_NONZERO_NEAR(data.nonzeros[7], tpl(3, 2, 4.0 / 48), 1e-14);
    EXPECT_NONZERO_NEAR(data.nonzeros[8], tpl(3, 3, 16.0 / 48), 1e-14);
    EXPECT_NONZERO_NEAR(data.nonzeros[9], tpl(3, 4, 8.0 / 48), 1e-14);
    EXPECT_NONZERO_NEAR(data.nonzeros[10], tpl(4, 2, 1.0 / 48), 1e-14);
    EXPECT_NONZERO_NEAR(data.nonzeros[11], tpl(4, 3, 4.0 / 48), 1e-14);
    EXPECT_NONZERO_NEAR(data.nonzeros[12], tpl(4, 4, 14.0 / 48), 1e-14);
}


}  // namespace
