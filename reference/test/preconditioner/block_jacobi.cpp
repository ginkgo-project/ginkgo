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

#include <core/preconditioner/block_jacobi.hpp>


#include <algorithm>


#include <gtest/gtest.h>


#include <core/matrix/csr.hpp>
#include <core/matrix/dense.hpp>


namespace {


template <typename ConcreteBlockJacobiFactory>
class BasicBlockJacobiTest : public ::testing::Test {
protected:
    using BjFactory = ConcreteBlockJacobiFactory;
    using Bj = typename ConcreteBlockJacobiFactory::generated_type;
    using Mtx = gko::matrix::Csr<>;
    using Vec = gko::matrix::Dense<>;

    BasicBlockJacobiTest()
        : exec(gko::ReferenceExecutor::create()),
          bj_factory(BjFactory::create(exec, 3)),
          block_pointers(exec, 3),
          mtx(gko::matrix::Csr<>::create(exec, gko::dim<2>{5}, 13))
    {
        block_pointers.get_data()[0] = 0;
        block_pointers.get_data()[1] = 2;
        block_pointers.get_data()[2] = 5;
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
    }

    void SetUp()
    {
        bj_factory->set_block_pointers(block_pointers);
        bj_lin_op = bj_factory->generate(mtx);
        bj = static_cast<Bj *>(bj_lin_op.get());
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
        ASSERT_EQ(a->get_max_block_size(), b->get_max_block_size());
        const auto b_ptr_a = a->get_const_block_pointers();
        const auto b_ptr_b = b->get_const_block_pointers();
        ASSERT_EQ(b_ptr_a[0], b_ptr_b[0]);
        for (int i = 0; i < a->get_num_blocks(); ++i) {
            ASSERT_EQ(b_ptr_a[i + 1], b_ptr_b[i + 1]);
            assert_same_block(
                b_ptr_a[i + 1] - b_ptr_a[i],
                a->get_const_blocks() + b_ptr_a[i] * a->get_stride(),
                a->get_stride(),
                b->get_const_blocks() + b_ptr_b[i] * b->get_stride(),
                b->get_stride());
        }
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<BjFactory> bj_factory;
    gko::Array<gko::int32> block_pointers;
    std::shared_ptr<gko::matrix::Csr<>> mtx;
    std::unique_ptr<gko::LinOp> bj_lin_op;
    Bj *bj;
};


class BlockJacobi
    : public BasicBlockJacobiTest<gko::preconditioner::BlockJacobiFactory<>> {};


TEST_F(BlockJacobi, CanBeCloned)
{
    auto bj_clone = bj->clone();
    assert_same_precond(static_cast<Bj *>(bj_clone.get()), bj);
}


TEST_F(BlockJacobi, CanBeCopied)
{
    gko::Array<gko::int32> empty(exec, 1);
    empty.get_data()[0] = 0;
    bj_factory->set_block_pointers(std::move(empty));
    auto copy = bj_factory->generate(Mtx::create(exec));

    copy->copy_from(bj_lin_op.get());

    assert_same_precond(static_cast<Bj *>(copy.get()), bj);
}


TEST_F(BlockJacobi, CanBeMoved)
{
    auto tmp = bj_lin_op->clone();
    gko::Array<gko::int32> empty(exec, 1);
    empty.get_data()[0] = 0;
    bj_factory->set_block_pointers(std::move(empty));
    auto copy = bj_factory->generate(Mtx::create(exec));

    copy->copy_from(std::move(bj_lin_op));

    assert_same_precond(static_cast<Bj *>(copy.get()),
                        static_cast<Bj *>(tmp.get()));
}


TEST_F(BlockJacobi, CanBeCleared)
{
    bj->clear();

    ASSERT_EQ(bj->get_size(), gko::dim<2>(0, 0));
    ASSERT_EQ(bj->get_num_stored_elements(), 0);
    ASSERT_EQ(bj->get_max_block_size(), 0);
    ASSERT_EQ(bj->get_stride(), 0);
    ASSERT_EQ(bj->get_const_block_pointers(), nullptr);
    ASSERT_EQ(bj->get_const_blocks(), nullptr);
}


#define EXPECT_NONZERO_NEAR(first, second, tol) \
    EXPECT_EQ(first.row, second.row);           \
    EXPECT_EQ(first.column, second.column);     \
    EXPECT_NEAR(first.value, second.value, tol)


TEST_F(BlockJacobi, GeneratesCorrectMatrixData)
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


class AdaptiveBlockJacobi
    : public BasicBlockJacobiTest<
          gko::preconditioner::AdaptiveBlockJacobiFactory<>> {
protected:
    AdaptiveBlockJacobi() : block_precisions(exec, 2)
    {
        block_precisions.get_data()[0] = Bj::single_precision;
        block_precisions.get_data()[1] = Bj::double_precision;
    }

    void SetUp()
    {
        bj_factory->set_block_precisions(block_precisions);
        BasicBlockJacobiTest<BjFactory>::SetUp();
    }

    gko::Array<Bj::precision> block_precisions;

    void assert_same_precond(const Bj *a, const Bj *b)
    {
        ASSERT_EQ(a->get_size()[0], b->get_size()[0]);
        ASSERT_EQ(a->get_size()[1], b->get_size()[1]);
        ASSERT_EQ(a->get_num_blocks(), b->get_num_blocks());
        ASSERT_EQ(a->get_max_block_size(), b->get_max_block_size());
        const auto b_ptr_a = a->get_const_block_pointers();
        const auto b_ptr_b = b->get_const_block_pointers();
        const auto b_prec_a = a->get_const_block_precisions();
        const auto b_prec_b = b->get_const_block_precisions();
        ASSERT_EQ(b_ptr_a[0], b_ptr_b[0]);
        for (int i = 0; i < a->get_num_blocks(); ++i) {
            ASSERT_EQ(b_prec_a[i], b_prec_b[i]);
            ASSERT_EQ(b_ptr_a[i + 1], b_ptr_b[i + 1]);
            // TODO: take into account different precisions
            assert_same_block(
                b_ptr_a[i + 1] - b_ptr_a[i],
                a->get_const_blocks() + b_ptr_a[i] * a->get_stride(),
                a->get_stride(),
                b->get_const_blocks() + b_ptr_b[i] * b->get_stride(),
                b->get_stride());
        }
    }
};


// TODO: take into account different precisions in the following tests


TEST_F(AdaptiveBlockJacobi, CanBeCloned)
{
    auto bj_clone = bj->clone();
    assert_same_precond(static_cast<Bj *>(bj_clone.get()), bj);
}


TEST_F(AdaptiveBlockJacobi, CanBeCopied)
{
    gko::Array<gko::int32> empty(exec, 1);
    empty.get_data()[0] = 0;
    bj_factory->set_block_pointers(std::move(empty));
    bj_factory->set_block_precisions(block_precisions);
    auto copy = bj_factory->generate(Mtx::create(exec));

    copy->copy_from(bj_lin_op.get());

    assert_same_precond(static_cast<Bj *>(copy.get()), bj);
}


TEST_F(AdaptiveBlockJacobi, CanBeMoved)
{
    auto tmp = bj_lin_op->clone();
    gko::Array<gko::int32> empty(exec, 1);
    empty.get_data()[0] = 0;
    bj_factory->set_block_pointers(std::move(empty));
    bj_factory->set_block_precisions(block_precisions);
    auto copy = bj_factory->generate(Mtx::create(exec));

    copy->copy_from(std::move(bj_lin_op));

    assert_same_precond(static_cast<Bj *>(copy.get()),
                        static_cast<Bj *>(tmp.get()));
}


TEST_F(AdaptiveBlockJacobi, CanBeCleared)
{
    bj->clear();

    ASSERT_EQ(bj->get_size(), gko::dim<2>(0, 0));
    ASSERT_EQ(bj->get_num_stored_elements(), 0);
    ASSERT_EQ(bj->get_max_block_size(), 0);
    ASSERT_EQ(bj->get_stride(), 0);
    ASSERT_EQ(bj->get_const_block_pointers(), nullptr);
    ASSERT_EQ(bj->get_const_block_precisions(), nullptr);
    ASSERT_EQ(bj->get_const_blocks(), nullptr);
}


TEST_F(AdaptiveBlockJacobi, GeneratesCorrectMatrixData)
{
    using tpl = gko::matrix_data<>::nonzero_type;
    gko::matrix_data<> data;

    bj->write(data);

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
