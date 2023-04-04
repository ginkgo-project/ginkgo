/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include <ginkgo/core/preconditioner/jacobi.hpp>


#include <algorithm>


#include <gtest/gtest.h>


#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/extended_float.hpp"
#include "core/preconditioner/jacobi_utils.hpp"
#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class Jacobi : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Bj = gko::preconditioner::Jacobi<value_type, index_type>;
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;

    Jacobi()
        : exec(gko::ReferenceExecutor::create()),
          bj_factory(Bj::build().with_max_block_size(3u).on(exec)),
          scalar_j_factory(Bj::build().with_max_block_size(1u).on(exec)),
          block_pointers(exec, 3),
          block_precisions(exec, 2),
          mtx(Mtx::create(exec, gko::dim<2>{5}, 13))
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
        init_array<index_type>(mtx->get_row_ptrs(), {0, 3, 5, 7, 10, 13});
        init_array<index_type>(mtx->get_col_idxs(),
                               {0, 1, 4, 0, 1, 2, 3, 2, 3, 4, 0, 3, 4});
        init_array<value_type>(mtx->get_values(),
                               {4.0, -2.0, -2.0, -1.0, 4.0, 4.0, -2.0, -1.0,
                                4.0, -2.0, -1.0, -1.0, 4.0});
        bj_factory = Bj::build()
                         .with_max_block_size(3u)
                         .with_block_pointers(block_pointers)
                         .on(exec);
        adaptive_bj_factory = Bj::build()
                                  .with_max_block_size(17u)
                                  // make sure group size is 1
                                  .with_block_pointers(block_pointers)
                                  .with_storage_optimization(block_precisions)
                                  .on(exec);

        bj = bj_factory->generate(mtx);
        adaptive_bj = adaptive_bj_factory->generate(mtx);
    }

    template <typename T>
    void init_array(T* arr, std::initializer_list<T> vals)
    {
        std::copy(std::begin(vals), std::end(vals), arr);
    }

    template <typename ValueType>
    void assert_same_block(gko::size_type block_size, const ValueType* ptr_a,
                           gko::size_type stride_a, const ValueType* ptr_b,
                           gko::size_type stride_b)
    {
        for (int i = 0; i < block_size; ++i) {
            for (int j = 0; j < block_size; ++j) {
                EXPECT_EQ(static_cast<value_type>(ptr_a[i * stride_a + j]),
                          static_cast<value_type>(ptr_b[i * stride_b + j]))
                    << "Mismatch at position (" << i << ", " << j << ")";
            }
        }
    }

    void assert_same_precond(gko::ptr_param<const Bj> a,
                             gko::ptr_param<const Bj> b)
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
                value_type, prec_a,
                assert_same_block(
                    b_ptr_a[i + 1] - b_ptr_a[i],
                    reinterpret_cast<const resolved_precision*>(
                        a->get_blocks() + scheme.get_group_offset(i)) +
                        scheme.get_block_offset(i),
                    scheme.get_stride(),
                    reinterpret_cast<const resolved_precision*>(
                        a->get_blocks() + scheme.get_group_offset(i)) +
                        scheme.get_block_offset(i),
                    scheme.get_stride()));
        }
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<typename Bj::Factory> bj_factory;
    std::unique_ptr<typename Bj::Factory> scalar_j_factory;
    std::unique_ptr<typename Bj::Factory> adaptive_bj_factory;
    gko::array<index_type> block_pointers;
    gko::array<gko::precision_reduction> block_precisions;
    std::shared_ptr<Mtx> mtx;
    std::unique_ptr<Bj> bj;
    std::unique_ptr<Bj> adaptive_bj;
};

TYPED_TEST_SUITE(Jacobi, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(Jacobi, GeneratesCorrectStorageScheme)
{
    auto scheme = this->bj->get_storage_scheme();

    ASSERT_EQ(scheme.group_power, 3);  // 8 3-by-3 blocks fit into 32-wide group
    ASSERT_EQ(scheme.block_offset, 3);
    ASSERT_EQ(scheme.group_offset, 8 * 3 * 3);
}


TYPED_TEST(Jacobi, CanBeCloned)
{
    auto bj_clone = clone(this->bj);

    this->assert_same_precond(bj_clone, this->bj);
}


TYPED_TEST(Jacobi, CanBeClonedWithAdaptvePrecision)
{
    auto bj_clone = clone(this->adaptive_bj);
    this->assert_same_precond(bj_clone, this->adaptive_bj);
}


TYPED_TEST(Jacobi, CanBeCopied)
{
    using Bj = typename TestFixture::Bj;
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    gko::array<index_type> empty(this->exec, 1);
    empty.get_data()[0] = 0;
    auto copy = Bj::build()
                    .with_block_pointers(empty)
                    .on(this->exec)
                    ->generate(Mtx::create(this->exec));

    copy->copy_from(this->bj);

    this->assert_same_precond(copy, this->bj);
}


TYPED_TEST(Jacobi, CanBeCopiedWithAdaptivePrecision)
{
    using Bj = typename TestFixture::Bj;
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    gko::array<index_type> empty(this->exec, 1);
    empty.get_data()[0] = 0;
    auto copy = Bj::build()
                    .with_block_pointers(empty)
                    .on(this->exec)
                    ->generate(Mtx::create(this->exec));

    copy->copy_from(this->adaptive_bj);

    this->assert_same_precond(copy, this->adaptive_bj);
}


TYPED_TEST(Jacobi, CanBeMoved)
{
    using Bj = typename TestFixture::Bj;
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto tmp = clone(this->bj);
    gko::array<index_type> empty(this->exec, 1);
    empty.get_data()[0] = 0;
    auto copy = Bj::build()
                    .with_block_pointers(empty)
                    .on(this->exec)
                    ->generate(Mtx::create(this->exec));

    copy->move_from(this->bj);

    this->assert_same_precond(copy.get(), tmp);
}


TYPED_TEST(Jacobi, CanBeMovedWithAdaptivePrecision)
{
    using Bj = typename TestFixture::Bj;
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto tmp = clone(this->adaptive_bj);
    gko::array<index_type> empty(this->exec, 1);
    empty.get_data()[0] = 0;
    auto copy = Bj::build()
                    .with_block_pointers(empty)
                    .on(this->exec)
                    ->generate(Mtx::create(this->exec));

    copy->move_from(this->adaptive_bj);

    this->assert_same_precond(copy.get(), tmp);
}


TYPED_TEST(Jacobi, CanBeCleared)
{
    this->bj->clear();

    ASSERT_EQ(this->bj->get_size(), gko::dim<2>(0, 0));
    ASSERT_EQ(this->bj->get_num_stored_elements(), 0);
    ASSERT_EQ(this->bj->get_parameters().max_block_size, 32);
    ASSERT_EQ(this->bj->get_parameters().block_pointers.get_const_data(),
              nullptr);
    ASSERT_EQ(this->bj->get_blocks(), nullptr);
}


TYPED_TEST(Jacobi, CanBeClearedWithAdaptivePrecision)
{
    this->adaptive_bj->clear();

    ASSERT_EQ(this->adaptive_bj->get_size(), gko::dim<2>(0, 0));
    ASSERT_EQ(this->adaptive_bj->get_num_stored_elements(), 0);
    ASSERT_EQ(this->adaptive_bj->get_parameters().max_block_size, 32);
    ASSERT_EQ(
        this->adaptive_bj->get_parameters().block_pointers.get_const_data(),
        nullptr);
    ASSERT_EQ(this->adaptive_bj->get_parameters()
                  .storage_optimization.block_wise.get_const_data(),
              nullptr);
    ASSERT_EQ(this->adaptive_bj->get_blocks(), nullptr);
}


TYPED_TEST(Jacobi, ScalarJacobiConvertsToDense)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Bj = typename TestFixture::Bj;
    gko::matrix_data<value_type, index_type> data;
    auto csr = gko::share(
        gko::matrix::Csr<value_type, index_type>::create(this->exec));
    csr->copy_from(this->mtx);
    auto scalar_j = this->scalar_j_factory->generate(csr);

    auto dense_j = gko::matrix::Dense<value_type>::create(this->exec);
    dense_j->copy_from(scalar_j);
    auto j_val = scalar_j->get_blocks();

    for (auto i = 0; i < dense_j->get_size()[0]; ++i) {
        for (auto j = 0; j < dense_j->get_size()[1]; ++j) {
            if (i == j) {
                EXPECT_EQ(dense_j->at(i, j), j_val[j]);
            } else {
                EXPECT_EQ(dense_j->at(i, j), value_type{0.0});
            }
        }
    }
}


TYPED_TEST(Jacobi, ScalarJacobiCanBeTransposed)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Bj = typename TestFixture::Bj;
    gko::matrix_data<value_type, index_type> data;
    auto csr = gko::share(
        gko::matrix::Csr<value_type, index_type>::create(this->exec));
    csr->copy_from(this->mtx);
    auto scalar_j = this->scalar_j_factory->generate(csr);

    auto dense_j = gko::matrix::Dense<value_type>::create(this->exec);
    auto t_j = scalar_j->transpose();
    auto trans_j = gko::as<Bj>(t_j.get())->get_blocks();
    auto scal_j = scalar_j->get_blocks();

    ASSERT_EQ(scalar_j->get_size(), gko::dim<2>(5, 5));
    ASSERT_EQ(scalar_j->get_num_stored_elements(), 5);
    ASSERT_EQ(scalar_j->get_parameters().max_block_size, 1);
    ASSERT_EQ(scalar_j->get_num_blocks(), 5);
    ASSERT_EQ(scalar_j->get_parameters().block_pointers.get_const_data(),
              nullptr);
    EXPECT_EQ(trans_j[0], scal_j[0]);
    EXPECT_EQ(trans_j[1], scal_j[1]);
    EXPECT_EQ(trans_j[2], scal_j[2]);
    EXPECT_EQ(trans_j[3], scal_j[3]);
    EXPECT_EQ(trans_j[4], scal_j[4]);
}


template <typename T>
void init_array(T* arr, std::initializer_list<T> vals)
{
    std::copy(std::begin(vals), std::end(vals), arr);
}


TEST(Jacobi, ScalarJacobiCanBeConjTransposed)
{
    using value_type = std::complex<double>;
    using vt = value_type;
    using index_type = int;
    using Bj = gko::preconditioner::Jacobi<value_type, index_type>;
    gko::matrix_data<value_type, index_type> data;
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    auto exec = gko::ReferenceExecutor::create();
    auto csr = gko::share(Mtx::create(exec, gko::dim<2>(5, 5), 13));
    auto scalar_j_factory = Bj::build().with_max_block_size(1u).on(exec);
    init_array<index_type>(csr->get_row_ptrs(), {0, 3, 5, 7, 10, 13});
    init_array<index_type>(csr->get_col_idxs(),
                           {0, 1, 4, 0, 1, 2, 3, 2, 3, 4, 0, 3, 4});
    init_array<value_type>(
        csr->get_values(),
        {vt(4.0, 1), vt(-2.0), vt(-2.0), vt(-1.0), vt(4.0, -1), vt(4.0),
         vt(-2.0), vt(-1.0), vt(4.0), vt(-2.0), vt(-1.0), vt(-1.0), vt(4.0)});
    auto scalar_j = scalar_j_factory->generate(csr);

    auto t_j = scalar_j->conj_transpose();
    auto trans_j = gko::as<Bj>(t_j.get())->get_blocks();
    auto scal_j = scalar_j->get_blocks();

    ASSERT_EQ(scalar_j->get_size(), gko::dim<2>(5, 5));
    ASSERT_EQ(scalar_j->get_num_stored_elements(), 5);
    ASSERT_EQ(scalar_j->get_parameters().max_block_size, 1);
    ASSERT_EQ(scalar_j->get_num_blocks(), 5);
    ASSERT_EQ(scalar_j->get_parameters().block_pointers.get_const_data(),
              nullptr);
    EXPECT_EQ(trans_j[0], gko::conj(scal_j[0]));
    EXPECT_EQ(trans_j[1], gko::conj(scal_j[1]));
    EXPECT_EQ(trans_j[2], gko::conj(scal_j[2]));
    EXPECT_EQ(trans_j[3], gko::conj(scal_j[3]));
    EXPECT_EQ(trans_j[4], gko::conj(scal_j[4]));
}


#define GKO_EXPECT_NONZERO_NEAR(first, second, tol) \
    EXPECT_EQ(first.row, second.row);               \
    EXPECT_EQ(first.column, second.column);         \
    GKO_EXPECT_NEAR(first.value, second.value, tol)


template <typename ValueType, typename IndexType>
void assert_matrix_data(gko::matrix_data<ValueType, IndexType> data)
{
    using tpl = typename decltype(data)::nonzero_type;
    auto tol = r<ValueType>::value;
    ASSERT_EQ(data.size, gko::dim<2>{5});
    ASSERT_EQ(data.nonzeros.size(), 13);
    GKO_EXPECT_NONZERO_NEAR(data.nonzeros[0], tpl(0, 0, 4.0 / 14), tol);
    GKO_EXPECT_NONZERO_NEAR(data.nonzeros[1], tpl(0, 1, 2.0 / 14), tol);
    GKO_EXPECT_NONZERO_NEAR(data.nonzeros[2], tpl(1, 0, 1.0 / 14), tol);
    GKO_EXPECT_NONZERO_NEAR(data.nonzeros[3], tpl(1, 1, 4.0 / 14), tol);
    GKO_EXPECT_NONZERO_NEAR(data.nonzeros[4], tpl(2, 2, 14.0 / 48), tol);
    GKO_EXPECT_NONZERO_NEAR(data.nonzeros[5], tpl(2, 3, 8.0 / 48), tol);
    GKO_EXPECT_NONZERO_NEAR(data.nonzeros[6], tpl(2, 4, 4.0 / 48), tol);
    GKO_EXPECT_NONZERO_NEAR(data.nonzeros[7], tpl(3, 2, 4.0 / 48), tol);
    GKO_EXPECT_NONZERO_NEAR(data.nonzeros[8], tpl(3, 3, 16.0 / 48), tol);
    GKO_EXPECT_NONZERO_NEAR(data.nonzeros[9], tpl(3, 4, 8.0 / 48), tol);
    GKO_EXPECT_NONZERO_NEAR(data.nonzeros[10], tpl(4, 2, 1.0 / 48), tol);
    GKO_EXPECT_NONZERO_NEAR(data.nonzeros[11], tpl(4, 3, 4.0 / 48), tol);
    GKO_EXPECT_NONZERO_NEAR(data.nonzeros[12], tpl(4, 4, 14.0 / 48), tol);
}


TYPED_TEST(Jacobi, GeneratesCorrectMatrixData)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    gko::matrix_data<value_type, index_type> data;

    this->bj->write(data);

    assert_matrix_data(data);
}


TYPED_TEST(Jacobi, ScalarJacobiGeneratesCorrectMatrixData)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Bj = typename TestFixture::Bj;
    gko::matrix_data<value_type, index_type> data;
    using tpl = typename decltype(data)::nonzero_type;
    auto csr = gko::share(
        gko::matrix::Csr<value_type, index_type>::create(this->exec));
    csr->copy_from(this->mtx);
    auto scalar_j = this->scalar_j_factory->generate(csr);

    scalar_j->write(data);

    auto tol = r<value_type>::value;
    ASSERT_EQ(data.size, gko::dim<2>{5});
    ASSERT_EQ(data.nonzeros.size(), 5);
    GKO_EXPECT_NONZERO_NEAR(data.nonzeros[0], tpl(0, 0, 1 / 4.0), tol);
    GKO_EXPECT_NONZERO_NEAR(data.nonzeros[1], tpl(1, 1, 1 / 4.0), tol);
    GKO_EXPECT_NONZERO_NEAR(data.nonzeros[2], tpl(2, 2, 1 / 4.0), tol);
    GKO_EXPECT_NONZERO_NEAR(data.nonzeros[3], tpl(3, 3, 1 / 4.0), tol);
    GKO_EXPECT_NONZERO_NEAR(data.nonzeros[4], tpl(4, 4, 1 / 4.0), tol);
}


TYPED_TEST(Jacobi, GeneratesCorrectMatrixDataWithAdaptivePrecision)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using tpl = typename gko::matrix_data<value_type, index_type>::nonzero_type;
    gko::matrix_data<value_type, index_type> data;
    auto tol = r<value_type>::value;
    auto half_tol = std::sqrt(r<value_type>::value);

    this->adaptive_bj->write(data);

    ASSERT_EQ(data.size, gko::dim<2>{5});
    ASSERT_EQ(data.nonzeros.size(), 13);
    GKO_EXPECT_NONZERO_NEAR(data.nonzeros[0], tpl(0, 0, 4.0 / 14), half_tol);
    GKO_EXPECT_NONZERO_NEAR(data.nonzeros[1], tpl(0, 1, 2.0 / 14), half_tol);
    GKO_EXPECT_NONZERO_NEAR(data.nonzeros[2], tpl(1, 0, 1.0 / 14), half_tol);
    GKO_EXPECT_NONZERO_NEAR(data.nonzeros[3], tpl(1, 1, 4.0 / 14), half_tol);
    GKO_EXPECT_NONZERO_NEAR(data.nonzeros[4], tpl(2, 2, 14.0 / 48), tol);
    GKO_EXPECT_NONZERO_NEAR(data.nonzeros[5], tpl(2, 3, 8.0 / 48), tol);
    GKO_EXPECT_NONZERO_NEAR(data.nonzeros[6], tpl(2, 4, 4.0 / 48), tol);
    GKO_EXPECT_NONZERO_NEAR(data.nonzeros[7], tpl(3, 2, 4.0 / 48), tol);
    GKO_EXPECT_NONZERO_NEAR(data.nonzeros[8], tpl(3, 3, 16.0 / 48), tol);
    GKO_EXPECT_NONZERO_NEAR(data.nonzeros[9], tpl(3, 4, 8.0 / 48), tol);
    GKO_EXPECT_NONZERO_NEAR(data.nonzeros[10], tpl(4, 2, 1.0 / 48), tol);
    GKO_EXPECT_NONZERO_NEAR(data.nonzeros[11], tpl(4, 3, 4.0 / 48), tol);
    GKO_EXPECT_NONZERO_NEAR(data.nonzeros[12], tpl(4, 4, 14.0 / 48), tol);
}


TYPED_TEST(Jacobi, ScalarJacobiGeneratesOnDifferentPrecision)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using next_type = next_precision<value_type>;
    using Bj = typename TestFixture::Bj;
    auto csr =
        gko::share(gko::matrix::Csr<next_type, index_type>::create(this->exec));
    csr->copy_from(this->mtx);
    std::shared_ptr<Bj> bj{};

    ASSERT_NO_THROW(bj = this->scalar_j_factory->generate(csr));
    ASSERT_EQ(bj->get_num_blocks(), 5u);
    ASSERT_EQ(bj->get_blocks()[0], value_type{0.25});
    ASSERT_EQ(bj->get_blocks()[1], value_type{0.25});
    ASSERT_EQ(bj->get_blocks()[2], value_type{0.25});
    ASSERT_EQ(bj->get_blocks()[3], value_type{0.25});
    ASSERT_EQ(bj->get_blocks()[4], value_type{0.25});
}


}  // namespace
