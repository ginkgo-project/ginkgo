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

#include <algorithm>
#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/identity.hpp>


#include "core/components/prefix_sum_kernels.hpp"
#include "core/factorization/cholesky_kernels.hpp"
#include "core/factorization/elimination_forest.hpp"
#include "core/factorization/symbolic.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"
#include "matrices/config.hpp"


namespace {


template <typename ValueIndexType>
class Cholesky : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using matrix_type = gko::matrix::Csr<value_type, index_type>;

    Cholesky() : ref(gko::ReferenceExecutor::create()), tmp{ref} {}

    std::unique_ptr<matrix_type> combined_factor(const matrix_type* l_factor)
    {
        auto one = gko::initialize<gko::matrix::Dense<value_type>>(
            {gko::one<value_type>()}, ref);
        auto id = gko::matrix::Identity<value_type>::create(
            ref, l_factor->get_size()[0]);
        auto result = gko::as<matrix_type>(l_factor->transpose());
        l_factor->apply(one.get(), id.get(), one.get(), result.get());
        return result;
    }

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    gko::array<index_type> tmp;
};

TYPED_TEST_SUITE(Cholesky, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(Cholesky, KernelSymbolicCountExample)
{
    using matrix_type = typename TestFixture::matrix_type;
    using index_type = typename TestFixture::index_type;
    auto mtx = gko::initialize<typename TestFixture::matrix_type>(
        {{1, 0, 1, 0, 0, 0, 0, 1, 0, 0},
         {0, 1, 0, 1, 0, 0, 0, 0, 0, 1},
         {1, 0, 1, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 1, 0, 0, 0, 0, 1, 1},
         {0, 1, 0, 0, 1, 0, 0, 0, 1, 1},
         {0, 0, 0, 0, 0, 1, 0, 1, 0, 0},
         {0, 0, 1, 0, 0, 1, 1, 0, 0, 0},
         {1, 0, 0, 0, 0, 1, 0, 1, 1, 1},
         {0, 0, 0, 1, 1, 0, 0, 1, 1, 0},
         {0, 1, 0, 1, 1, 0, 0, 1, 0, 1}},
        this->ref);
    std::unique_ptr<gko::factorization::elimination_forest<index_type>> forest;
    gko::factorization::compute_elim_forest(mtx.get(), forest);
    gko::array<index_type> row_nnz{this->ref, 10};

    gko::kernels::reference::cholesky::cholesky_symbolic_count(
        this->ref, mtx.get(), *forest, row_nnz.get_data(), this->tmp);

    GKO_ASSERT_ARRAY_EQ(row_nnz, I<index_type>({1, 1, 2, 1, 2, 1, 3, 5, 4, 6}));
}


TYPED_TEST(Cholesky, KernelSymbolicFactorizeExample)
{
    using matrix_type = typename TestFixture::matrix_type;
    using index_type = typename TestFixture::index_type;
    auto mtx = gko::initialize<typename TestFixture::matrix_type>(
        {{1, 0, 1, 0, 0, 0, 0, 1, 0, 0},
         {0, 1, 0, 1, 0, 0, 0, 0, 0, 1},
         {1, 0, 1, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 1, 0, 0, 0, 0, 1, 1},
         {0, 1, 0, 0, 1, 0, 0, 0, 1, 1},
         {0, 0, 0, 0, 0, 1, 0, 1, 0, 0},
         {0, 0, 1, 0, 0, 1, 1, 0, 0, 0},
         {1, 0, 0, 0, 0, 1, 0, 1, 1, 1},
         {0, 0, 0, 1, 1, 0, 0, 1, 1, 0},
         {0, 1, 0, 1, 1, 0, 0, 1, 0, 1}},
        this->ref);
    std::unique_ptr<gko::factorization::elimination_forest<index_type>> forest;
    gko::factorization::compute_elim_forest(mtx.get(), forest);
    auto l_factor = matrix_type::create(this->ref, gko::dim<2>{10, 10}, 26);
    gko::kernels::reference::cholesky::cholesky_symbolic_count(
        this->ref, mtx.get(), *forest, l_factor->get_row_ptrs(), this->tmp);
    gko::kernels::reference::components::prefix_sum(
        this->ref, l_factor->get_row_ptrs(), 11);

    gko::kernels::reference::cholesky::cholesky_symbolic_factorize(
        this->ref, mtx.get(), *forest, l_factor.get(), this->tmp);

    GKO_ASSERT_MTX_EQ_SPARSITY(l_factor,
                               l({{1., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
                                  {0., 1., 0., 0., 0., 0., 0., 0., 0., 0.},
                                  {1., 0., 1., 0., 0., 0., 0., 0., 0., 0.},
                                  {0., 0., 0., 1., 0., 0., 0., 0., 0., 0.},
                                  {0., 1., 0., 0., 1., 0., 0., 0., 0., 0.},
                                  {0., 0., 0., 0., 0., 1., 0., 0., 0., 0.},
                                  {0., 0., 1., 0., 0., 1., 1., 0., 0., 0.},
                                  {1., 0., 1., 0., 0., 1., 1., 1., 0., 0.},
                                  {0., 0., 0., 1., 1., 0., 0., 1., 1., 0.},
                                  {0., 1., 0., 1., 1., 0., 0., 1., 1., 1.}}));
}


TYPED_TEST(Cholesky, KernelSymbolicCountSeparable)
{
    using matrix_type = typename TestFixture::matrix_type;
    using index_type = typename TestFixture::index_type;
    auto mtx = gko::initialize<typename TestFixture::matrix_type>(
        {{1, 0, 1, 0, 0, 0, 0, 0, 0, 0},
         {0, 1, 1, 0, 0, 0, 0, 0, 0, 0},
         {1, 1, 1, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 1, 1, 0, 0, 0, 0, 0},
         {0, 0, 0, 1, 1, 1, 0, 0, 0, 1},
         {0, 0, 0, 0, 1, 1, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 1, 1, 0, 1},
         {0, 0, 0, 0, 0, 0, 1, 1, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 1, 1},
         {0, 0, 0, 0, 1, 0, 1, 0, 1, 1}},
        this->ref);
    std::unique_ptr<gko::factorization::elimination_forest<index_type>> forest;
    gko::factorization::compute_elim_forest(mtx.get(), forest);
    gko::array<index_type> row_nnz{this->ref, 10};

    gko::kernels::reference::cholesky::cholesky_symbolic_count(
        this->ref, mtx.get(), *forest, row_nnz.get_data(), this->tmp);

    GKO_ASSERT_ARRAY_EQ(row_nnz, I<index_type>({1, 1, 3, 1, 2, 2, 1, 2, 1, 6}));
}


TYPED_TEST(Cholesky, KernelSymbolicFactorizeSeparable)
{
    using matrix_type = typename TestFixture::matrix_type;
    using index_type = typename TestFixture::index_type;
    auto mtx = gko::initialize<typename TestFixture::matrix_type>(
        {{1, 0, 1, 0, 0, 0, 0, 0, 0, 0},
         {0, 1, 1, 0, 0, 0, 0, 0, 0, 0},
         {1, 1, 1, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 1, 1, 0, 0, 0, 0, 0},
         {0, 0, 0, 1, 1, 1, 0, 0, 0, 1},
         {0, 0, 0, 0, 1, 1, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 1, 1, 0, 1},
         {0, 0, 0, 0, 0, 0, 1, 1, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 1, 1},
         {0, 0, 0, 0, 1, 0, 1, 0, 1, 1}},
        this->ref);
    std::unique_ptr<gko::factorization::elimination_forest<index_type>> forest;
    gko::factorization::compute_elim_forest(mtx.get(), forest);
    auto l_factor = matrix_type::create(this->ref, gko::dim<2>{10, 10}, 26);
    gko::kernels::reference::cholesky::cholesky_symbolic_count(
        this->ref, mtx.get(), *forest, l_factor->get_row_ptrs(), this->tmp);
    gko::kernels::reference::components::prefix_sum(
        this->ref, l_factor->get_row_ptrs(), 11);

    gko::kernels::reference::cholesky::cholesky_symbolic_factorize(
        this->ref, mtx.get(), *forest, l_factor.get(), this->tmp);

    GKO_ASSERT_MTX_EQ_SPARSITY(l_factor,
                               l({{1., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
                                  {0., 1., 0., 0., 0., 0., 0., 0., 0., 0.},
                                  {1., 1., 1., 0., 0., 0., 0., 0., 0., 0.},
                                  {0., 0., 0., 1., 0., 0., 0., 0., 0., 0.},
                                  {0., 0., 0., 1., 1., 0., 0., 0., 0., 0.},
                                  {0., 0., 0., 0., 1., 1., 0., 0., 0., 0.},
                                  {0., 0., 0., 0., 0., 0., 1., 0., 0., 0.},
                                  {0., 0., 0., 0., 0., 0., 1., 1., 0., 0.},
                                  {0., 0., 0., 0., 0., 0., 0., 0., 1., 0.},
                                  {0., 0., 0., 0., 1., 1., 1., 1., 1., 1.}}));
}


TYPED_TEST(Cholesky, KernelSymbolicCountMissingDiagonal)
{
    using matrix_type = typename TestFixture::matrix_type;
    using index_type = typename TestFixture::index_type;
    auto mtx = gko::initialize<typename TestFixture::matrix_type>(
        {{1, 0, 1, 0, 0, 0, 0, 0, 0, 0},
         {0, 1, 1, 0, 0, 0, 0, 0, 0, 0},
         {1, 1, 0, 1, 0, 0, 0, 0, 0, 0},
         {0, 0, 1, 1, 1, 0, 0, 0, 0, 0},
         {0, 0, 0, 1, 0, 1, 0, 0, 0, 0},
         {0, 0, 0, 0, 1, 1, 1, 0, 0, 0},
         {0, 0, 0, 0, 0, 1, 1, 1, 0, 1},
         {0, 0, 0, 0, 0, 0, 1, 1, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 1, 1},
         {0, 0, 0, 0, 0, 0, 1, 0, 1, 0}},
        this->ref);
    std::unique_ptr<gko::factorization::elimination_forest<index_type>> forest;
    gko::factorization::compute_elim_forest(mtx.get(), forest);
    gko::array<index_type> row_nnz{this->ref, 10};

    gko::kernels::reference::cholesky::cholesky_symbolic_count(
        this->ref, mtx.get(), *forest, row_nnz.get_data(), this->tmp);

    GKO_ASSERT_ARRAY_EQ(row_nnz, I<index_type>({1, 1, 3, 2, 2, 2, 2, 2, 1, 4}));
}


TYPED_TEST(Cholesky, KernelSymbolicFactorizeMissingDiagonal)
{
    using matrix_type = typename TestFixture::matrix_type;
    using index_type = typename TestFixture::index_type;
    auto mtx = gko::initialize<typename TestFixture::matrix_type>(
        {{1, 0, 1, 0, 0, 0, 0, 0, 0, 0},
         {0, 1, 1, 0, 0, 0, 0, 0, 0, 0},
         {1, 1, 0, 1, 0, 0, 0, 0, 0, 0},
         {0, 0, 1, 1, 1, 0, 0, 0, 0, 0},
         {0, 0, 0, 1, 0, 1, 0, 0, 0, 0},
         {0, 0, 0, 0, 1, 1, 1, 0, 0, 0},
         {0, 0, 0, 0, 0, 1, 1, 1, 0, 1},
         {0, 0, 0, 0, 0, 0, 1, 1, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 1, 1},
         {0, 0, 0, 0, 0, 0, 1, 0, 1, 0}},
        this->ref);
    std::unique_ptr<gko::factorization::elimination_forest<index_type>> forest;
    gko::factorization::compute_elim_forest(mtx.get(), forest);
    auto l_factor = matrix_type::create(this->ref, gko::dim<2>{10, 10}, 20);
    gko::kernels::reference::cholesky::cholesky_symbolic_count(
        this->ref, mtx.get(), *forest, l_factor->get_row_ptrs(), this->tmp);
    gko::kernels::reference::components::prefix_sum(
        this->ref, l_factor->get_row_ptrs(), 11);

    gko::kernels::reference::cholesky::cholesky_symbolic_factorize(
        this->ref, mtx.get(), *forest, l_factor.get(), this->tmp);

    GKO_ASSERT_MTX_EQ_SPARSITY(l_factor,
                               l({{1., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
                                  {0., 1., 0., 0., 0., 0., 0., 0., 0., 0.},
                                  {1., 1., 1., 0., 0., 0., 0., 0., 0., 0.},
                                  {0., 0., 1., 1., 0., 0., 0., 0., 0., 0.},
                                  {0., 0., 0., 1., 1., 0., 0., 0., 0., 0.},
                                  {0., 0., 0., 0., 1., 1., 0., 0., 0., 0.},
                                  {0., 0., 0., 0., 0., 1., 1., 0., 0., 0.},
                                  {0., 0., 0., 0., 0., 0., 1., 1., 0., 0.},
                                  {0., 0., 0., 0., 0., 0., 0., 0., 1., 0.},
                                  {0., 0., 0., 0., 0., 0., 1., 1., 1., 1.}}));
}


TYPED_TEST(Cholesky, KernelSymbolicCountAni1)
{
    using matrix_type = typename TestFixture::matrix_type;
    using index_type = typename TestFixture::index_type;
    std::ifstream stream{gko::matrices::location_ani1_mtx};
    auto mtx = gko::read<matrix_type>(stream, this->ref);
    std::unique_ptr<gko::factorization::elimination_forest<index_type>> forest;
    gko::factorization::compute_elim_forest(mtx.get(), forest);
    gko::array<index_type> row_nnz{this->ref, mtx->get_size()[0]};

    gko::kernels::reference::cholesky::cholesky_symbolic_count(
        this->ref, mtx.get(), *forest, row_nnz.get_data(), this->tmp);

    GKO_ASSERT_ARRAY_EQ(
        row_nnz, I<index_type>({1, 2, 3, 3, 2, 2,  7,  7,  7,  8, 8, 7,
                                8, 8, 8, 8, 2, 10, 10, 10, 10, 9, 8, 8,
                                8, 7, 8, 2, 8, 8,  7,  5,  8,  6, 4, 4}));
}


TYPED_TEST(Cholesky, KernelSymbolicFactorizeAni1)
{
    using matrix_type = typename TestFixture::matrix_type;
    using index_type = typename TestFixture::index_type;
    std::ifstream stream{gko::matrices::location_ani1_mtx};
    std::ifstream ref_stream{gko::matrices::location_ani1_chol_mtx};
    auto mtx = gko::read<matrix_type>(stream, this->ref);
    auto l_factor_ref = gko::read<matrix_type>(ref_stream, this->ref);
    std::unique_ptr<gko::factorization::elimination_forest<index_type>> forest;
    gko::factorization::compute_elim_forest(mtx.get(), forest);
    auto l_factor =
        matrix_type::create(this->ref, l_factor_ref->get_size(),
                            l_factor_ref->get_num_stored_elements());
    gko::kernels::reference::cholesky::cholesky_symbolic_count(
        this->ref, mtx.get(), *forest, l_factor->get_row_ptrs(), this->tmp);
    gko::kernels::reference::components::prefix_sum(
        this->ref, l_factor->get_row_ptrs(), mtx->get_size()[0] + 1);

    gko::kernels::reference::cholesky::cholesky_symbolic_factorize(
        this->ref, mtx.get(), *forest, l_factor.get(), this->tmp);

    GKO_ASSERT_MTX_EQ_SPARSITY(l_factor, l_factor_ref);
}


TYPED_TEST(Cholesky, SymbolicFactorizeAni1)
{
    using matrix_type = typename TestFixture::matrix_type;
    using index_type = typename TestFixture::index_type;
    std::ifstream stream{gko::matrices::location_ani1_mtx};
    std::ifstream ref_stream{gko::matrices::location_ani1_chol_mtx};
    auto mtx = gko::read<matrix_type>(stream, this->ref);
    auto l_factor_ref = gko::read<matrix_type>(ref_stream, this->ref);
    auto combined_factor_ref = this->combined_factor(l_factor_ref.get());

    std::unique_ptr<matrix_type> combined_factor;
    gko::factorization::symbolic_cholesky(mtx.get(), combined_factor);

    GKO_ASSERT_MTX_EQ_SPARSITY(combined_factor, combined_factor_ref);
}


TYPED_TEST(Cholesky, KernelSymbolicCountAni1Amd)
{
    using matrix_type = typename TestFixture::matrix_type;
    using index_type = typename TestFixture::index_type;
    std::ifstream stream{gko::matrices::location_ani1_amd_mtx};
    auto mtx = gko::read<matrix_type>(stream, this->ref);
    std::unique_ptr<gko::factorization::elimination_forest<index_type>> forest;
    gko::factorization::compute_elim_forest(mtx.get(), forest);
    gko::array<index_type> row_nnz{this->ref, mtx->get_size()[0]};

    gko::kernels::reference::cholesky::cholesky_symbolic_count(
        this->ref, mtx.get(), *forest, row_nnz.get_data(), this->tmp);

    GKO_ASSERT_ARRAY_EQ(
        row_nnz, I<index_type>({1, 1,  2, 3, 5,  4, 1, 2,  3,  4, 1,  2,
                                2, 2,  5, 1, 4,  4, 4, 1,  2,  3, 4,  3,
                                8, 10, 4, 8, 10, 7, 7, 13, 21, 6, 11, 14}));
}


TYPED_TEST(Cholesky, KernelSymbolicFactorizeAni1Amd)
{
    using matrix_type = typename TestFixture::matrix_type;
    using index_type = typename TestFixture::index_type;
    std::ifstream stream{gko::matrices::location_ani1_amd_mtx};
    std::ifstream ref_stream{gko::matrices::location_ani1_amd_chol_mtx};
    auto mtx = gko::read<matrix_type>(stream, this->ref);
    auto l_factor_ref = gko::read<matrix_type>(ref_stream, this->ref);
    std::unique_ptr<gko::factorization::elimination_forest<index_type>> forest;
    gko::factorization::compute_elim_forest(mtx.get(), forest);
    auto l_factor =
        matrix_type::create(this->ref, l_factor_ref->get_size(),
                            l_factor_ref->get_num_stored_elements());
    gko::kernels::reference::cholesky::cholesky_symbolic_count(
        this->ref, mtx.get(), *forest, l_factor->get_row_ptrs(), this->tmp);
    gko::kernels::reference::components::prefix_sum(
        this->ref, l_factor->get_row_ptrs(), mtx->get_size()[0] + 1);

    gko::kernels::reference::cholesky::cholesky_symbolic_factorize(
        this->ref, mtx.get(), *forest, l_factor.get(), this->tmp);

    GKO_ASSERT_MTX_EQ_SPARSITY(l_factor, l_factor_ref);
}


TYPED_TEST(Cholesky, SymbolicFactorizeAni1Amd)
{
    using matrix_type = typename TestFixture::matrix_type;
    using value_type = typename TestFixture::value_type;
    std::ifstream stream{gko::matrices::location_ani1_amd_mtx};
    std::ifstream ref_stream{gko::matrices::location_ani1_amd_chol_mtx};
    auto mtx = gko::read<matrix_type>(stream, this->ref);
    auto l_factor_ref = gko::read<matrix_type>(ref_stream, this->ref);
    auto combined_factor_ref = this->combined_factor(l_factor_ref.get());

    std::unique_ptr<matrix_type> combined_factor;
    gko::factorization::symbolic_cholesky(mtx.get(), combined_factor);

    GKO_ASSERT_MTX_EQ_SPARSITY(combined_factor, combined_factor_ref);
}


}  // namespace
