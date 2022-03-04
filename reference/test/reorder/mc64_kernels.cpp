/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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
#include <cmath>
#include <fstream>
#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/reorder/mc64.hpp>


#include "core/reorder/mc64_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"
#include "matrices/config.hpp"


namespace {


template <typename ValueIndexType>
class Mc64 : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using real_type = gko::remove_complex<value_type>;
    using matrix_type = gko::matrix::Csr<value_type, index_type>;

    Mc64()
        : ref(gko::ReferenceExecutor::create()),
          tmp{ref},
          mtx(gko::initialize<matrix_type>({{1., 2., 0., 0., 3., 0.},
                                            {5., 1., 0., 0., 0., 0.},
                                            {0., 0., 0., 6., 0., 4.},
                                            {0., 0., 4., 0., 0., 3.},
                                            {0., 0., 0., 4., 2., 0.},
                                            {0., 5., 8., 0., 0., 0.}},
                                           ref)),
          expected_workspace_sum{
              ref, I<real_type>({2., 1., 0., 0., 4., 0., 2., 0., 1., 0., 2.,
                                 3., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.,
                                 0., 0., 0., 0., 0., 0., 0., 0., 0.})},
          expected_workspace_product{ref, I<real_type>({std::log2(3.),
                                                        std::log2(1.5),
                                                        0.,
                                                        0.,
                                                        std::log2(5.),
                                                        0.,
                                                        std::log2(1.5),
                                                        0.,
                                                        std::log2(4. / 3.),
                                                        0.,
                                                        std::log2(2.),
                                                        std::log2(1.6),
                                                        0.,
                                                        0.,
                                                        std::log2(1.5),
                                                        0.,
                                                        0.,
                                                        0.,
                                                        std::log2(4. / 3.),
                                                        0.,
                                                        0.,
                                                        0.,
                                                        0.,
                                                        0.,
                                                        0.,
                                                        0.,
                                                        0.,
                                                        0.,
                                                        0.,
                                                        0.,
                                                        0.})},
          expected_perm{ref, I<index_type>({1, 0, 3, 5, -1, 2})},
          expected_inv_perm{ref, I<index_type>({1, 0, 5, 2, -1, 3})},
          tolerance{std::numeric_limits<real_type>::epsilon()}
    {}

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    gko::Array<real_type> tmp;
    std::shared_ptr<matrix_type> mtx;
    gko::Array<real_type> expected_workspace_sum;
    gko::Array<real_type> expected_workspace_product;
    gko::Array<index_type> expected_perm;
    gko::Array<index_type> expected_inv_perm;
    const real_type tolerance;
};

TYPED_TEST_SUITE(Mc64, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(Mc64, InitializeWeightsExampleSum)
{
    using matrix_type = typename TestFixture::matrix_type;
    using real_type = typename TestFixture::real_type;

    gko::kernels::reference::mc64::initialize_weights(
        this->ref, this->mtx.get(), this->tmp,
        gko::reorder::reordering_strategy::max_diagonal_sum);

    GKO_ASSERT_ARRAY_EQ(this->tmp, this->expected_workspace_sum);
}


TYPED_TEST(Mc64, InitializeWeightsExampleProduct)
{
    using matrix_type = typename TestFixture::matrix_type;
    using real_type = typename TestFixture::real_type;

    gko::kernels::reference::mc64::initialize_weights(
        this->ref, this->mtx.get(), this->tmp,
        gko::reorder::reordering_strategy::max_diagonal_product);

    GKO_ASSERT_EQ(this->tmp.get_num_elems(),
                  this->expected_workspace_product.get_num_elems());
    for (gko::size_type i = 0; i < this->tmp.get_num_elems(); i++) {
        GKO_ASSERT_NEAR(this->tmp.get_data()[i],
                        this->expected_workspace_product.get_data()[i],
                        this->tolerance);
    }
}


TYPED_TEST(Mc64, InitialMatchingExampleSum)
{
    using index_type = typename TestFixture::index_type;
    gko::Array<index_type> p{this->ref,
                             I<index_type>({-1, -1, -1, -1, -1, -1})};
    gko::Array<index_type> ip{this->ref,
                              I<index_type>({-1, -1, -1, -1, -1, -1})};
    std::list<index_type> unmatched_rows{};

    gko::kernels::reference::mc64::initial_matching(
        this->ref, this->mtx->get_size()[0], this->mtx->get_const_row_ptrs(),
        this->mtx->get_const_col_idxs(), this->expected_workspace_sum, p, ip,
        unmatched_rows);

    GKO_ASSERT_ARRAY_EQ(p, this->expected_perm);
    GKO_ASSERT_ARRAY_EQ(ip, this->expected_inv_perm);
    GKO_ASSERT_EQ(unmatched_rows.size(), 1u);
    GKO_ASSERT_EQ(unmatched_rows.front(), 4 * gko::one<index_type>());
}


TYPED_TEST(Mc64, InitialMatchingExampleProduct)
{
    using index_type = typename TestFixture::index_type;
    gko::Array<index_type> p{this->ref,
                             I<index_type>({-1, -1, -1, -1, -1, -1})};
    gko::Array<index_type> ip{this->ref,
                              I<index_type>({-1, -1, -1, -1, -1, -1})};
    std::list<index_type> unmatched_rows{};

    gko::kernels::reference::mc64::initial_matching(
        this->ref, this->mtx->get_size()[0], this->mtx->get_const_row_ptrs(),
        this->mtx->get_const_col_idxs(), this->expected_workspace_product, p,
        ip, unmatched_rows);

    GKO_ASSERT_ARRAY_EQ(p, this->expected_perm);
    GKO_ASSERT_ARRAY_EQ(ip, this->expected_inv_perm);
    GKO_ASSERT_EQ(unmatched_rows.size(), 1u);
    GKO_ASSERT_EQ(unmatched_rows.front(), 4 * gko::one<index_type>());
}


TYPED_TEST(Mc64, ShortestAugmentingPathExample)
{
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    gko::Array<index_type> expected_perm{this->ref,
                                         I<index_type>{1, 0, 3, 5, 4, 2}};
    gko::Array<index_type> expected_inv_perm{this->ref,
                                             I<index_type>{1, 0, 5, 2, 4, 3}};
    gko::Array<index_type> parents{
        this->ref, I<index_type>{-1, -1, -1, -1, -1, -1, -2, -2, -2, -2, -2, -2,
                                 -2, -2, -2, -2, -2, -2}};
    gko::Array<index_type> expected_parents{
        this->ref, I<index_type>{-1, -1, 3, 4, 4, 2, -1, -1, -1, -1, -1, -1, -2,
                                 -2, -2, -2, -2, -2}};

    gko::kernels::reference::mc64::shortest_augmenting_path(
        this->ref, this->mtx->get_size()[0], this->mtx->get_const_row_ptrs(),
        this->mtx->get_const_col_idxs(), this->expected_workspace_sum,
        this->expected_perm, this->expected_inv_perm,
        4 * gko::one<index_type>(), parents);

    GKO_ASSERT_ARRAY_EQ(expected_perm, this->expected_perm);
    GKO_ASSERT_ARRAY_EQ(expected_inv_perm, this->expected_inv_perm);
    GKO_ASSERT_ARRAY_EQ(parents, expected_parents);
}


/*TYPED_TEST(Mc64, ShortestAugmentingPathExample2)
{
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    gko::Array<index_type> row_ptrs{
        this->ref, I<index_type>{0, 2, 6, 7, 10, 12, 15, 19, 21}};
    gko::Array<index_type> col_idxs{
        this->ref, I<index_type>{0, 1, 0, 1, 4, 6, 2, 3, 4, 5, 4,
                                 7, 4, 5, 6, 1, 3, 5, 7, 0, 2}};
    gko::Array<real_type> workspace{
        this->ref,
        I<real_type>{1., 0., 0., 0., 2., 4., 0., 0., 4., 2., 0., 1., 8.,
                     0., 6., 2., 4., 1., 8., 6., 4., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.}};
    gko::Array<real_type> expected_workspace{
        this->ref,
        I<real_type>{1.,  0., 0., 0., 2., 4., 0., 0., 4.,  2.,  0., 1.,  8.,
                     0.,  6., 2., 4., 1., 8., 6., 4., -3., -4., 0., -2., -1.,
                     -5., 0., 0., 4., 3., 0., 2., 1., 5.,  6.,  0.}};
    gko::Array<index_type> perm{this->ref,
                                I<index_type>{1, 0, 2, 3, 4, 5, -1, -1}};
    gko::Array<index_type> inv_perm{this->ref,
                                    I<index_type>{1, 0, 2, 3, 4, 5, -1, -1}};
    gko::Array<index_type> expected_perm{
        this->ref, I<index_type>{0, 4, 2, 3, 7, 5, 1, -1}};
    gko::Array<index_type> expected_inv_perm{
        this->ref, I<index_type>{0, 6, 2, 3, 1, 5, -1, 4}};
    gko::Array<index_type> parents{
        this->ref, I<index_type>{-1, -1, -1, -1, -1, -1, -1, -1}};
    gko::Array<index_type> expected_parents{
        this->ref, I<index_type>{0, 6, -1, 6, 1, 6, 5, 4}};

    gko::kernels::reference::mc64::shortest_augmenting_path(
        this->ref, 8u, row_ptrs.get_data(), col_idxs.get_data(), workspace,
        perm, inv_perm, 6 * gko::one<index_type>(), parents);

    GKO_ASSERT_ARRAY_EQ(perm, expected_perm);
    GKO_ASSERT_ARRAY_EQ(inv_perm, expected_inv_perm);
    GKO_ASSERT_ARRAY_EQ(parents, expected_parents);
    GKO_ASSERT_ARRAY_EQ(workspace, expected_workspace);
}*/


TYPED_TEST(Mc64, CreatesCorrectPermutationAndScalingExampleSum)
{
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    using value_type = typename TestFixture::value_type;

    auto mc64_factory =
        gko::reorder::Mc64<value_type, index_type>::build()
            .with_strategy(gko::reorder::reordering_strategy::max_diagonal_sum)
            .on(this->ref);
    auto mc64 = mc64_factory->generate(this->mtx);

    auto perm = mc64->get_permutation()->get_const_permutation();
    auto inv_perm = mc64->get_inverse_permutation()->get_const_permutation();
    GKO_ASSERT_EQ(perm[0], 1);
    GKO_ASSERT_EQ(perm[1], 0);
    GKO_ASSERT_EQ(perm[2], 3);
    GKO_ASSERT_EQ(perm[3], 5);
    GKO_ASSERT_EQ(perm[4], 4);
    GKO_ASSERT_EQ(perm[5], 2);
    GKO_ASSERT_EQ(inv_perm[0], 1);
    GKO_ASSERT_EQ(inv_perm[1], 0);
    GKO_ASSERT_EQ(inv_perm[2], 5);
    GKO_ASSERT_EQ(inv_perm[3], 2);
    GKO_ASSERT_EQ(inv_perm[4], 4);
    GKO_ASSERT_EQ(inv_perm[5], 3);
}


TYPED_TEST(Mc64, CreatesCorrectPermutationAndScalingExampleProduct)
{
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    using value_type = typename TestFixture::value_type;
    using matrix_type = typename TestFixture::matrix_type;

    auto expected_result =
        gko::initialize<matrix_type>({{1., 0.3, 0., 0., 0., 0.},
                                      {0., 1., 1., 0., 0., 0.},
                                      {0., 0., 1., 0., 0., 1.},
                                      {0., 0., 0., 1., 0.6, 0.},
                                      {1. / 3., 1., 0., 0., 1., 0.},
                                      {0., 0., 0., 1., 0., 1.}},
                                     this->ref);

    auto mc64_factory =
        gko::reorder::Mc64<value_type, index_type>::build()
            .with_strategy(
                gko::reorder::reordering_strategy::max_diagonal_product)
            .on(this->ref);
    auto mc64 = mc64_factory->generate(this->mtx);

    auto perm = mc64->get_permutation();  //->get_permutation();
    auto inv_perm =
        mc64->get_inverse_permutation();         //->get_const_permutation();
    auto row_scaling = mc64->get_row_scaling();  //->get_const_values();
    auto col_scaling = mc64->get_col_scaling();  //->get_const_values();

    auto result = gko::clone(this->ref, this->mtx);
    col_scaling->rapply(result.get(), result.get());
    row_scaling->apply(result.get(), result.get());
    perm->apply(result.get(), result.get());

    auto rp = result->get_row_ptrs();
    auto ci = result->get_col_idxs();
    auto v = result->get_values();
    for (auto i = 0; i < result->get_size()[0]; i++) {
        for (auto idx = rp[i]; idx < rp[i + 1]; idx++)
            std::cout << "(" << i << "," << ci[idx] << "," << v[idx] << ")";
    }
    std::cout << "CHECKING" << std::endl;
    GKO_ASSERT_MTX_NEAR(result, expected_result, this->tolerance);
    std::cout << "DONE" << std::endl;
    /*GKO_ASSERT_EQ(perm[0], 4);
    GKO_ASSERT_EQ(perm[1], 0);
    GKO_ASSERT_EQ(perm[2], 5);
    GKO_ASSERT_EQ(perm[3], 2);
    GKO_ASSERT_EQ(perm[4], 3);
    GKO_ASSERT_EQ(perm[5], 1);
    GKO_ASSERT_EQ(inv_perm[0], 1);
    GKO_ASSERT_EQ(inv_perm[1], 5);
    GKO_ASSERT_EQ(inv_perm[2], 3);
    GKO_ASSERT_EQ(inv_perm[3], 4);
    GKO_ASSERT_EQ(inv_perm[4], 0);
    GKO_ASSERT_EQ(inv_perm[5], 2);
    GKO_ASSERT_NEAR(row_scaling[0], real_type{0.6}, this->tolerance);
    GKO_ASSERT_NEAR(row_scaling[1], real_type{1./3.}, this->tolerance);
    GKO_ASSERT_NEAR(row_scaling[2], real_type{2./3.}, this->tolerance);
    GKO_ASSERT_NEAR(row_scaling[3], real_type{0.75}, this->tolerance);
    GKO_ASSERT_NEAR(row_scaling[4], real_type{5./6.}, this->tolerance);
    GKO_ASSERT_NEAR(row_scaling[5], real_type{0.5}, this->tolerance);
    GKO_ASSERT_NEAR(col_scaling[0], real_type{1./3.}, this->tolerance);
    GKO_ASSERT_NEAR(col_scaling[1], real_type{0.6}, this->tolerance);
    GKO_ASSERT_NEAR(col_scaling[2], real_type{0.375}, this->tolerance);
    GKO_ASSERT_NEAR(col_scaling[3], real_type{1./3.}, this->tolerance);
    GKO_ASSERT_NEAR(col_scaling[4], real_type{0.4}, this->tolerance);
    GKO_ASSERT_NEAR(col_scaling[5], real_type{0.5}, this->tolerance);*/
}


TYPED_TEST(Mc64, CreatesCorrectPermutationAndScalingLargeTrivialExampleProduct)
{
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    using value_type = typename TestFixture::value_type;
    using matrix_type = typename TestFixture::matrix_type;

    std::ifstream mtx_stream{gko::matrices::location_1138_bus_mtx};
    auto mtx = gko::share(gko::read<matrix_type>(mtx_stream, this->ref));
    std::ifstream result_stream{gko::matrices::location_1138_bus_mc64_result};
    auto expected_result = gko::read<matrix_type>(result_stream, this->ref);

    auto mc64_factory =
        gko::reorder::Mc64<value_type, index_type>::build()
            .with_strategy(
                gko::reorder::reordering_strategy::max_diagonal_product)
            .on(this->ref);
    auto mc64 = mc64_factory->generate(mtx);

    auto perm = mc64->get_permutation();
    auto row_scaling = mc64->get_row_scaling();
    auto col_scaling = mc64->get_col_scaling();

    col_scaling->rapply(mtx.get(), mtx.get());
    row_scaling->apply(mtx.get(), mtx.get());
    perm->apply(mtx.get(), mtx.get());

    GKO_ASSERT_MTX_NEAR(mtx, expected_result, this->tolerance);
}


TYPED_TEST(Mc64, CreatesCorrectPermutationAndScalingLargeExampleProduct)
{
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    using value_type = typename TestFixture::value_type;
    using matrix_type = typename TestFixture::matrix_type;

    std::ifstream mtx_stream{gko::matrices::location_nontrivial_mc64_example};
    auto mtx = gko::share(gko::read<matrix_type>(mtx_stream, this->ref));
    mtx->sort_by_column_index();
    std::ifstream result_stream{gko::matrices::location_nontrivial_mc64_result};
    auto expected_result = gko::read<matrix_type>(result_stream, this->ref);

    auto mc64_factory =
        gko::reorder::Mc64<value_type, index_type>::build()
            .with_strategy(gko::reorder::reordering_strategy::max_diagonal_sum)
            .on(this->ref);
    auto mc64 = mc64_factory->generate(mtx);

    auto perm = mc64->get_permutation();
    auto row_scaling = mc64->get_row_scaling();
    auto col_scaling = mc64->get_col_scaling();

    col_scaling->rapply(mtx.get(), mtx.get());
    row_scaling->apply(mtx.get(), mtx.get());
    perm->apply(mtx.get(), mtx.get());

    // GKO_ASSERT_MTX_NEAR(mtx, expected_result, this->tolerance);
    GKO_ASSERT_MTX_EQ_SPARSITY(mtx, expected_result);
}


}  // namespace
