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
#include <ginkgo/core/matrix/permutation.hpp>
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
    using permutation_type = gko::matrix::Permutation<index_type>;
    static constexpr auto inf = std::numeric_limits<real_type>::infinity();
    static constexpr real_type tol = 1e-14;

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
          tolerance{10 * std::numeric_limits<real_type>::epsilon()}
    {}

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    gko::array<real_type> tmp;
    std::shared_ptr<matrix_type> mtx;
    const real_type tolerance;
};

TYPED_TEST_SUITE(Mc64, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(Mc64, InitializeWeightsSum)
{
    using matrix_type = typename TestFixture::matrix_type;
    using real_type = typename TestFixture::real_type;

    const auto num_rows = this->mtx->get_size()[0];
    const auto nnz = this->mtx->get_num_stored_elements();
    gko::array<real_type> value_workspace{this->ref, nnz + 3 * num_rows};
    gko::array<real_type> expected_value_workspace{
        this->ref,
        I<real_type>{2.,        1.,        0.,        0.,        4.,
                     0.,        2.,        0.,        1.,        0.,
                     2.,        3.,        0.,        0.,        1.,
                     0.,        0.,        0.,        1.,        this->inf,
                     this->inf, this->inf, this->inf, this->inf, this->inf,
                     3.,        5.,        6.,        4.,        4.,
                     8.}};

    gko::kernels::reference::mc64::initialize_weights(
        this->ref, this->mtx.get(), value_workspace,
        gko::reorder::reordering_strategy::max_diagonal_sum);

    GKO_ASSERT_ARRAY_EQ(value_workspace, expected_value_workspace);
}


TYPED_TEST(Mc64, InitializeWeightsProduct)
{
    using matrix_type = typename TestFixture::matrix_type;
    using real_type = typename TestFixture::real_type;

    const auto num_rows = this->mtx->get_size()[0];
    const auto nnz = this->mtx->get_num_stored_elements();
    gko::array<real_type> value_workspace{this->ref, nnz + 3 * num_rows};
    // This is really ugly, but the logarithms screw up the result if they are
    // merged
    gko::array<real_type> expected_value_workspace{
        this->ref,
        I<real_type>{real_type{std::log2(3.)},
                     real_type{std::log2(3.)} - real_type{std::log2(2.)},
                     0.,
                     0.,
                     real_type{std::log2(5.)},
                     0.,
                     real_type{std::log2(6.)} - real_type{std::log2(4.)},
                     0.,
                     real_type{std::log2(4.)} - real_type{std::log2(3.)},
                     0.,
                     real_type{std::log2(4.)} - real_type{std::log2(2.)},
                     real_type{std::log2(8.)} - real_type{std::log2(5.)},
                     0.,
                     0.,
                     real_type{std::log2(3.)} - real_type{std::log2(2.)},
                     0.,
                     0.,
                     0.,
                     real_type{std::log2(4.)} - real_type{std::log2(3.)},
                     this->inf,
                     this->inf,
                     this->inf,
                     this->inf,
                     this->inf,
                     this->inf,
                     real_type{std::log2(3.)},
                     real_type{std::log2(5.)},
                     real_type{std::log2(6.)},
                     real_type{std::log2(4.)},
                     real_type{std::log2(4.)},
                     real_type{std::log2(8.)}}};

    gko::kernels::reference::mc64::initialize_weights(
        this->ref, this->mtx.get(), value_workspace,
        gko::reorder::reordering_strategy::max_diagonal_product);

    GKO_ASSERT_ARRAY_EQ(value_workspace, expected_value_workspace);
}


TYPED_TEST(Mc64, InitialMatching)
{
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    gko::array<index_type> p{this->ref,
                             I<index_type>({-1, -1, -1, -1, -1, -1})};
    gko::array<index_type> ip{this->ref,
                              I<index_type>({-1, -1, -1, -1, -1, -1})};
    const auto num_rows = this->mtx->get_size()[0];
    gko::array<real_type> value_workspace{
        this->ref,
        I<real_type>{2.,        1.,        0.,        0.,        4.,
                     0.,        2.,        0.,        1.,        0.,
                     2.,        3.,        0.,        0.,        1.,
                     0.,        0.,        0.,        1.,        this->inf,
                     this->inf, this->inf, this->inf, this->inf, this->inf,
                     3.,        5.,        6.,        4.,        4.,
                     8.}};
    gko::array<index_type> index_workspace{this->ref, 6 * num_rows};
    index_workspace.fill(gko::zero<index_type>());
    gko::array<index_type> expected_index_workspace{
        this->ref,
        I<index_type>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 1, 3, 5, 8, 0, 12, 4, -1, 0, 0, 0, 0}};
    gko::array<index_type> expected_perm{this->ref,
                                         I<index_type>({1, 0, 3, 5, -1, 2})};
    gko::array<index_type> expected_inv_perm{
        this->ref, I<index_type>({1, 0, 5, 2, -1, 3})};

    gko::kernels::reference::mc64::initial_matching(
        this->ref, num_rows, this->mtx->get_const_row_ptrs(),
        this->mtx->get_const_col_idxs(), value_workspace, p, ip,
        index_workspace, this->tol);

    GKO_ASSERT_ARRAY_EQ(p, expected_perm);
    GKO_ASSERT_ARRAY_EQ(ip, expected_inv_perm);
    GKO_ASSERT_ARRAY_EQ(index_workspace, expected_index_workspace);
}


TYPED_TEST(Mc64, ShortestAugmentingPathExample)
{
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    gko::array<index_type> perm{this->ref, I<index_type>({1, 0, 3, 5, -1, 2})};
    gko::array<index_type> inv_perm{this->ref,
                                    I<index_type>({1, 0, 5, 2, -1, 3})};
    gko::array<index_type> expected_perm{this->ref,
                                         I<index_type>{1, 0, 3, 5, 4, 2}};
    gko::array<index_type> expected_inv_perm{this->ref,
                                             I<index_type>{1, 0, 5, 2, 4, 3}};
    gko::array<index_type> index_workspace{
        this->ref,
        I<index_type>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 1, 3, 5, 8, 0, 12, 4, -1, 0, 0, 0, 0}};
    gko::array<index_type> expected_index_workspace{
        this->ref, I<index_type>{0, 0, 3,  4,  4,  2,  0, 0,  0, 0, 0, 0,
                                 0, 0, -4, -4, 0,  -4, 3, 5,  2, 0, 0, 0,
                                 1, 3, 5,  8,  10, 12, 4, -1, 0, 0, 0, 0}};
    gko::array<real_type> value_workspace{
        this->ref,
        I<real_type>{2.,        1.,        0.,        0.,        4.,
                     0.,        2.,        0.,        1.,        0.,
                     2.,        3.,        0.,        0.,        1.,
                     0.,        0.,        0.,        1.,        this->inf,
                     this->inf, this->inf, this->inf, this->inf, this->inf,
                     3.,        5.,        6.,        4.,        4.,
                     8.}};
    gko::array<real_type> expected_value_workspace{
        this->ref,
        I<real_type>{2.,  1., 0., 0.,        4.,        0., 2., 0.,
                     1.,  0., 2., 3.,        0.,        0., 1., -1.,
                     -2., 0., 0., this->inf, this->inf, 1., 0., this->inf,
                     1.,  3., 5., 6.,        4.,        4., 8.}};
    gko::addressable_priority_queue<real_type, index_type> Q{4};
    std::vector<index_type> q_j{};

    gko::kernels::reference::mc64::shortest_augmenting_path(
        this->ref, this->mtx->get_size()[0], this->mtx->get_const_row_ptrs(),
        this->mtx->get_const_col_idxs(), value_workspace, perm, inv_perm,
        4 * gko::one<index_type>(), index_workspace, Q, q_j, this->tol);

    GKO_ASSERT_ARRAY_EQ(perm, expected_perm);
    GKO_ASSERT_ARRAY_EQ(inv_perm, expected_inv_perm);
    GKO_ASSERT_ARRAY_EQ(index_workspace, expected_index_workspace);
    GKO_ASSERT_ARRAY_EQ(value_workspace, expected_value_workspace);
}


TYPED_TEST(Mc64, CreatesCorrectPermutationAndScalingExampleSum)
{
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    using value_type = typename TestFixture::value_type;
    using permutation_type = typename TestFixture::permutation_type;

    auto mc64_factory =
        gko::reorder::Mc64<value_type, index_type>::build()
            .with_strategy(gko::reorder::reordering_strategy::max_diagonal_sum)
            .on(this->ref);
    auto mc64 = mc64_factory->generate(this->mtx);

    auto perm = gko::as<permutation_type>(mc64->get_permutation())
                    ->get_const_permutation();
    auto inv_perm = gko::as<permutation_type>(mc64->get_inverse_permutation())
                        ->get_const_permutation();
    GKO_ASSERT_EQ(perm[0], 1);
    GKO_ASSERT_EQ(perm[1], 0);
    GKO_ASSERT_EQ(perm[2], 5);
    GKO_ASSERT_EQ(perm[3], 2);
    GKO_ASSERT_EQ(perm[4], 4);
    GKO_ASSERT_EQ(perm[5], 3);
    GKO_ASSERT_EQ(inv_perm[0], 1);
    GKO_ASSERT_EQ(inv_perm[1], 0);
    GKO_ASSERT_EQ(inv_perm[2], 3);
    GKO_ASSERT_EQ(inv_perm[3], 5);
    GKO_ASSERT_EQ(inv_perm[4], 4);
    GKO_ASSERT_EQ(inv_perm[5], 2);
}


TYPED_TEST(Mc64, CreatesCorrectPermutationAndScalingExampleProduct)
{
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    using value_type = typename TestFixture::value_type;
    using matrix_type = typename TestFixture::matrix_type;
    using permutation_type = typename TestFixture::permutation_type;

    auto mc64_factory =
        gko::reorder::Mc64<value_type, index_type>::build()
            .with_strategy(
                gko::reorder::reordering_strategy::max_diagonal_product)
            .on(this->ref);
    auto mc64 = mc64_factory->generate(this->mtx);

    auto perm = gko::as<permutation_type>(mc64->get_permutation())
                    ->get_const_permutation();
    auto inv_perm = gko::as<permutation_type>(mc64->get_inverse_permutation())
                        ->get_const_permutation();
    auto row_scaling = mc64->get_row_scaling()->get_const_values();
    auto col_scaling = mc64->get_col_scaling()->get_const_values();

    GKO_ASSERT_EQ(perm[0], 1);
    GKO_ASSERT_EQ(perm[1], 5);
    GKO_ASSERT_EQ(perm[2], 3);
    GKO_ASSERT_EQ(perm[3], 4);
    GKO_ASSERT_EQ(perm[4], 0);
    GKO_ASSERT_EQ(perm[5], 2);
    GKO_ASSERT_EQ(inv_perm[0], 4);
    GKO_ASSERT_EQ(inv_perm[1], 0);
    GKO_ASSERT_EQ(inv_perm[2], 5);
    GKO_ASSERT_EQ(inv_perm[3], 2);
    GKO_ASSERT_EQ(inv_perm[4], 3);
    GKO_ASSERT_EQ(inv_perm[5], 1);
    GKO_ASSERT_NEAR(row_scaling[0], value_type{1. / 3.}, this->tolerance);
    GKO_ASSERT_NEAR(row_scaling[1], value_type{0.2}, this->tolerance);
    GKO_ASSERT_NEAR(row_scaling[2], value_type{0.2}, this->tolerance);
    GKO_ASSERT_NEAR(row_scaling[3], value_type{4. / 15.}, this->tolerance);
    GKO_ASSERT_NEAR(row_scaling[4], value_type{0.3}, this->tolerance);
    GKO_ASSERT_NEAR(row_scaling[5], value_type{2. / 15.}, this->tolerance);
    GKO_ASSERT_NEAR(col_scaling[0], value_type{1.}, this->tolerance);
    GKO_ASSERT_NEAR(col_scaling[1], value_type{1.5}, this->tolerance);
    GKO_ASSERT_NEAR(col_scaling[2], value_type{0.9375}, this->tolerance);
    GKO_ASSERT_NEAR(col_scaling[3], value_type{5. / 6.}, this->tolerance);
    GKO_ASSERT_NEAR(col_scaling[4], value_type{1.}, this->tolerance);
    GKO_ASSERT_NEAR(col_scaling[5], value_type{1.25}, this->tolerance);
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

    GKO_ASSERT_MTX_NEAR(mtx, expected_result, 1e-6);
    GKO_ASSERT_MTX_EQ_SPARSITY(mtx, expected_result);
}


}  // namespace
