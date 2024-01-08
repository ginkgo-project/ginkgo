// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/reorder/mc64.hpp>


#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/permutation.hpp>


#include "core/components/addressable_pq.hpp"
#include "core/reorder/mc64.hpp"
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
    using perm_type = gko::matrix::ScaledPermutation<value_type, index_type>;
    // this is a constexpr member functions to avoid having to export the symbol
    // for the constant variable
    static constexpr real_type inf()
    {
        return std::numeric_limits<real_type>::infinity();
    }

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
          weights{ref, 13},
          dual_u{ref, 6},
          distance{ref, 6},
          row_maxima{ref, 6},
          initialized_weights_sum{ref, I<real_type>{2., 1., 0., 0., 4., 0., 2.,
                                                    0., 1., 0., 2., 3., 0.}},
          initialized_dual_u_sum{ref, I<real_type>{0., 1., 0., 0., 0., 1.}},
          initialized_row_maxima_sum{ref, I<real_type>{3., 5., 6., 4., 4., 8.}},
          // if the logarithms are merged together, the rounding messes up the
          // accuracy for GKO_ASSRT_ARRAY_EQ
          initialized_weights_product{
              ref,
              I<real_type>{static_cast<real_type>(std::log2(3.)),
                           static_cast<real_type>(std::log2(3.)) -
                               static_cast<real_type>(std::log2(2.)),
                           0., 0., static_cast<real_type>(std::log2(5.)), 0.,
                           static_cast<real_type>(std::log2(6.)) -
                               static_cast<real_type>(std::log2(4.)),
                           0.,
                           static_cast<real_type>(std::log2(4.)) -
                               static_cast<real_type>(std::log2(3.)),
                           0.,
                           static_cast<real_type>(std::log2(4.)) -
                               static_cast<real_type>(std::log2(2.)),
                           static_cast<real_type>(std::log2(8.)) -
                               static_cast<real_type>(std::log2(5.)),
                           0.}},
          initialized_dual_u_product{
              ref, I<real_type>{0.,
                                static_cast<real_type>(std::log2(3.)) -
                                    static_cast<real_type>(std::log2(2.)),
                                0., 0., 0.,
                                static_cast<real_type>(std::log2(4.)) -
                                    static_cast<real_type>(std::log2(3.))}},
          initialized_row_maxima_product{
              ref, I<real_type>{static_cast<real_type>(std::log2(3.)),
                                static_cast<real_type>(std::log2(5.)),
                                static_cast<real_type>(std::log2(6.)),
                                static_cast<real_type>(std::log2(4.)),
                                static_cast<real_type>(std::log2(4.)),
                                static_cast<real_type>(std::log2(8.))}},
          initialized_distance{
              ref, I<real_type>{inf(), inf(), inf(), inf(), inf(), inf()}},
          empty_permutation{ref, I<index_type>{-1, -1, -1, -1, -1, -1}},
          empty_inverse_permutation{ref, I<index_type>{-1, -1, -1, -1, -1, -1}},
          empty_matched_idxs{ref, I<index_type>{0, 0, 0, 0, 0, 0}},
          empty_unmatched_rows{ref, I<index_type>{0, 0, 0, 0, 0, 0}},
          initial_parents{ref, I<index_type>{0, 0, 0, 0, 0, 0}},
          initial_generation{ref, I<index_type>{0, 0, 0, 0, 0, 0}},
          initial_marked_cols{ref, I<index_type>{0, 0, 0, 0, 0, 0}},
          initial_matched_idxs{ref, I<index_type>{1, 3, 5, 8, 0, 12}},
          initial_unmatched_rows{ref, I<index_type>{4, -1, 0, 0, 0, 0}},
          initial_matching_permutation{ref, I<index_type>{1, 0, 3, 5, -1, 2}},
          initial_matching_inverse_permutation{
              ref, I<index_type>{1, 0, 5, 2, -1, 3}},
          final_permutation{ref, I<index_type>{1, 0, 3, 5, 4, 2}},
          final_inverse_permutation{ref, I<index_type>{1, 0, 5, 2, 4, 3}},
          final_parents{ref, I<index_type>{0, 0, 3, 4, 4, 2}},
          final_generation{ref, I<index_type>{0, 0, -4, -4, 0, -4}},
          final_marked_cols{ref, I<index_type>{3, 5, 2, 0, 0, 0}},
          final_matched_idxs{ref, I<index_type>{1, 3, 5, 8, 10, 12}},
          final_weights{ref, I<real_type>{2., 1., 0., 0., 4., 0., 2., 0., 1.,
                                          0., 2., 3., 0.}},
          final_dual_u{ref, I<real_type>{0., 1., -1., -2., 0., 0.}},
          final_distance{ref, I<real_type>{inf(), inf(), 1., 0., inf(), 1.}},
          zero_tol{1e-14}
    {}

    std::pair<std::shared_ptr<const perm_type>,
              std::shared_ptr<const perm_type>>
    unpack(const gko::Composition<value_type>* result)
    {
        return std::make_pair(gko::as<perm_type>(result->get_operators()[0]),
                              gko::as<perm_type>(result->get_operators()[1]));
    }

    void assert_array_near(const gko::array<real_type>& a,
                           const gko::array<real_type>& b, std::string name)
    {
        ASSERT_EQ(a.get_size(), b.get_size());
        for (gko::size_type i = 0; i < a.get_size(); i++) {
            if (std::isfinite(a.get_const_data()[i]) ||
                std::isfinite(b.get_const_data()[i])) {
                ASSERT_NEAR(a.get_const_data()[i], b.get_const_data()[i],
                            r<value_type>::value)
                    << name << '[' << i << ']';
            }
        }
    }

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    gko::array<real_type> tmp;
    gko::array<real_type> weights;
    gko::array<real_type> dual_u;
    gko::array<real_type> distance;
    gko::array<real_type> row_maxima;
    gko::array<real_type> initialized_weights_sum;
    gko::array<real_type> initialized_dual_u_sum;
    gko::array<real_type> initialized_row_maxima_sum;
    gko::array<real_type> initialized_weights_product;
    gko::array<real_type> initialized_dual_u_product;
    gko::array<real_type> initialized_row_maxima_product;
    gko::array<real_type> initialized_distance;
    gko::array<real_type> final_weights;
    gko::array<real_type> final_dual_u;
    gko::array<real_type> final_distance;
    gko::array<index_type> empty_permutation;
    gko::array<index_type> empty_inverse_permutation;
    gko::array<index_type> empty_matched_idxs;
    gko::array<index_type> empty_unmatched_rows;
    gko::array<index_type> initial_matching_permutation;
    gko::array<index_type> initial_matching_inverse_permutation;
    gko::array<index_type> initial_parents;
    gko::array<index_type> initial_generation;
    gko::array<index_type> initial_marked_cols;
    gko::array<index_type> initial_matched_idxs;
    gko::array<index_type> initial_unmatched_rows;
    gko::array<index_type> final_permutation;
    gko::array<index_type> final_inverse_permutation;
    gko::array<index_type> final_parents;
    gko::array<index_type> final_generation;
    gko::array<index_type> final_marked_cols;
    gko::array<index_type> final_matched_idxs;
    std::shared_ptr<matrix_type> mtx;
    const real_type zero_tol;
};

TYPED_TEST_SUITE(Mc64, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(Mc64, InitializeWeightsSum)
{
    this->dual_u.fill(
        std::numeric_limits<typename TestFixture::real_type>::infinity());

    gko::experimental::reorder::mc64::initialize_weights(
        this->mtx.get(), this->weights, this->dual_u, this->row_maxima,
        gko::experimental::reorder::mc64_strategy::max_diagonal_sum);

    this->assert_array_near(this->weights, this->initialized_weights_sum,
                            "weights");
    this->assert_array_near(this->dual_u, this->initialized_dual_u_sum,
                            "dual_u");
    this->assert_array_near(this->row_maxima, this->initialized_row_maxima_sum,
                            "row_maxima");
}


TYPED_TEST(Mc64, InitializeWeightsProduct)
{
    this->dual_u.fill(
        std::numeric_limits<typename TestFixture::real_type>::infinity());

    gko::experimental::reorder::mc64::initialize_weights(
        this->mtx.get(), this->weights, this->dual_u, this->row_maxima,
        gko::experimental::reorder::mc64_strategy::max_diagonal_product);

    this->assert_array_near(this->weights, this->initialized_weights_product,
                            "weights");
    this->assert_array_near(this->dual_u, this->initialized_dual_u_product,
                            "dual_u");
    this->assert_array_near(this->row_maxima,
                            this->initialized_row_maxima_product, "row_maxima");
}


TYPED_TEST(Mc64, InitialMatching)
{
    const auto num_rows = this->mtx->get_size()[0];

    gko::experimental::reorder::mc64::initial_matching(
        num_rows, this->mtx->get_const_row_ptrs(),
        this->mtx->get_const_col_idxs(), this->initialized_weights_sum,
        this->initialized_dual_u_sum, this->empty_permutation,
        this->empty_inverse_permutation, this->empty_matched_idxs,
        this->empty_unmatched_rows, this->zero_tol);

    GKO_ASSERT_ARRAY_EQ(this->empty_permutation,
                        this->initial_matching_permutation);
    GKO_ASSERT_ARRAY_EQ(this->empty_inverse_permutation,
                        this->initial_matching_inverse_permutation);
    GKO_ASSERT_ARRAY_EQ(this->empty_matched_idxs, this->initial_matched_idxs);
    GKO_ASSERT_ARRAY_EQ(this->empty_unmatched_rows,
                        this->initial_unmatched_rows);
}


TYPED_TEST(Mc64, ShortestAugmentingPath)
{
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    gko::addressable_priority_queue<real_type, index_type> Q{
        this->ref, this->mtx->get_size()[0]};
    std::vector<index_type> q_j{};

    gko::experimental::reorder::mc64::shortest_augmenting_path(
        this->mtx->get_size()[0], this->mtx->get_const_row_ptrs(),
        this->mtx->get_const_col_idxs(), this->initialized_weights_sum,
        this->initialized_dual_u_sum, this->initialized_distance,
        this->initial_matching_permutation,
        this->initial_matching_inverse_permutation, 4 * gko::one<index_type>(),
        this->initial_parents, this->initial_generation,
        this->initial_marked_cols, this->initial_matched_idxs, Q, q_j,
        this->zero_tol);

    GKO_ASSERT_ARRAY_EQ(this->initial_matching_permutation,
                        this->final_permutation);
    GKO_ASSERT_ARRAY_EQ(this->initial_matching_inverse_permutation,
                        this->final_inverse_permutation);
    GKO_ASSERT_ARRAY_EQ(this->initial_parents, this->final_parents);
    GKO_ASSERT_ARRAY_EQ(this->initial_generation, this->final_generation);
    GKO_ASSERT_ARRAY_EQ(this->initial_marked_cols, this->final_marked_cols);
    GKO_ASSERT_ARRAY_EQ(this->initial_matched_idxs, this->final_matched_idxs);
    this->assert_array_near(this->initialized_weights_sum, this->final_weights,
                            "weights");
    this->assert_array_near(this->initialized_dual_u_sum, this->final_dual_u,
                            "dual_u");
    this->assert_array_near(this->initialized_distance, this->final_distance,
                            "distance");
}


TYPED_TEST(Mc64, CreatesCorrectPermutationAndScalingExampleSum)
{
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    using value_type = typename TestFixture::value_type;
    auto mc64_factory =
        gko::experimental::reorder::Mc64<value_type, index_type>::build()
            .with_strategy(
                gko::experimental::reorder::mc64_strategy::max_diagonal_sum)
            .on(this->ref);

    auto mc64 = mc64_factory->generate(this->mtx);

    auto perm_obj = this->unpack(mc64.get()).first;
    auto perm = perm_obj->get_const_permutation();
    ASSERT_EQ(perm[0], 1);
    ASSERT_EQ(perm[1], 0);
    ASSERT_EQ(perm[2], 5);
    ASSERT_EQ(perm[3], 2);
    ASSERT_EQ(perm[4], 4);
    ASSERT_EQ(perm[5], 3);
}


TYPED_TEST(Mc64, CreatesCorrectPermutationAndScalingExampleProduct)
{
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    auto mc64_factory =
        gko::experimental::reorder::Mc64<value_type, index_type>::build()
            .with_strategy(
                gko::experimental::reorder::mc64_strategy::max_diagonal_product)
            .on(this->ref);
    auto mc64 = mc64_factory->generate(this->mtx);

    auto perm = this->unpack(mc64.get()).first->get_const_permutation();
    auto row_scaling =
        this->unpack(mc64.get()).first->get_const_scaling_factors();
    auto col_scaling =
        this->unpack(mc64.get()).second->get_const_scaling_factors();

    ASSERT_EQ(perm[0], 1);
    ASSERT_EQ(perm[1], 5);
    ASSERT_EQ(perm[2], 3);
    ASSERT_EQ(perm[3], 4);
    ASSERT_EQ(perm[4], 0);
    ASSERT_EQ(perm[5], 2);
    GKO_ASSERT_NEAR(row_scaling[0], value_type{1. / 3.}, r<value_type>::value);
    GKO_ASSERT_NEAR(row_scaling[1], value_type{0.2}, r<value_type>::value);
    GKO_ASSERT_NEAR(row_scaling[2], value_type{0.2}, r<value_type>::value);
    GKO_ASSERT_NEAR(row_scaling[3], value_type{4. / 15.}, r<value_type>::value);
    GKO_ASSERT_NEAR(row_scaling[4], value_type{0.3}, r<value_type>::value);
    GKO_ASSERT_NEAR(row_scaling[5], value_type{2. / 15.}, r<value_type>::value);
    GKO_ASSERT_NEAR(col_scaling[0], value_type{1.}, r<value_type>::value);
    GKO_ASSERT_NEAR(col_scaling[1], value_type{1.5}, r<value_type>::value);
    GKO_ASSERT_NEAR(col_scaling[2], value_type{0.9375}, r<value_type>::value);
    GKO_ASSERT_NEAR(col_scaling[3], value_type{5. / 6.}, r<value_type>::value);
    GKO_ASSERT_NEAR(col_scaling[4], value_type{1.}, r<value_type>::value);
    GKO_ASSERT_NEAR(col_scaling[5], value_type{1.25}, r<value_type>::value);
}


TYPED_TEST(Mc64, CreatesCorrectPermutationAndScalingLargeTrivialExampleProduct)
{
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    using value_type = typename TestFixture::value_type;
    using matrix_type = typename TestFixture::matrix_type;
    using perm_type = typename TestFixture::perm_type;
    // read input data
    std::ifstream mtx_stream{gko::matrices::location_1138_bus_mtx};
    auto mtx = gko::share(gko::read<matrix_type>(mtx_stream, this->ref));
    std::ifstream result_stream{gko::matrices::location_1138_bus_mc64_result};
    auto expected_result = gko::read<matrix_type>(result_stream, this->ref);
    // compute mc64
    auto mc64_factory =
        gko::experimental::reorder::Mc64<value_type, index_type>::build()
            .with_strategy(
                gko::experimental::reorder::mc64_strategy::max_diagonal_product)
            .on(this->ref);
    auto mc64 = mc64_factory->generate(mtx);
    // get components
    auto row_perm = gko::as<perm_type>(mc64->get_operators()[0]);
    auto col_perm = gko::as<perm_type>(mc64->get_operators()[1]);

    mtx = mtx->scale_permute(row_perm, col_perm);

    GKO_ASSERT_MTX_NEAR(mtx, expected_result, r<value_type>::value);
}


TYPED_TEST(Mc64, CreatesCorrectPermutationAndScalingLargeExampleProduct)
{
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    using value_type = typename TestFixture::value_type;
    using matrix_type = typename TestFixture::matrix_type;
    using perm_type = typename TestFixture::perm_type;
    // read input data
    std::ifstream mtx_stream{gko::matrices::location_nontrivial_mc64_example};
    auto mtx = gko::share(gko::read<matrix_type>(mtx_stream, this->ref));
    mtx->sort_by_column_index();
    std::ifstream result_stream{gko::matrices::location_nontrivial_mc64_result};
    auto expected_result = gko::read<matrix_type>(result_stream, this->ref);
    // compute mc64
    auto mc64_factory =
        gko::experimental::reorder::Mc64<value_type, index_type>::build()
            .with_strategy(
                gko::experimental::reorder::mc64_strategy::max_diagonal_product)
            .on(this->ref);
    auto mc64 = mc64_factory->generate(mtx);
    // get components
    auto row_perm = gko::as<perm_type>(mc64->get_operators()[0]);
    auto col_perm = gko::as<perm_type>(mc64->get_operators()[1]);

    mtx = mtx->scale_permute(row_perm, col_perm);

    GKO_ASSERT_MTX_NEAR(mtx, expected_result, 1e-6);
    GKO_ASSERT_MTX_EQ_SPARSITY(mtx, expected_result);
}


}  // namespace
