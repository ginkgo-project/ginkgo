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

#include "core/factorization/elimination_forest.hpp"


#include <algorithm>
#include <numeric>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>


#include "core/test/utils.hpp"
#include "matrices/config.hpp"


namespace {


template <typename ValueIndexType>
class EliminationForest : public ::testing::Test {
public:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using matrix_type = gko::matrix::Csr<value_type, index_type>;

protected:
    EliminationForest() : ref(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<const gko::ReferenceExecutor> ref;
};

TYPED_TEST_SUITE(EliminationForest, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(EliminationForest, WorksForExample)
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

    GKO_ASSERT_ARRAY_EQ(forest->parents,
                        I<index_type>({2, 4, 6, 8, 8, 6, 7, 8, 9, 10}));
    GKO_ASSERT_ARRAY_EQ(forest->child_ptrs,
                        I<index_type>({0, 0, 0, 1, 1, 2, 2, 4, 5, 8, 9, 10}));
    GKO_ASSERT_ARRAY_EQ(forest->children,
                        I<index_type>({0, 1, 2, 5, 6, 3, 4, 7, 8, 9}));
    GKO_ASSERT_ARRAY_EQ(forest->postorder,
                        I<index_type>({3, 1, 4, 0, 2, 5, 6, 7, 8, 9}));
    GKO_ASSERT_ARRAY_EQ(forest->inv_postorder,
                        I<index_type>({3, 1, 4, 0, 2, 5, 6, 7, 8, 9}));
    GKO_ASSERT_ARRAY_EQ(forest->postorder_parents,
                        I<index_type>({8, 2, 8, 4, 6, 6, 7, 8, 9, 10}));
}


TYPED_TEST(EliminationForest, WorksForSeparable)
{
    using matrix_type = typename TestFixture::matrix_type;
    using index_type = typename TestFixture::index_type;
    auto mtx = gko::initialize<typename TestFixture::matrix_type>(
        {
            {1, 0, 1, 0, 0, 0, 0, 0, 0, 0},
            {0, 1, 1, 0, 0, 0, 0, 0, 0, 0},
            {1, 1, 1, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 1, 1, 0, 0, 0, 0, 0},
            {0, 0, 0, 1, 1, 1, 0, 0, 0, 1},
            {0, 0, 0, 0, 1, 1, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 1, 0, 0, 1},
            {0, 0, 0, 0, 0, 0, 1, 1, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 1, 1},
            {0, 0, 0, 0, 1, 0, 1, 0, 1, 1},
        },
        this->ref);

    std::unique_ptr<gko::factorization::elimination_forest<index_type>> forest;
    gko::factorization::compute_elim_forest(mtx.get(), forest);

    GKO_ASSERT_ARRAY_EQ(forest->parents,
                        I<index_type>({2, 2, 10, 4, 5, 9, 7, 9, 9, 10}));
    GKO_ASSERT_ARRAY_EQ(forest->child_ptrs,
                        I<index_type>({0, 0, 0, 2, 2, 3, 4, 4, 5, 5, 8, 10}));
    GKO_ASSERT_ARRAY_EQ(forest->children,
                        I<index_type>({0, 1, 3, 4, 6, 5, 7, 8, 2, 9}));
    GKO_ASSERT_ARRAY_EQ(forest->postorder,
                        I<index_type>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
    GKO_ASSERT_ARRAY_EQ(forest->inv_postorder,
                        I<index_type>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
    GKO_ASSERT_ARRAY_EQ(forest->postorder_parents,
                        I<index_type>({2, 2, 10, 4, 5, 9, 7, 9, 9, 10}));
}


TYPED_TEST(EliminationForest, WorksForPostOrderNotSelfInverse)
{
    using matrix_type = typename TestFixture::matrix_type;
    using index_type = typename TestFixture::index_type;
    auto mtx = gko::initialize<typename TestFixture::matrix_type>(
        {
            {1, 0, 1, 0, 0, 0, 0, 0, 0, 0},
            {0, 1, 0, 0, 1, 0, 0, 0, 0, 0},
            {1, 0, 1, 0, 0, 0, 1, 0, 0, 0},
            {0, 0, 0, 1, 0, 0, 0, 0, 1, 0},
            {0, 1, 0, 0, 1, 1, 0, 0, 0, 0},
            {0, 0, 0, 0, 1, 1, 1, 0, 0, 0},
            {0, 0, 1, 0, 0, 1, 1, 1, 0, 0},
            {0, 0, 0, 0, 0, 0, 1, 1, 1, 0},
            {0, 0, 0, 1, 0, 0, 0, 1, 1, 1},
            {0, 0, 0, 0, 0, 0, 1, 0, 1, 1},
        },
        this->ref);
    std::unique_ptr<gko::factorization::elimination_forest<index_type>> forest;
    gko::factorization::compute_elim_forest(mtx.get(), forest);
    GKO_ASSERT_ARRAY_EQ(forest->parents,
                        I<index_type>({2, 4, 6, 8, 5, 6, 7, 8, 9, 10}));
    GKO_ASSERT_ARRAY_EQ(forest->child_ptrs,
                        I<index_type>({0, 0, 0, 1, 1, 2, 3, 5, 6, 8, 9, 10}));
    GKO_ASSERT_ARRAY_EQ(forest->children,
                        I<index_type>({0, 1, 4, 2, 5, 6, 3, 7, 8, 9}));
    GKO_ASSERT_ARRAY_EQ(forest->postorder,
                        I<index_type>({3, 0, 2, 1, 4, 5, 6, 7, 8, 9}));
    GKO_ASSERT_ARRAY_EQ(forest->inv_postorder,
                        I<index_type>({1, 3, 2, 0, 4, 5, 6, 7, 8, 9}));
    GKO_ASSERT_ARRAY_EQ(forest->postorder_parents,
                        I<index_type>({8, 2, 6, 4, 5, 6, 7, 8, 9, 10}));
}


TYPED_TEST(EliminationForest, WorksForAni1)
{
    using matrix_type = typename TestFixture::matrix_type;
    using index_type = typename TestFixture::index_type;
    std::ifstream stream{gko::matrices::location_ani1_mtx};
    auto mtx = gko::read<matrix_type>(stream, this->ref);

    std::unique_ptr<gko::factorization::elimination_forest<index_type>> forest;
    gko::factorization::compute_elim_forest(mtx.get(), forest);

    // the elimination tree is a path
    gko::array<index_type> iota_arr{this->ref, 36};
    gko::array<index_type> iota_arr2{this->ref, 38};
    std::iota(iota_arr.get_data(), iota_arr.get_data() + 36, 1);
    std::iota(iota_arr2.get_data() + 1, iota_arr2.get_data() + 38, 0);
    iota_arr2.get_data()[0] = 0;
    GKO_ASSERT_ARRAY_EQ(forest->parents, iota_arr);
    GKO_ASSERT_ARRAY_EQ(forest->postorder_parents, iota_arr);
    GKO_ASSERT_ARRAY_EQ(forest->child_ptrs, iota_arr2);
    std::iota(iota_arr.get_data(), iota_arr.get_data() + 36, 0);
    GKO_ASSERT_ARRAY_EQ(forest->children, iota_arr);
    GKO_ASSERT_ARRAY_EQ(forest->postorder, iota_arr);
    GKO_ASSERT_ARRAY_EQ(forest->inv_postorder, iota_arr);
}


TYPED_TEST(EliminationForest, WorksForAni1Amd)
{
    using matrix_type = typename TestFixture::matrix_type;
    using index_type = typename TestFixture::index_type;
    std::ifstream stream{gko::matrices::location_ani1_amd_mtx};
    auto mtx = gko::read<matrix_type>(stream, this->ref);

    std::unique_ptr<gko::factorization::elimination_forest<index_type>> forest;
    gko::factorization::compute_elim_forest(mtx.get(), forest);

    GKO_ASSERT_ARRAY_EQ(
        forest->parents,
        I<index_type>({4,  2,  3,  4,  5,  29, 7,  8,  9,  27, 11, 12,
                       13, 14, 16, 16, 17, 18, 24, 20, 21, 22, 23, 24,
                       25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36}));
    GKO_ASSERT_ARRAY_EQ(
        forest->child_ptrs,
        I<index_type>({0,  0,  0,  1,  2,  4,  5,  5,  6,  7,  8,  8,  9,
                       10, 11, 12, 12, 14, 15, 16, 16, 17, 18, 19, 20, 22,
                       23, 24, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36}));
    GKO_ASSERT_ARRAY_EQ(
        forest->children,
        I<index_type>({1,  2,  0,  3,  4,  6,  7,  8,  10, 11, 12, 13,
                       14, 15, 16, 17, 19, 20, 21, 22, 18, 23, 24, 25,
                       9,  26, 27, 5,  28, 29, 30, 31, 32, 33, 34, 35}));
    gko::array<index_type> iota_arr{this->ref, 36};
    std::iota(iota_arr.get_data(), iota_arr.get_data() + 36, 0);
    GKO_ASSERT_ARRAY_EQ(forest->postorder, iota_arr);
    GKO_ASSERT_ARRAY_EQ(forest->inv_postorder, iota_arr);
    GKO_ASSERT_ARRAY_EQ(
        forest->postorder_parents,
        I<index_type>({4,  2,  3,  4,  5,  29, 7,  8,  9,  27, 11, 12,
                       13, 14, 16, 16, 17, 18, 24, 20, 21, 22, 23, 24,
                       25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36}));
}


}  // namespace
