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

#include <ginkgo/core/reorder/amd.hpp>


#include <algorithm>
#include <initializer_list>
#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/matrix/csr.hpp>


#include "core/factorization/symbolic.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"
#include "matrices/config.hpp"


template <typename ValueIndexType>
class Amd : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using matrix_type = gko::matrix::Csr<value_type, index_type>;

    Amd() : ref(gko::ReferenceExecutor::create()), permutation_ref{ref} {}

    void setup(
        std::initializer_list<std::initializer_list<value_type>> mtx_list,
        std::initializer_list<index_type> permutation, int fillin_reduction)
    {
        mtx = gko::initialize<matrix_type>(mtx_list, ref);
        permutation_ref = gko::array<index_type>{ref, permutation};
        num_rows = mtx->get_size()[0];
        this->fillin_reduction = fillin_reduction;
    }

    void setup(const char* name_mtx,
               std::initializer_list<index_type> permutation,
               int fillin_reduction)
    {
        std::ifstream stream{name_mtx};
        mtx = gko::read<matrix_type>(stream, this->ref);
        permutation_ref = gko::array<index_type>{ref, permutation};
        num_rows = mtx->get_size()[0];
        this->fillin_reduction = fillin_reduction;
    }

    void forall_matrices(std::function<void()> fn)
    {
        {
            SCOPED_TRACE("ani1");
            this->setup(gko::matrices::location_ani1_mtx,
                        {0,  2,  6,  1,  12, 20, 11, 5,  4,  10, 35, 34,
                         31, 27, 32, 25, 30, 28, 33, 26, 19, 14, 3,  7,
                         23, 16, 15, 22, 17, 8,  21, 13, 18, 24, 29, 9},
                        60);
            fn();
        }
        {
            SCOPED_TRACE("ani1_amd");
            this->setup(gko::matrices::location_ani1_amd_mtx,
                        {20, 19, 22, 21, 23, 6,  9,  8,  7,  1,  4,  0,
                         3,  2,  5,  15, 13, 12, 11, 10, 14, 18, 16, 17,
                         32, 31, 33, 30, 29, 24, 25, 26, 28, 35, 34, 27},
                        -10);
            fn();
        }
        {
            SCOPED_TRACE("example");
            this->setup({{1, 0, 1, 0, 0, 0, 0, 1, 0, 0},
                         {0, 1, 0, 0, 1, 0, 0, 0, 0, 1},
                         {1, 0, 1, 0, 0, 0, 1, 0, 0, 0},
                         {0, 0, 0, 1, 0, 0, 0, 0, 1, 1},
                         {0, 1, 0, 0, 1, 0, 0, 0, 1, 1},
                         {0, 0, 0, 0, 0, 1, 1, 1, 0, 0},
                         {0, 0, 1, 0, 0, 1, 1, 0, 0, 0},
                         {1, 0, 0, 0, 0, 1, 0, 1, 1, 1},
                         {0, 0, 0, 1, 1, 0, 0, 1, 1, 0},
                         {0, 1, 0, 1, 1, 0, 0, 1, 0, 1}},
                        {6, 5, 0, 2, 7, 3, 8, 1, 9, 4}, 0);
            fn();
        }
        {
            SCOPED_TRACE("separable");
            this->setup({{1, 0, 1, 0, 0, 0, 0, 0, 0, 0},
                         {0, 1, 1, 0, 0, 0, 0, 0, 0, 0},
                         {1, 1, 1, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 1, 1, 0, 0, 0, 0, 0},
                         {0, 0, 0, 1, 1, 1, 0, 0, 0, 1},
                         {0, 0, 0, 0, 1, 1, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 1, 1, 0, 1},
                         {0, 0, 0, 0, 0, 0, 1, 1, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 1, 1},
                         {0, 0, 0, 0, 1, 0, 1, 0, 1, 1}},
                        {1, 0, 2, 8, 9, 7, 6, 3, 5, 4}, 0);
            fn();
        }
        {
            SCOPED_TRACE("missing diagonal");
            this->setup({{1, 0, 1, 0, 0, 0, 0, 0, 0, 0},
                         {0, 1, 1, 0, 0, 0, 0, 0, 0, 0},
                         {1, 1, 0, 1, 0, 0, 0, 0, 0, 0},
                         {0, 0, 1, 1, 1, 0, 0, 0, 0, 0},
                         {0, 0, 0, 1, 0, 1, 0, 0, 0, 0},
                         {0, 0, 0, 0, 1, 1, 1, 0, 0, 0},
                         {0, 0, 0, 0, 0, 1, 1, 1, 0, 1},
                         {0, 0, 0, 0, 0, 0, 1, 1, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 1, 1},
                         {0, 0, 0, 0, 0, 0, 1, 0, 1, 0}},
                        {0, 1, 3, 7, 5, 6, 8, 2, 4, 9}, -5);
            fn();
        }
    }

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    int fillin_reduction;
    gko::size_type num_rows;
    gko::array<index_type> permutation_ref;
    std::shared_ptr<matrix_type> mtx;
};

TYPED_TEST_SUITE(Amd, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(Amd, WorksSymmetric)
{
    using matrix_type = typename TestFixture::matrix_type;
    using index_type = typename TestFixture::index_type;
    this->forall_matrices([this] {
        auto amd =
            gko::reorder::Amd<index_type>::build().with_symmetric(true).on(
                this->ref);

        auto perm = amd->generate(this->mtx);

        auto perm_array = gko::make_array_view(this->ref, this->num_rows,
                                               perm->get_permutation());
        GKO_ASSERT_ARRAY_EQ(perm_array, this->permutation_ref);
    });
}


TYPED_TEST(Amd, ReducesFillInSymmetric)
{
    using matrix_type = typename TestFixture::matrix_type;
    using index_type = typename TestFixture::index_type;
    this->forall_matrices([this] {
        auto amd =
            gko::reorder::Amd<index_type>::build().with_symmetric(true).on(
                this->ref);

        auto perm = amd->generate(this->mtx);

        auto perm_array = gko::make_array_view(this->ref, this->num_rows,
                                               perm->get_permutation());
        auto permuted_mtx =
            gko::as<matrix_type>(this->mtx->permute(&perm_array));
        std::unique_ptr<gko::factorization::elimination_forest<index_type>>
            forest;
        std::unique_ptr<matrix_type> factorized_mtx;
        std::unique_ptr<matrix_type> factorized_permuted_mtx;
        gko::factorization::symbolic_cholesky(this->mtx.get(), true,
                                              factorized_mtx, forest);
        gko::factorization::symbolic_cholesky(permuted_mtx.get(), true,
                                              factorized_permuted_mtx, forest);
        int fillin_mtx = factorized_mtx->get_num_stored_elements() -
                         this->mtx->get_num_stored_elements();
        int fillin_permuted =
            factorized_permuted_mtx->get_num_stored_elements() -
            permuted_mtx->get_num_stored_elements();
        ASSERT_LE(fillin_permuted, fillin_mtx - this->fillin_reduction);
    });
}
