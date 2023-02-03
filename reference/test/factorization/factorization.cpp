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

#include <ginkgo/core/factorization/factorization.hpp>


#include <algorithm>
#include <fstream>
#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"


template <typename ValueIndexType>
class Factorization : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using vector_type = gko::matrix::Dense<value_type>;
    using matrix_type = gko::matrix::Csr<value_type, index_type>;
    using diag_type = gko::matrix::Diagonal<value_type>;
    using composition_type = gko::Composition<value_type>;
    using factorization_type =
        gko::experimental::factorization::Factorization<value_type, index_type>;

    Factorization()
        : ref(gko::ReferenceExecutor::create()),
          lower_mtx{gko::initialize<matrix_type>(
              {{1.0, 0.0, 0.0}, {3.0, 1.0, 0.0}, {1.0, 2.0, 1.0}}, ref)},
          diagonal{diag_type::create(ref, 3)},
          upper_mtx(gko::initialize<matrix_type>(
              {{1.0, 2.0, 1.0}, {0.0, 1.0, 3.0}, {0.0, 0.0, 1.0}}, ref)),
          combined_mtx(gko::initialize<matrix_type>(
              {{1.0, 2.0, 1.0}, {3.0, -1.0, 3.0}, {1.0, 2.0, 5.0}}, ref)),
          input(gko::initialize<vector_type>({1.0, 2.0, 3.0}, ref)),
          output(gko::initialize<vector_type>({0.0, 0.0, 0.0}, ref)),
          alpha(gko::initialize<vector_type>({2.5}, ref)),
          beta(gko::initialize<vector_type>({-3.5}, ref))
    {
        diagonal->get_values()[0] = 1.0;
        diagonal->get_values()[1] = -1.0;
        diagonal->get_values()[2] = 5.0;
    }

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    std::shared_ptr<matrix_type> lower_mtx;
    std::shared_ptr<diag_type> diagonal;
    std::shared_ptr<matrix_type> upper_mtx;
    std::shared_ptr<matrix_type> combined_mtx;
    std::shared_ptr<vector_type> input;
    std::shared_ptr<vector_type> output;
    std::shared_ptr<vector_type> alpha;
    std::shared_ptr<vector_type> beta;
};

TYPED_TEST_SUITE(Factorization, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(Factorization, CreateCompositionWorks)
{
    using factorization_type = typename TestFixture::factorization_type;
    using composition_type = typename TestFixture::composition_type;

    auto fact = factorization_type::create_from_composition(
        composition_type::create(this->lower_mtx, this->upper_mtx));

    ASSERT_EQ(fact->get_size(), this->lower_mtx->get_size());
    ASSERT_EQ(fact->get_storage_type(),
              gko::experimental::factorization::storage_type::composition);
    ASSERT_EQ(fact->get_lower_factor(), this->lower_mtx);
    ASSERT_EQ(fact->get_diagonal(), nullptr);
    ASSERT_EQ(fact->get_upper_factor(), this->upper_mtx);
    ASSERT_EQ(fact->get_combined(), nullptr);
}


TYPED_TEST(Factorization, CreateCompositionWithDiagonalWorks)
{
    using factorization_type = typename TestFixture::factorization_type;
    using composition_type = typename TestFixture::composition_type;

    auto fact =
        factorization_type::create_from_composition(composition_type::create(
            this->lower_mtx, this->diagonal, this->upper_mtx));

    ASSERT_EQ(fact->get_size(), this->lower_mtx->get_size());
    ASSERT_EQ(fact->get_storage_type(),
              gko::experimental::factorization::storage_type::composition);
    ASSERT_EQ(fact->get_lower_factor(), this->lower_mtx);
    ASSERT_EQ(fact->get_diagonal(), this->diagonal);
    ASSERT_EQ(fact->get_upper_factor(), this->upper_mtx);
    ASSERT_EQ(fact->get_combined(), nullptr);
}


TYPED_TEST(Factorization, CreateSymmCompositionWorks)
{
    using factorization_type = typename TestFixture::factorization_type;
    using composition_type = typename TestFixture::composition_type;

    auto fact = factorization_type::create_from_symm_composition(
        composition_type::create(this->lower_mtx, this->upper_mtx));

    ASSERT_EQ(fact->get_size(), this->lower_mtx->get_size());
    ASSERT_EQ(fact->get_storage_type(),
              gko::experimental::factorization::storage_type::symm_composition);
    ASSERT_EQ(fact->get_lower_factor(), this->lower_mtx);
    ASSERT_EQ(fact->get_diagonal(), nullptr);
    ASSERT_EQ(fact->get_upper_factor(), this->upper_mtx);
    ASSERT_EQ(fact->get_combined(), nullptr);
}


TYPED_TEST(Factorization, CreateSymmCompositionWithDiagonalWorks)
{
    using factorization_type = typename TestFixture::factorization_type;
    using composition_type = typename TestFixture::composition_type;

    auto fact = factorization_type::create_from_symm_composition(
        composition_type::create(this->lower_mtx, this->diagonal,
                                 this->upper_mtx));

    ASSERT_EQ(fact->get_size(), this->lower_mtx->get_size());
    ASSERT_EQ(fact->get_storage_type(),
              gko::experimental::factorization::storage_type::symm_composition);
    ASSERT_EQ(fact->get_lower_factor(), this->lower_mtx);
    ASSERT_EQ(fact->get_diagonal(), this->diagonal);
    ASSERT_EQ(fact->get_upper_factor(), this->upper_mtx);
    ASSERT_EQ(fact->get_combined(), nullptr);
}


TYPED_TEST(Factorization, CreateCombinedLUWorks)
{
    using factorization_type = typename TestFixture::factorization_type;

    auto fact = factorization_type::create_from_combined_lu(
        this->combined_mtx->clone());

    ASSERT_EQ(fact->get_size(), this->combined_mtx->get_size());
    ASSERT_EQ(fact->get_storage_type(),
              gko::experimental::factorization::storage_type::combined_lu);
    ASSERT_EQ(fact->get_lower_factor(), nullptr);
    ASSERT_EQ(fact->get_diagonal(), nullptr);
    ASSERT_EQ(fact->get_upper_factor(), nullptr);
    GKO_ASSERT_MTX_NEAR(fact->get_combined(), this->combined_mtx, 0.0);
    ASSERT_THROW(fact->apply(this->input, this->output), gko::NotSupported);
    ASSERT_THROW(
        fact->apply(this->alpha, this->input, this->beta, this->output),
        gko::NotSupported);
}


TYPED_TEST(Factorization, CreateCombinedLDUWorks)
{
    using factorization_type = typename TestFixture::factorization_type;

    auto fact = factorization_type::create_from_combined_ldu(
        this->combined_mtx->clone());

    ASSERT_EQ(fact->get_size(), this->combined_mtx->get_size());
    ASSERT_EQ(fact->get_storage_type(),
              gko::experimental::factorization::storage_type::combined_ldu);
    ASSERT_EQ(fact->get_lower_factor(), nullptr);
    ASSERT_EQ(fact->get_diagonal(), nullptr);
    ASSERT_EQ(fact->get_upper_factor(), nullptr);
    GKO_ASSERT_MTX_NEAR(fact->get_combined(), this->combined_mtx, 0.0);
    ASSERT_THROW(fact->apply(this->input, this->output), gko::NotSupported);
    ASSERT_THROW(
        fact->apply(this->alpha, this->input, this->beta, this->output),
        gko::NotSupported);
}


TYPED_TEST(Factorization, CreateSymmCombinedCholeskyWorks)
{
    using factorization_type = typename TestFixture::factorization_type;

    auto fact = factorization_type::create_from_combined_cholesky(
        this->combined_mtx->clone());

    ASSERT_EQ(fact->get_size(), this->combined_mtx->get_size());
    ASSERT_EQ(
        fact->get_storage_type(),
        gko::experimental::factorization::storage_type::symm_combined_cholesky);
    ASSERT_EQ(fact->get_lower_factor(), nullptr);
    ASSERT_EQ(fact->get_diagonal(), nullptr);
    ASSERT_EQ(fact->get_upper_factor(), nullptr);
    GKO_ASSERT_MTX_NEAR(fact->get_combined(), this->combined_mtx, 0.0);
    ASSERT_THROW(fact->apply(this->input, this->output), gko::NotSupported);
    ASSERT_THROW(
        fact->apply(this->alpha, this->input, this->beta, this->output),
        gko::NotSupported);
}


TYPED_TEST(Factorization, CreateSymmCombinedLDLWorks)
{
    using factorization_type = typename TestFixture::factorization_type;

    auto fact = factorization_type::create_from_combined_ldl(
        this->combined_mtx->clone());

    ASSERT_EQ(fact->get_size(), this->combined_mtx->get_size());
    ASSERT_EQ(
        fact->get_storage_type(),
        gko::experimental::factorization::storage_type::symm_combined_ldl);
    ASSERT_EQ(fact->get_lower_factor(), nullptr);
    ASSERT_EQ(fact->get_diagonal(), nullptr);
    ASSERT_EQ(fact->get_upper_factor(), nullptr);
    GKO_ASSERT_MTX_NEAR(fact->get_combined(), this->combined_mtx, 0.0);
    ASSERT_THROW(fact->apply(this->input, this->output), gko::NotSupported);
    ASSERT_THROW(
        fact->apply(this->alpha, this->input, this->beta, this->output),
        gko::NotSupported);
}


TYPED_TEST(Factorization, ApplyFromCompositionWorks)
{
    using factorization_type = typename TestFixture::factorization_type;
    using composition_type = typename TestFixture::composition_type;
    auto comp = composition_type::create(this->lower_mtx, this->upper_mtx);
    auto fact = factorization_type::create_from_composition(comp->clone());
    auto ref_out = this->output->clone();

    fact->apply(this->input, this->output);
    comp->apply(this->input, ref_out);

    GKO_ASSERT_MTX_NEAR(this->output, ref_out, 0.0);
}


TYPED_TEST(Factorization, ApplyFromCompositionWithDiagonalWorks)
{
    using factorization_type = typename TestFixture::factorization_type;
    using composition_type = typename TestFixture::composition_type;
    auto comp = composition_type::create(this->lower_mtx, this->diagonal,
                                         this->upper_mtx);
    auto fact = factorization_type::create_from_composition(comp->clone());
    auto ref_out = this->output->clone();

    fact->apply(this->input, this->output);
    comp->apply(this->input, ref_out);

    GKO_ASSERT_MTX_NEAR(this->output, ref_out, 0.0);
}


TYPED_TEST(Factorization, ApplyFromSymmCompositionWorks)
{
    using factorization_type = typename TestFixture::factorization_type;
    using composition_type = typename TestFixture::composition_type;
    auto comp = composition_type::create(this->lower_mtx, this->upper_mtx);
    auto fact = factorization_type::create_from_symm_composition(comp->clone());
    auto ref_out = this->output->clone();

    fact->apply(this->input, this->output);
    comp->apply(this->input, ref_out);

    GKO_ASSERT_MTX_NEAR(this->output, ref_out, 0.0);
}


TYPED_TEST(Factorization, ApplyFromSymmCompositionWithDiagonalWorks)
{
    using factorization_type = typename TestFixture::factorization_type;
    using composition_type = typename TestFixture::composition_type;
    auto comp = composition_type::create(this->lower_mtx, this->diagonal,
                                         this->upper_mtx);
    auto fact = factorization_type::create_from_symm_composition(comp->clone());
    auto ref_out = this->output->clone();

    fact->apply(this->input, this->output);
    comp->apply(this->input, ref_out);

    GKO_ASSERT_MTX_NEAR(this->output, ref_out, 0.0);
}


TYPED_TEST(Factorization, AdvancedApplyFromCompositionWorks)
{
    using factorization_type = typename TestFixture::factorization_type;
    using composition_type = typename TestFixture::composition_type;
    auto comp = composition_type::create(this->lower_mtx, this->upper_mtx);
    auto fact = factorization_type::create_from_composition(comp->clone());
    auto ref_out = this->output->clone();

    fact->apply(this->alpha, this->input, this->beta, this->output);
    comp->apply(this->alpha, this->input, this->beta, ref_out);

    GKO_ASSERT_MTX_NEAR(this->output, ref_out, 0.0);
}


TYPED_TEST(Factorization, AdvancedApplyFromCompositionWithDiagonalWorks)
{
    using factorization_type = typename TestFixture::factorization_type;
    using composition_type = typename TestFixture::composition_type;
    auto comp = composition_type::create(this->lower_mtx, this->diagonal,
                                         this->upper_mtx);
    auto fact = factorization_type::create_from_composition(comp->clone());
    auto ref_out = this->output->clone();

    fact->apply(this->alpha, this->input, this->beta, this->output);
    comp->apply(this->alpha, this->input, this->beta, ref_out);

    GKO_ASSERT_MTX_NEAR(this->output, ref_out, 0.0);
}


TYPED_TEST(Factorization, AdvancedApplyFromSymmCompositionWorks)
{
    using factorization_type = typename TestFixture::factorization_type;
    using composition_type = typename TestFixture::composition_type;
    auto comp = composition_type::create(this->lower_mtx, this->upper_mtx);
    auto fact = factorization_type::create_from_symm_composition(comp->clone());
    auto ref_out = this->output->clone();

    fact->apply(this->alpha, this->input, this->beta, this->output);
    comp->apply(this->alpha, this->input, this->beta, ref_out);

    GKO_ASSERT_MTX_NEAR(this->output, ref_out, 0.0);
}


TYPED_TEST(Factorization, AdvancedApplyFromSymmCompositionWithDiagonalWorks)
{
    using factorization_type = typename TestFixture::factorization_type;
    using composition_type = typename TestFixture::composition_type;
    auto comp = composition_type::create(this->lower_mtx, this->diagonal,
                                         this->upper_mtx);
    auto fact = factorization_type::create_from_symm_composition(comp->clone());
    auto ref_out = this->output->clone();

    fact->apply(this->alpha, this->input, this->beta, this->output);
    comp->apply(this->alpha, this->input, this->beta, ref_out);

    GKO_ASSERT_MTX_NEAR(this->output, ref_out, 0.0);
}
