// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <initializer_list>
#include <memory>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/reorder/amd.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/unsort_matrix.hpp"
#include "matrices/config.hpp"
#include "test/utils/executor.hpp"


template <typename ValueIndexType>
class Amd : public CommonTestFixture {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using matrix_type = gko::matrix::Csr<value_type, index_type>;
    using reorder_type = gko::experimental::reorder::Amd<index_type>;

    Amd() : rng{63420}
    {
        std::ifstream stream{gko::matrices::location_ani4_mtx};
        mtx = gko::read<matrix_type>(stream, ref);
        dmtx = gko::clone(exec, mtx);
    }

    std::default_random_engine rng;
    std::shared_ptr<matrix_type> mtx;
    std::shared_ptr<matrix_type> dmtx;
};

TYPED_TEST_SUITE(Amd, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(Amd, IsEquivalentToRef)
{
    using reorder_type = typename TestFixture::reorder_type;
    auto factory = reorder_type::build().on(this->ref);
    auto dfactory = reorder_type::build().on(this->exec);

    auto perm = factory->generate(this->mtx);
    auto dperm = dfactory->generate(this->dmtx);

    auto perm_array = gko::make_array_view(this->ref, this->mtx->get_size()[0],
                                           perm->get_permutation());
    auto dperm_array = gko::make_array_view(
        this->exec, this->mtx->get_size()[0], dperm->get_permutation());
    GKO_ASSERT_ARRAY_EQ(perm_array, dperm_array);
}


TYPED_TEST(Amd, IsEquivalentToRefWithSkipSorting)
{
    using reorder_type = typename TestFixture::reorder_type;
    auto factory = reorder_type::build().on(this->ref);
    auto dfactory =
        reorder_type::build().with_skip_sorting(true).on(this->exec);

    auto perm = factory->generate(this->mtx);
    auto dperm = dfactory->generate(this->dmtx);

    auto perm_array = gko::make_array_view(this->ref, this->mtx->get_size()[0],
                                           perm->get_permutation());
    auto dperm_array = gko::make_array_view(
        this->exec, this->mtx->get_size()[0], dperm->get_permutation());
    GKO_ASSERT_ARRAY_EQ(perm_array, dperm_array);
}


TYPED_TEST(Amd, IsEquivalentToRefWithSkipSymmetrizeSorting)
{
    using reorder_type = typename TestFixture::reorder_type;
    auto factory = reorder_type::build().on(this->ref);
    auto dfactory = reorder_type::build()
                        .with_skip_sorting(true)
                        .with_skip_symmetrize(true)
                        .on(this->exec);

    auto perm = factory->generate(this->mtx);
    auto dperm = dfactory->generate(this->dmtx);

    auto perm_array = gko::make_array_view(this->ref, this->mtx->get_size()[0],
                                           perm->get_permutation());
    auto dperm_array = gko::make_array_view(
        this->exec, this->mtx->get_size()[0], dperm->get_permutation());
    GKO_ASSERT_ARRAY_EQ(perm_array, dperm_array);
}


TYPED_TEST(Amd, IsEquivalentToRefWithSkipSymmetrize)
{
    using reorder_type = typename TestFixture::reorder_type;
    auto factory = reorder_type::build().on(this->ref);
    auto dfactory =
        reorder_type::build().with_skip_symmetrize(true).on(this->exec);

    auto perm = factory->generate(this->mtx);
    auto dperm = dfactory->generate(this->dmtx);

    auto perm_array = gko::make_array_view(this->ref, this->mtx->get_size()[0],
                                           perm->get_permutation());
    auto dperm_array = gko::make_array_view(
        this->exec, this->mtx->get_size()[0], dperm->get_permutation());
    GKO_ASSERT_ARRAY_EQ(perm_array, dperm_array);
}


TYPED_TEST(Amd, IsEquivalentToRefUnsorted)
{
    using reorder_type = typename TestFixture::reorder_type;
    auto factory = reorder_type::build().on(this->ref);
    auto dfactory = reorder_type::build().on(this->exec);
    gko::test::unsort_matrix(this->dmtx, this->rng);

    auto perm = factory->generate(this->mtx);
    auto dperm = dfactory->generate(this->dmtx);

    auto perm_array = gko::make_array_view(this->ref, this->mtx->get_size()[0],
                                           perm->get_permutation());
    auto dperm_array = gko::make_array_view(
        this->exec, this->mtx->get_size()[0], dperm->get_permutation());
    GKO_ASSERT_ARRAY_EQ(perm_array, dperm_array);
}


TYPED_TEST(Amd, IsEquivalentToRefUnsortedSkipSymmetrize)
{
    using reorder_type = typename TestFixture::reorder_type;
    auto factory = reorder_type::build().on(this->ref);
    auto dfactory =
        reorder_type::build().with_skip_symmetrize(true).on(this->exec);
    gko::test::unsort_matrix(this->dmtx, this->rng);

    auto perm = factory->generate(this->mtx);
    auto dperm = dfactory->generate(this->dmtx);

    auto perm_array = gko::make_array_view(this->ref, this->mtx->get_size()[0],
                                           perm->get_permutation());
    auto dperm_array = gko::make_array_view(
        this->exec, this->mtx->get_size()[0], dperm->get_permutation());
    GKO_ASSERT_ARRAY_EQ(perm_array, dperm_array);
}
