// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/reorder/nested_dissection.hpp>


#include <memory>


#include <gtest/gtest.h>
#include GKO_METIS_HEADER


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"


namespace {


template <typename IndexType>
class NestedDissection : public ::testing::Test {
protected:
    using value_type = double;
    using index_type = IndexType;
    using reorder_type =
        gko::experimental::reorder::NestedDissection<value_type, index_type>;
    using Mtx = gko::matrix::Dense<value_type>;
    NestedDissection()
        : exec(gko::ReferenceExecutor::create()),
          nd_factory(reorder_type::build().on(exec)),
          star_mtx{gko::initialize<Mtx>({{1.0, 1.0, 1.0, 1.0},
                                         {1.0, 1.0, 0.0, 0.0},
                                         {1.0, 0.0, 1.0, 0.0},
                                         {1.0, 0.0, 0.0, 1.0}},
                                        exec)}
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> star_mtx;
    std::unique_ptr<reorder_type> nd_factory;
};

TYPED_TEST_SUITE(NestedDissection, gko::test::IndexTypes,
                 TypenameNameGenerator);


TYPED_TEST(NestedDissection, HasSensibleDefaults)
{
    using reorder_type = typename TestFixture::reorder_type;

    auto factory = reorder_type::build().on(this->exec);

    ASSERT_TRUE(factory->get_parameters().options.empty());
}


TYPED_TEST(NestedDissection, FailsWithInvalidOption)
{
    using value_type = typename TestFixture::value_type;
    using reorder_type = typename TestFixture::reorder_type;
    auto factory = reorder_type::build()
                       .with_options({{METIS_NOPTIONS, 0}})
                       .on(this->exec);

    ASSERT_THROW(factory->generate(this->star_mtx), gko::MetisError);
}


TYPED_TEST(NestedDissection, FailsWithOneBasedIndexing)
{
    using value_type = typename TestFixture::value_type;
    using reorder_type = typename TestFixture::reorder_type;
    auto factory = reorder_type::build()
                       .with_options({{METIS_OPTION_NUMBERING, 1}})
                       .on(this->exec);

    ASSERT_THROW(factory->generate(this->star_mtx), gko::MetisError);
}


TYPED_TEST(NestedDissection, ComputesSensiblePermutation)
{
    auto perm = this->nd_factory->generate(this->star_mtx);

    auto perm_array = gko::make_array_view(this->exec, perm->get_size()[0],
                                           perm->get_permutation());
    auto permuted = gko::as<typename TestFixture::Mtx>(
        this->star_mtx->permute(&perm_array));
    GKO_ASSERT_MTX_NEAR(permuted,
                        I<I<double>>({{1.0, 0.0, 0.0, 1.0},
                                      {0.0, 1.0, 0.0, 1.0},
                                      {0.0, 0.0, 1.0, 1.0},
                                      {1.0, 1.0, 1.0, 1.0}}),
                        0.0);
}


}  // namespace
