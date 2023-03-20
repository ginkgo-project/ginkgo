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
