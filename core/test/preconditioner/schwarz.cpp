/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include <ginkgo/core/preconditioner/schwarz.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/matrix/csr.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class SchwarzFactory : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Schwarz = gko::preconditioner::Schwarz<value_type, index_type>;

    SchwarzFactory()
        : exec(gko::ReferenceExecutor::create()),
          schwarz_factory(
              Schwarz::build()
                  .with_subdomain_sizes(std::vector<gko::size_type>{3, 2, 1})
                  .on(exec)),
          mtx(gko::matrix::Csr<value_type, index_type>::create(
              exec, gko::dim<2>{5, 5}, 13))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<typename Schwarz::Factory> schwarz_factory;
    std::shared_ptr<gko::matrix::Csr<value_type, index_type>> mtx;
};

TYPED_TEST_SUITE(SchwarzFactory, gko::test::ValueIndexTypes);


TYPED_TEST(SchwarzFactory, KnowsItsExecutor)
{
    ASSERT_EQ(this->schwarz_factory->get_executor(), this->exec);
}


TYPED_TEST(SchwarzFactory, KnowsNumSubdomains)
{
    ASSERT_EQ(this->schwarz_factory->get_parameters().subdomain_sizes[0], 3);
    ASSERT_EQ(this->schwarz_factory->get_parameters().subdomain_sizes[1], 2);
    ASSERT_EQ(this->schwarz_factory->get_parameters().subdomain_sizes[2], 1);
}


}  // namespace
