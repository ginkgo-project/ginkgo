/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#include <ginkgo/core/factorization/par_ilu.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace {


class DummyLinOp : public gko::EnableLinOp<DummyLinOp>,
                   public gko::EnableCreateMethod<DummyLinOp> {
public:
    DummyLinOp(std::shared_ptr<const gko::Executor> exec,
               gko::dim<2> size = gko::dim<2>{})
        : EnableLinOp<DummyLinOp>(exec, size)
    {}

protected:
    void apply_impl(const gko::LinOp *b, gko::LinOp *x) const override {}

    void apply_impl(const gko::LinOp *alpha, const gko::LinOp *b,
                    const gko::LinOp *beta, gko::LinOp *x) const override
    {}
};


class ParIlu : public ::testing::Test {
public:
    using value_type = gko::default_precision;
    using index_type = gko::int32;
    using ilu_factory_type = gko::factorization::ParIlu<value_type, index_type>;
    using Dense = gko::matrix::Dense<value_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using Coo = gko::matrix::Coo<value_type, index_type>;

protected:
    ParIlu()
        : ref(gko::ReferenceExecutor::create()),
          linOp(DummyLinOp::create(ref)),
          dense_mtx(gko::initialize<Dense>({{1., 0.}, {0., 1.}}, ref)),
          coo_mtx(Coo::create(ref)),
          csr_mtx(Csr::create(ref))
    {
        dense_mtx->convert_to(coo_mtx.get());
        dense_mtx->convert_to(csr_mtx.get());
    }

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    std::unique_ptr<DummyLinOp> linOp;
    std::shared_ptr<const Dense> dense_mtx;
    std::shared_ptr<Coo> coo_mtx;
    std::shared_ptr<Csr> csr_mtx;
};


TEST_F(ParIlu, ThrowNotSupportedForWrongLinOp)
{
    auto factory = ilu_factory_type::build().on(ref);

    ASSERT_THROW(factory->generate(gko::share(linOp)), gko::NotSupported);
}


TEST_F(ParIlu, NoThrowCooMatrix)
{
    auto factory = ilu_factory_type::build().on(ref);

    ASSERT_NO_THROW(factory->generate(gko::share(coo_mtx)));
}


TEST_F(ParIlu, NoThrowCsrMatrix)
{
    auto factory = ilu_factory_type::build().on(ref);

    ASSERT_NO_THROW(factory->generate(gko::share(csr_mtx)));
}


TEST_F(ParIlu, NoThrowDenseMatrix)
{
    auto factory = ilu_factory_type::build().on(ref);

    ASSERT_NO_THROW(factory->generate(gko::share(dense_mtx)));
}


TEST_F(ParIlu, SetIterationsForDenseMatrix)
{
    auto factory = ilu_factory_type::build().with_iterations(5u).on(ref);

    ASSERT_NO_THROW(factory->generate(gko::share(dense_mtx)));
}


TEST_F(ParIlu, LUFactorFunctionsSetProperly)
{
    auto factory = ilu_factory_type::build().on(ref);

    auto factors = factory->generate(gko::share(dense_mtx));
    auto lin_op_l_factor =
        static_cast<const gko::LinOp *>(factors->get_l_factor());
    auto lin_op_u_factor =
        static_cast<const gko::LinOp *>(factors->get_u_factor());
    auto first_operator = factors->get_operators()[0].get();
    auto second_operator = factors->get_operators()[1].get();

    ASSERT_EQ(lin_op_l_factor, first_operator);
    ASSERT_EQ(lin_op_u_factor, second_operator);
}


}  // namespace
