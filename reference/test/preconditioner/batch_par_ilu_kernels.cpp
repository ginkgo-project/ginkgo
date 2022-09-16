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

#include <ginkgo/core/preconditioner/batch_par_ilu.hpp>


#include <limits>


#include <gtest/gtest.h>


#include <ginkgo/core/factorization/ilu.hpp>
#include <ginkgo/core/preconditioner/ilu.hpp>
#include <ginkgo/core/solver/upper_trs.hpp>


#include "core/factorization/factorization_kernels.hpp"
#include "core/factorization/ilu_kernels.hpp"
#include "core/preconditioner/batch_par_ilu_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"


namespace {


template <typename T>
class BatchParIlu : public ::testing::Test {
protected:
    using value_type = T;
    using index_type = int;
    using real_type = gko::remove_complex<value_type>;
    using Mtx = gko::matrix::BatchCsr<value_type>;
    using BDense = gko::matrix::BatchDense<value_type>;
    using RBDense = gko::matrix::BatchDense<real_type>;
    using prec_type = gko::preconditioner::BatchParIlu<value_type>;

    BatchParIlu() : exec(gko::ReferenceExecutor::create()), mtx(get_matrix()) {}

    std::shared_ptr<const gko::ReferenceExecutor> exec;

    const size_t nbatch = 2;
    const index_type nrows = 3;
    std::shared_ptr<const Mtx> mtx;

    std::unique_ptr<Mtx> get_matrix()
    {
        auto mat = Mtx::create(exec, nbatch, gko::dim<2>(nrows, nrows), 6);
        index_type* const row_ptrs = mat->get_row_ptrs();
        index_type* const col_idxs = mat->get_col_idxs();
        value_type* const vals = mat->get_values();
        // clang-format off
		row_ptrs[0] = 0; row_ptrs[1] = 2; row_ptrs[2] = 4; row_ptrs[3] = 6;
		col_idxs[0] = 0; col_idxs[1] = 1; col_idxs[2] = 0; col_idxs[3] = 1;
		col_idxs[4] = 0; col_idxs[5] = 2;
		vals[0] = 2.0; vals[1] = 0.25; vals[2] = -1.0; vals[3] = -3.0;
		vals[4] = 2.0; vals[5] = 0.2;
		vals[6] = -1.5; vals[7] = 0.55; vals[8] = -1.0; vals[9] = 4.0;
		vals[10] = 2.0; vals[11] = -0.25;
        // clang-format on
        return mat;
    }
};

TYPED_TEST_SUITE(BatchParIlu, gko::test::ValueTypes);


TYPED_TEST(BatchParIlu, GenerationFromIsEquivalentToUnbatched)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using mtx_type = typename TestFixture::Mtx;
    using unbatch_type = typename mtx_type::unbatch_type;
    using unbatch_ilu_type = gko::factorization::ParIlu<value_type>;
    using prec_type = typename TestFixture::prec_type;
    auto exec = this->exec;
    const auto sys_csr = this->mtx.get();
    const auto nbatch = this->nbatch;
    const auto nrows = this->nrows;
    const auto nnz = sys_csr->get_num_stored_elements() / nbatch;
    // unbatch for check
    auto mtxs = gko::test::share(sys_csr->unbatch());
    auto ilu_fact = unbatch_ilu_type::build()
                        .with_skip_sorting(true)
                        .with_iterations(static_cast<gko::size_type>(20))
                        .on(exec);
    std::vector<std::shared_ptr<const unbatch_type>> check_l_factors(
        mtxs.size());
    std::vector<std::shared_ptr<const unbatch_type>> check_u_factors(
        mtxs.size());
    for (size_t i = 0; i < mtxs.size(); i++) {
        auto facts = ilu_fact->generate(mtxs[i]);
        check_l_factors[i] = facts->get_l_factor();
        check_u_factors[i] = facts->get_u_factor();
    }

    auto prec_fact =
        prec_type::build().with_skip_sorting(true).with_num_sweeps(20).on(exec);
    auto prec = prec_fact->generate(this->mtx);

    auto l_factors = prec->get_const_l_factor()->unbatch();
    auto u_factors = prec->get_const_u_factor()->unbatch();
    for (size_t i = 0; i < mtxs.size(); i++) {
        GKO_ASSERT_MTX_NEAR(l_factors[i], check_l_factors[i], 0.0);
        GKO_ASSERT_MTX_NEAR(u_factors[i], check_u_factors[i], 0.0);
    }
}


TYPED_TEST(BatchParIlu, ApplyToSingleVectorIsEquivalentToUnbatched)
{
    using value_type = typename TestFixture::value_type;
    using BDense = typename TestFixture::BDense;
    using prec_type = typename TestFixture::prec_type;
    using mtx_type = typename TestFixture::Mtx;
    using unbatch_type = typename mtx_type::unbatch_type;
    using unbatch_prec_type =
        gko::preconditioner::Ilu<gko::solver::LowerTrs<value_type>,
                                 gko::solver::UpperTrs<value_type>>;

    using unbatch_fact_type = gko::factorization::ParIlu<value_type>;

    auto exec = this->exec;
    auto b = gko::batch_initialize<BDense>({{-2.0, 9.0, 4.0}, {-3.0, 5.0, 3.0}},
                                           exec);
    auto x =
        BDense::create(exec, gko::batch_dim<>(2, gko::dim<2>(this->nrows, 1)));
    auto umtxs = gko::test::share(this->mtx->unbatch());
    auto ub = b->unbatch();
    auto ux = x->unbatch();

    auto unbatch_factorization_fact =
        gko::share(unbatch_fact_type::build()
                       .with_iterations(static_cast<gko::size_type>(20))
                       .with_skip_sorting(true)
                       .on(exec));
    auto unbatch_prec_fact =
        unbatch_prec_type::build()
            .with_factorization_factory(unbatch_factorization_fact)
            .on(exec);

    for (size_t i = 0; i < umtxs.size(); i++) {
        auto unbatch_prec = unbatch_prec_fact->generate(umtxs[i]);
        unbatch_prec->apply(ub[i].get(), ux[i].get());
    }

    auto prec_fact =
        prec_type::build().with_skip_sorting(true).with_num_sweeps(20).on(exec);
    auto prec = prec_fact->generate(this->mtx);

    gko::kernels::reference::batch_par_ilu::apply_par_ilu0(
        exec, prec->get_const_l_factor(), prec->get_const_u_factor(), b.get(),
        x.get());

    auto xs = x->unbatch();
    for (size_t i = 0; i < umtxs.size(); i++) {
        GKO_ASSERT_MTX_NEAR(ux[i], xs[i], r<value_type>::value);
    }
}


}  // namespace
