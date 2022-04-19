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

#include <limits>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/factorization/ilu.hpp>


#include "core/factorization/factorization_kernels.hpp"
#include "core/factorization/ilu_kernels.hpp"
#include "core/preconditioner/batch_ilu_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"
#include "ginkgo/core/preconditioner/batch_ilu.hpp"


namespace {


template <typename T>
class BatchIlu : public ::testing::Test {
protected:
    using value_type = T;
    using real_type = gko::remove_complex<value_type>;
    using Mtx = gko::matrix::BatchCsr<value_type>;
    using BDense = gko::matrix::BatchDense<value_type>;
    using RBDense = gko::matrix::BatchDense<real_type>;
    using prec_type = gko::preconditioner::BatchIlu<value_type>;

    BatchIlu() : exec(gko::ReferenceExecutor::create()), mtx(get_matrix()) {}

    std::shared_ptr<const gko::ReferenceExecutor> exec;

    const size_t nbatch = 2;
    const int nrows = 3;
    std::shared_ptr<const Mtx> mtx;

    std::unique_ptr<Mtx> get_matrix()
    {
        auto mat = Mtx::create(exec, nbatch, gko::dim<2>(nrows, nrows), 6);
        int* const row_ptrs = mat->get_row_ptrs();
        int* const col_idxs = mat->get_col_idxs();
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

TYPED_TEST_SUITE(BatchIlu, gko::test::ValueTypes);


TYPED_TEST(BatchIlu, GenerationIsEquivalentToUnbatched)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using mtx_type = typename TestFixture::Mtx;
    using unbatch_type = typename mtx_type::unbatch_type;
    auto exec = this->exec;
    const auto sys_csr = this->mtx.get();
    const auto nbatch = this->nbatch;
    const auto nrows = this->nrows;
    const auto nnz = sys_csr->get_num_stored_elements() / nbatch;
    // extract the first matrix, as a view, into a regular Csr matrix.
    const auto unbatch_size = gko::dim<2>{nrows, sys_csr->get_size().at(0)[1]};
    auto sys_rows_view = gko::Array<index_type>::const_view(
        exec, nrows + 1, sys_csr->get_const_row_ptrs());
    auto sys_cols_view = gko::Array<index_type>::const_view(
        exec, nnz, sys_csr->get_const_col_idxs());
    auto sys_vals_view = gko::Array<value_type>::const_view(
        exec, nnz, sys_csr->get_const_values());
    auto first_csr = unbatch_type::create_const(
        exec, unbatch_size, std::move(sys_vals_view), std::move(sys_cols_view),
        std::move(sys_rows_view));
    // initialize L and U factors
    gko::Array<index_type> l_row_ptrs(exec, nrows + 1);
    gko::Array<index_type> u_row_ptrs(exec, nrows + 1);
    gko::kernels::reference::factorization::initialize_row_ptrs_l_u(
        exec, first_csr.get(), l_row_ptrs->get_data(), u_row_ptrs->get_data());
    const auto l_nnz = exec->copy_val_to_host(&l_row_ptrs[nrows]);
    const auto u_nnz = exec->copy_val_to_host(&u_row_ptrs[nrows]);
    auto first_L = unbatch_type::create(exec, unbatch_size, l_nnz);
    auto first_U = unbatch_type::create(exec, unbatch_size, u_nnz);
    exec->copy(nrows + 1, l_row_ptrs.get_const_data(), first_L->get_row_ptrs());
    exec->copy(nrows + 1, u_row_ptrs.get_const_data(), first_U->get_row_ptrs());
    gko::kernels::reference::factorization::initialize_l_u(
        exec, first_csr.get(), first_L.get(), first_U.get());
    auto l_factor = mtx_type::create(exec, nbatch, unbatch_size, l_nnz);
    auto u_factor = mtx_type::create(exec, nbatch, unbatch_size, u_nnz);
    exec->copy(nrows + 1, first_L->get_const_row_ptrs(),
               l_factor->get_row_ptrs());
    exec->copy(nrows + 1, first_U->get_const_row_ptrs(),
               u_factor->get_row_ptrs());
    exec->copy(nnz, first_L->get_const_col_idxs(), l_factor->get_col_idxs());
    exec->copy(nnz, first_U->get_const_col_idxs(), u_factor->get_col_idxs());
    // unbatch for check
    auto mtxs = sys_csr->unbatch();
    using unbatch_ilu_type = gko::factorization::Ilu<value_type>;
    auto ilu_fact = unbatch_ilu_type::build().with_skip_sorting(true).on(exec);
    std::vector<std::shared_ptr<unbatch_type>> check_l_factors(mtxs.size());
    std::vector<std::shared_ptr<unbatch_type>> check_u_factors(mtxs.size());
    for (size_t i = 0; i < mtxs.size(); i++) {
        auto facts = ilu_fact->generate(mtxs[i].get());
        check_l_factors[i] = facts->get_operators()[0];
        check_u_factors[i] = facts->get_operators()[1];
    }

    gko::kernels::reference::batch_ilu::generate_split(
        exec, gko::preconditioner::batch_factorization_type::exact, sys_csr,
        l_factor.get(), u_factor.get());

    auto l_factors = l_factor->unbatch();
    auto u_factors = u_factor->unbatch();
    for (size_t i = 0; i < mtxs.size(); i++) {
        GKO_ASSERT_MTX_NEAR(l_factors[i], check_l_factors[i], 0.0);
        GKO_ASSERT_MTX_NEAR(u_factors[i], check_u_factors[i], 0.0);
    }
}


TYPED_TEST(BatchIlu, ExactTrsvAppliesToSingleVector)
{
    using value_type = typename TestFixture::value_type;
    using BDense = typename TestFixture::BDense;
    using prec_type = typename TestFixture::prec_type;
    auto exec = this->exec;
    auto xex = gko::batch_initialize<BDense>(
        {{-1.0, -3.0, 20.0}, {2.0, 1.25, -12.0}}, this->exec);
    auto b = gko::batch_initialize<BDense>({{-2.0, 9.0, 4.0}, {-3.0, 5.0, 3.0}},
                                           this->exec);
    auto x = BDense::create(this->exec,
                            gko::batch_dim<>(2, gko::dim<2>(this->nrows, 1)));
    auto batch_ilu_fact = prec_type::build().on(exec);
    auto prec = batch_ilu_fact->generate(this->mtx.get());
    auto l_factor = prec->get_const_lower_factor();
    auto u_factor = prec->get_const_upper_factor();

    gko::kernels::reference::batch_ilu::apply_split(
        this->exec, l_factor.get(), u_factor.get(), b.get(), x.get());

    const auto eps = r<value_type>::value;
    GKO_ASSERT_BATCH_MTX_NEAR(x, xex, eps);
}


}  // namespace
