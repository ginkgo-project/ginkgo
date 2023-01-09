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

#include "core/preconditioner/batch_isai_kernels.hpp"


#include <limits>


#include <gtest/gtest.h>


#include <ginkgo/core/factorization/ilu.hpp>
#include <ginkgo/core/preconditioner/batch_isai.hpp>
#include <ginkgo/core/preconditioner/ilu.hpp>
#include <ginkgo/core/solver/upper_trs.hpp>


#include "core/factorization/factorization_kernels.hpp"
#include "core/factorization/ilu_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"
#include "test/utils/executor.hpp"

namespace {


class BatchIsai : public CommonTestFixture {
protected:
    using value_type = double;
    using index_type = int;
    using real_type = gko::remove_complex<value_type>;
    using Mtx = gko::matrix::BatchCsr<value_type>;
    using ubatched_mat_type = Mtx::unbatch_type;
    using BDense = gko::matrix::BatchDense<value_type>;
    using RBDense = gko::matrix::BatchDense<real_type>;
    using prec_type = gko::preconditioner::BatchIsai<value_type>;

    BatchIsai()
        : general_mtx(
              gko::share(gko::test::generate_uniform_batch_random_matrix<Mtx>(
                  nbatch, nrows, nrows,
                  std::uniform_int_distribution<>(min_nnz_row, nrows),
                  std::normal_distribution<real_type>(0.0, 1.0), rand_engine,
                  true, ref))),
          lower_mtx(get_lower_matrix()),
          upper_mtx(get_upper_matrix())
    {}

    std::ranlux48 rand_engine;

    const size_t nbatch = 9;
    const index_type nrows = 29;
    const int min_nnz_row = 5;
    std::shared_ptr<const Mtx> general_mtx;
    std::shared_ptr<const Mtx> lower_mtx;
    std::shared_ptr<const Mtx> upper_mtx;

    std::unique_ptr<Mtx> get_lower_matrix()
    {
        auto unbatch_mat =
            gko::test::generate_random_triangular_matrix<ubatched_mat_type>(
                nrows, false, true,
                std::uniform_int_distribution<>(min_nnz_row, nrows),
                std::normal_distribution<real_type>(0.0, 1.0), rand_engine,
                ref);

        return Mtx::create(ref, nbatch, unbatch_mat.get());
    }

    std::unique_ptr<Mtx> get_upper_matrix()
    {
        auto unbatch_mat =
            gko::test::generate_random_triangular_matrix<ubatched_mat_type>(
                nrows, false, false,
                std::uniform_int_distribution<>(min_nnz_row, nrows),
                std::normal_distribution<real_type>(0.0, 1.0), rand_engine,
                ref);

        return Mtx::create(ref, nbatch, unbatch_mat.get());
    }

    // TODO: Add tests for non-sorted input matrix
    void test_generate_eqvt_to_ref(
        gko::preconditioner::batch_isai_input_matrix_type isai_type,
        const int spy_power,
        std::shared_ptr<const gko::matrix::BatchCsr<value_type>> mtx)
    {
        auto d_mtx = gko::share(gko::clone(exec, mtx.get()));
        auto prec_fact = prec_type::build()
                             .with_skip_sorting(true)
                             .with_isai_input_matrix_type(isai_type)
                             .with_sparsity_power(spy_power)
                             .on(ref);
        auto d_prec_fact = prec_type::build()
                               .with_skip_sorting(true)
                               .with_isai_input_matrix_type(isai_type)
                               .with_sparsity_power(spy_power)
                               .on(exec);

        auto prec = prec_fact->generate(mtx);
        auto d_prec = d_prec_fact->generate(d_mtx);

        const auto approx_inv = prec->get_const_approximate_inverse().get();
        const auto d_approx_inv = d_prec->get_const_approximate_inverse().get();
        const auto tol = 500 * r<value_type>::value;
        GKO_ASSERT_BATCH_MTX_NEAR(approx_inv, d_approx_inv, tol);
    }

    // TODO: Add tests for non-sorted input matrix
    void test_apply_eqvt_to_ref(
        gko::preconditioner::batch_isai_input_matrix_type isai_type,
        const int spy_power,
        std::shared_ptr<const gko::matrix::BatchCsr<value_type>> mtx)
    {
        using BDense = gko::matrix::BatchDense<value_type>;

        auto prec_fact = prec_type::build()
                             .with_skip_sorting(true)
                             .with_isai_input_matrix_type(isai_type)
                             .with_sparsity_power(spy_power)
                             .on(ref);
        auto prec = prec_fact->generate(mtx);
        const auto approx_inv = prec->get_const_approximate_inverse().get();


        auto d_mtx = gko::share(gko::clone(exec, mtx.get()));
        auto d_prec_fact = prec_type::build()
                               .with_skip_sorting(true)
                               .with_isai_input_matrix_type(isai_type)
                               .with_sparsity_power(spy_power)
                               .on(exec);
        auto d_prec = d_prec_fact->generate(d_mtx);
        const auto d_approx_inv = d_prec->get_const_approximate_inverse().get();

        auto rv = gko::test::generate_uniform_batch_random_matrix<BDense>(
            nbatch, nrows, 1, std::uniform_int_distribution<>(1, 1),
            std::normal_distribution<real_type>(0.0, 1.0), rand_engine, false,
            ref);
        auto z = BDense::create(ref, rv->get_size());
        auto d_rv = gko::as<BDense>(gko::clone(exec, rv));
        auto d_z = BDense::create(exec, rv->get_size());

        gko::kernels::reference::batch_isai::apply_isai(
            ref, mtx.get(), approx_inv, rv.get(), z.get());

        gko::kernels::EXEC_NAMESPACE::batch_isai::apply_isai(
            exec, d_mtx.get(), d_approx_inv, d_rv.get(), d_z.get());

        const auto tol = 500 * r<value_type>::value;
        GKO_ASSERT_BATCH_MTX_NEAR(z, d_z, tol);
    }
};


TEST_F(BatchIsai, GeneralIsaiGenerateIsEquivalentToReferenceSpy1)
{
    this->test_generate_eqvt_to_ref(
        gko::preconditioner::batch_isai_input_matrix_type::general, 1,
        this->general_mtx);
}


TEST_F(BatchIsai, GeneralIsaiGenerateIsEquivalentToReferenceSpy2)
{
    this->test_generate_eqvt_to_ref(
        gko::preconditioner::batch_isai_input_matrix_type::general, 2,
        this->general_mtx);
}


TEST_F(BatchIsai, LowerIsaiGenerateIsEquivalentToReferenceSpy1)
{
    this->test_generate_eqvt_to_ref(
        gko::preconditioner::batch_isai_input_matrix_type::lower_tri, 1,
        this->lower_mtx);
}


TEST_F(BatchIsai, LowerIsaiGenerateIsEquivalentToReferenceSpy2)
{
    this->test_generate_eqvt_to_ref(
        gko::preconditioner::batch_isai_input_matrix_type::lower_tri, 2,
        this->lower_mtx);
}


TEST_F(BatchIsai, UpperIsaiGenerateIsEquivalentToReferenceSpy1)
{
    this->test_generate_eqvt_to_ref(
        gko::preconditioner::batch_isai_input_matrix_type::upper_tri, 1,
        this->upper_mtx);
}


TEST_F(BatchIsai, UpperIsaiGenerateIsEquivalentToReferenceSpy2)
{
    this->test_generate_eqvt_to_ref(
        gko::preconditioner::batch_isai_input_matrix_type::upper_tri, 2,
        this->upper_mtx);
}


TEST_F(BatchIsai, GeneralIsaiApplyIsEquivalentToReferenceSpy1)
{
    this->test_apply_eqvt_to_ref(
        gko::preconditioner::batch_isai_input_matrix_type::general, 1,
        this->general_mtx);
}


TEST_F(BatchIsai, GeneralIsaiApplyIsEquivalentToReferenceSpy2)
{
    this->test_apply_eqvt_to_ref(
        gko::preconditioner::batch_isai_input_matrix_type::general, 2,
        this->general_mtx);
}


TEST_F(BatchIsai, LowerIsaiApplyIsEquivalentToReferenceSpy1)
{
    this->test_apply_eqvt_to_ref(
        gko::preconditioner::batch_isai_input_matrix_type::lower_tri, 1,
        this->lower_mtx);
}


TEST_F(BatchIsai, LowerIsaiApplyIsEquivalentToReferenceSpy2)
{
    this->test_apply_eqvt_to_ref(
        gko::preconditioner::batch_isai_input_matrix_type::lower_tri, 2,
        this->lower_mtx);
}


TEST_F(BatchIsai, UpperIsaiApplyIsEquivalentToReferenceSpy1)
{
    this->test_apply_eqvt_to_ref(
        gko::preconditioner::batch_isai_input_matrix_type::upper_tri, 1,
        this->upper_mtx);
}


TEST_F(BatchIsai, UpperIsaiApplyIsEquivalentToReferenceSpy2)
{
    this->test_apply_eqvt_to_ref(
        gko::preconditioner::batch_isai_input_matrix_type::upper_tri, 2,
        this->upper_mtx);
}

}  // namespace
