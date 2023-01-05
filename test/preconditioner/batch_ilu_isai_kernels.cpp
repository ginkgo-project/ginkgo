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

#include "core/preconditioner/batch_ilu_isai_kernels.hpp"


#include <limits>


#include <gtest/gtest.h>


#include <ginkgo/core/factorization/ilu.hpp>
#include <ginkgo/core/preconditioner/batch_ilu_isai.hpp>
#include <ginkgo/core/preconditioner/ilu.hpp>
#include <ginkgo/core/solver/upper_trs.hpp>


#include "core/factorization/factorization_kernels.hpp"
#include "core/factorization/ilu_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"
#include "test/utils/executor.hpp"

namespace {


class BatchIluIsai : public CommonTestFixture {
protected:
    using value_type = double;
    using index_type = int;
    using real_type = gko::remove_complex<value_type>;
    using Mtx = gko::matrix::BatchCsr<value_type>;
    using BDense = gko::matrix::BatchDense<value_type>;
    using RBDense = gko::matrix::BatchDense<real_type>;
    using prec_type = gko::preconditioner::BatchIluIsai<value_type>;

    BatchIluIsai()
        : mtx_small(
              gko::share(gko::test::generate_uniform_batch_random_matrix<Mtx>(
                  nbatch, nrows_small, nrows_small,
                  std::uniform_int_distribution<>(min_nnz_row_small,
                                                  nrows_small),
                  std::normal_distribution<real_type>(0.0, 1.0), rand_engine,
                  true, ref))),
          mtx_big(
              gko::share(gko::test::generate_uniform_batch_random_matrix<Mtx>(
                  nbatch, nrows_big, nrows_big,
                  std::uniform_int_distribution<>(min_nnz_row_big, nrows_big),
                  std::normal_distribution<real_type>(0.0, 1.0), rand_engine,
                  true, ref)))
    {}

    std::ranlux48 rand_engine;

    const size_t nbatch = 9;
    const index_type nrows_small = 15;
    const int min_nnz_row_small = 3;
    std::shared_ptr<const Mtx> mtx_small;
    const index_type nrows_big = 50;
    const int min_nnz_row_big = 33;
    std::shared_ptr<const Mtx> mtx_big;

    // TODO: Add tests for non-sorted input matrix
    void test_generate_eqvt_to_ref(
        const gko::preconditioner::batch_ilu_type ilu_type,
        const int num_sweeps = 30, const int lower_spy_power = 2,
        const int upper_spy_power = 3, bool test_isai_extension = false)
    {
        auto mtx = test_isai_extension == true ? mtx_big : mtx_small;
        auto d_mtx = gko::share(gko::clone(exec, mtx.get()));
        auto prec_fact =
            prec_type::build()
                .with_skip_sorting(true)
                .with_ilu_type(ilu_type)
                .with_parilu_num_sweeps(num_sweeps)
                .with_lower_factor_isai_sparsity_power(lower_spy_power)
                .with_upper_factor_isai_sparsity_power(upper_spy_power)
                .on(ref);
        auto d_prec_fact =
            prec_type::build()
                .with_skip_sorting(true)
                .with_ilu_type(ilu_type)
                .with_parilu_num_sweeps(num_sweeps)
                .with_lower_factor_isai_sparsity_power(lower_spy_power)
                .with_upper_factor_isai_sparsity_power(upper_spy_power)
                .on(exec);

        auto prec = prec_fact->generate(mtx);
        auto d_prec = d_prec_fact->generate(d_mtx);

        const auto lower_factor_isai = prec->get_const_lower_factor_isai();
        const auto d_lower_factor_isai = d_prec->get_const_lower_factor_isai();
        const auto upper_factor_isai = prec->get_const_upper_factor_isai();
        const auto d_upper_factor_isai = d_prec->get_const_upper_factor_isai();
        const auto tol = test_isai_extension == true
                             ? 1e+6 * r<value_type>::value
                             : 50 * r<value_type>::value;
        GKO_ASSERT_BATCH_MTX_NEAR(lower_factor_isai, d_lower_factor_isai, tol);
        GKO_ASSERT_BATCH_MTX_NEAR(upper_factor_isai, d_upper_factor_isai, tol);
    }

    // TODO: Add tests for non-sorted input matrix
    void test_apply_eqvt_to_ref(
        const gko::preconditioner::batch_ilu_isai_apply apply_type,
        const int num_relaxation_steps,
        const gko::preconditioner::batch_ilu_type ilu_type,
        const int num_sweeps = 30, const int lower_spy_power = 2,
        const int upper_spy_power = 3, bool test_isai_extension = false)
    {
        using BDense = gko::matrix::BatchDense<value_type>;
        auto mtx = test_isai_extension == true ? mtx_big : mtx_small;
        auto prec_fact =
            prec_type::build()
                .with_skip_sorting(true)
                .with_ilu_type(ilu_type)
                .with_parilu_num_sweeps(num_sweeps)
                .with_lower_factor_isai_sparsity_power(lower_spy_power)
                .with_upper_factor_isai_sparsity_power(upper_spy_power)
                .with_apply_type(apply_type)
                .with_num_relaxation_steps(num_relaxation_steps)
                .on(ref);
        auto prec = prec_fact->generate(mtx);

        const auto l = prec->get_const_lower_factor().get();
        const auto u = prec->get_const_upper_factor().get();
        const auto l_isai = prec->get_const_lower_factor_isai().get();
        const auto u_isai = prec->get_const_upper_factor_isai().get();
        const auto mult_inv = prec->get_const_mult_inv().get();
        const auto iter_mat_lower_solve =
            prec->get_const_iteration_matrix_lower_solve().get();
        const auto iter_mat_upper_solve =
            prec->get_const_iteration_matrix_upper_solve().get();


        auto d_mtx = gko::share(gko::clone(exec, mtx.get()));
        auto d_prec_fact =
            prec_type::build()
                .with_skip_sorting(true)
                .with_ilu_type(ilu_type)
                .with_parilu_num_sweeps(num_sweeps)
                .with_lower_factor_isai_sparsity_power(lower_spy_power)
                .with_upper_factor_isai_sparsity_power(upper_spy_power)
                .with_apply_type(apply_type)
                .with_num_relaxation_steps(num_relaxation_steps)
                .on(exec);
        auto d_prec = d_prec_fact->generate(d_mtx);

        const auto d_l = d_prec->get_const_lower_factor().get();
        const auto d_u = d_prec->get_const_upper_factor().get();
        const auto d_l_isai = d_prec->get_const_lower_factor_isai().get();
        const auto d_u_isai = d_prec->get_const_upper_factor_isai().get();
        const auto d_mult_inv = d_prec->get_const_mult_inv().get();
        const auto d_iter_mat_lower_solve =
            d_prec->get_const_iteration_matrix_lower_solve().get();
        const auto d_iter_mat_upper_solve =
            d_prec->get_const_iteration_matrix_upper_solve().get();

        auto nrows = test_isai_extension == true ? nrows_big : nrows_small;
        auto rv = gko::test::generate_uniform_batch_random_matrix<BDense>(
            nbatch, nrows, 1, std::uniform_int_distribution<>(1, 1),
            std::normal_distribution<real_type>(0.0, 1.0), rand_engine, false,
            ref);
        auto z = BDense::create(ref, rv->get_size());
        auto d_rv = gko::as<BDense>(gko::clone(exec, rv));
        auto d_z = BDense::create(exec, rv->get_size());

        gko::kernels::reference::batch_ilu_isai::apply_ilu_isai(
            ref, mtx.get(), l, u, l_isai, u_isai, mult_inv,
            iter_mat_lower_solve, iter_mat_upper_solve, prec->get_apply_type(),
            prec->get_num_relaxation_steps(), rv.get(), z.get());
        gko::kernels::EXEC_NAMESPACE::batch_ilu_isai::apply_ilu_isai(
            exec, d_mtx.get(), d_l, d_u, d_l_isai, d_u_isai, d_mult_inv,
            d_iter_mat_lower_solve, d_iter_mat_upper_solve,
            d_prec->get_apply_type(), prec->get_num_relaxation_steps(),
            d_rv.get(), d_z.get());

        const auto tol = test_isai_extension == true
                             ? 1e+7 * r<value_type>::value
                             : 50 * r<value_type>::value;
        GKO_ASSERT_BATCH_MTX_NEAR(z, d_z, tol);
    }
};


TEST_F(BatchIluIsai, IluIsaiGenerateIsEquivalentToReference)
{
    test_generate_eqvt_to_ref(gko::preconditioner::batch_ilu_type::exact_ilu,
                              30, 1, 3);
    test_generate_eqvt_to_ref(gko::preconditioner::batch_ilu_type::exact_ilu,
                              30, 2, 2);
    test_generate_eqvt_to_ref(gko::preconditioner::batch_ilu_type::parilu, 30,
                              1, 3);
    test_generate_eqvt_to_ref(gko::preconditioner::batch_ilu_type::parilu, 30,
                              2, 2);
}


TEST_F(BatchIluIsai, IluIsaiSpmvSimpleApplyIsEquivalentToReference)
{
    const auto apply_type =
        gko::preconditioner::batch_ilu_isai_apply::spmv_isai_simple;
    test_apply_eqvt_to_ref(apply_type, 2,
                           gko::preconditioner::batch_ilu_type::exact_ilu, 30,
                           1, 3);
    test_apply_eqvt_to_ref(apply_type, 2,
                           gko::preconditioner::batch_ilu_type::exact_ilu, 30,
                           2, 2);
    test_apply_eqvt_to_ref(
        apply_type, 2, gko::preconditioner::batch_ilu_type::parilu, 30, 1, 3);
    test_apply_eqvt_to_ref(
        apply_type, 2, gko::preconditioner::batch_ilu_type::parilu, 30, 2, 2);
}


// TODO: Implement batch_csr spgemm
// TEST_F(BatchIluIsai, IluIsaiSpmvWithSpgemmApplyIsEquivalentToReference)
// {
//     const auto apply_type =
//         gko::preconditioner::batch_ilu_isai_apply::spmv_isai_with_spgemm;
//     test_apply_eqvt_to_ref(apply_type, 2,
//                            gko::preconditioner::batch_ilu_type::exact_ilu,
//                            30, 1, 3);
//     test_apply_eqvt_to_ref(apply_type, 2,
//                            gko::preconditioner::batch_ilu_type::exact_ilu,
//                            30, 2, 2);
//     test_apply_eqvt_to_ref(
//         apply_type, 2, gko::preconditioner::batch_ilu_type::parilu, 30, 1,
//         3);
//     test_apply_eqvt_to_ref(
//         apply_type, 2, gko::preconditioner::batch_ilu_type::parilu, 30, 2,
//         2);
// }


TEST_F(BatchIluIsai, IluIsaiRelaxtionStepsSimpleApplyIsEquivalentToReference)
{
    const auto apply_type =
        gko::preconditioner::batch_ilu_isai_apply::relaxation_steps_isai_simple;
    test_apply_eqvt_to_ref(apply_type, 2,
                           gko::preconditioner::batch_ilu_type::exact_ilu, 30,
                           1, 3);
    test_apply_eqvt_to_ref(apply_type, 2,
                           gko::preconditioner::batch_ilu_type::exact_ilu, 30,
                           2, 2);
    test_apply_eqvt_to_ref(
        apply_type, 2, gko::preconditioner::batch_ilu_type::parilu, 30, 1, 3);
    test_apply_eqvt_to_ref(
        apply_type, 2, gko::preconditioner::batch_ilu_type::parilu, 30, 2, 2);
}


// TODO: Implement batch_csr spgemm
// TEST_F(BatchIluIsai,
//        IluIsaiRelaxtionStepsWithSpgemmApplyIsEquivalentToReference)
// {
//     const auto apply_type = gko::preconditioner::batch_ilu_isai_apply::
//         relaxation_steps_isai_with_spgemm;
//     test_apply_eqvt_to_ref(apply_type, 2,
//                            gko::preconditioner::batch_ilu_type::exact_ilu,
//                            30, 1, 3);
//     test_apply_eqvt_to_ref(apply_type, 2,
//                            gko::preconditioner::batch_ilu_type::exact_ilu,
//                            30, 2, 2);
//     test_apply_eqvt_to_ref(
//         apply_type, 2, gko::preconditioner::batch_ilu_type::parilu, 30, 1,
//         3);
//     test_apply_eqvt_to_ref(
//         apply_type, 2, gko::preconditioner::batch_ilu_type::parilu, 30, 2,
//         2);
// }


TEST_F(BatchIluIsai, IluExtendedIsaiGenerateIsEquivalentToReference)
{
    test_generate_eqvt_to_ref(gko::preconditioner::batch_ilu_type::exact_ilu,
                              30, 1, 3, true);
    test_generate_eqvt_to_ref(gko::preconditioner::batch_ilu_type::exact_ilu,
                              30, 2, 2, true);
    test_generate_eqvt_to_ref(gko::preconditioner::batch_ilu_type::parilu, 70,
                              1, 3, true);
    test_generate_eqvt_to_ref(gko::preconditioner::batch_ilu_type::parilu, 70,
                              2, 2, true);
}


TEST_F(BatchIluIsai, IluExtendedIsaiApplyIsEquivalentToReference)
{
    const auto apply_type =
        gko::preconditioner::batch_ilu_isai_apply::spmv_isai_simple;
    test_apply_eqvt_to_ref(apply_type, 2,
                           gko::preconditioner::batch_ilu_type::exact_ilu, 30,
                           1, 3, true);
    test_apply_eqvt_to_ref(apply_type, 2,
                           gko::preconditioner::batch_ilu_type::exact_ilu, 30,
                           2, 2, true);
    test_apply_eqvt_to_ref(apply_type, 2,
                           gko::preconditioner::batch_ilu_type::parilu, 70, 1,
                           3, true);
    test_apply_eqvt_to_ref(apply_type, 2,
                           gko::preconditioner::batch_ilu_type::parilu, 70, 2,
                           2, true);
}


}  // namespace
