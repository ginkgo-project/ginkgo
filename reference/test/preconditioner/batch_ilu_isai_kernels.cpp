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

#include <ginkgo/core/preconditioner/batch_ilu_isai.hpp>


#include <limits>


#include <gtest/gtest.h>


#include <ginkgo/core/factorization/ilu.hpp>
#include <ginkgo/core/preconditioner/ilu.hpp>
#include <ginkgo/core/preconditioner/isai.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/solver/upper_trs.hpp>


#include "core/factorization/factorization_kernels.hpp"
#include "core/factorization/ilu_kernels.hpp"
#include "core/preconditioner/batch_ilu_isai_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"


namespace {


template <typename T>
class BatchIluIsai : public ::testing::Test {
protected:
    using value_type = T;
    using index_type = int;
    using real_type = gko::remove_complex<value_type>;
    using Mtx = gko::matrix::BatchCsr<value_type, index_type>;
    using BDense = gko::matrix::BatchDense<value_type>;
    using RBDense = gko::matrix::BatchDense<real_type>;
    using prec_type = gko::preconditioner::BatchIluIsai<value_type, index_type>;
    using exact_ilu_factorization = gko::factorization::Ilu<value_type>;
    using parilu_factorization = gko::factorization::ParIlu<value_type>;
    using lower_isai = gko::preconditioner::LowerIsai<value_type, index_type>;
    using upper_isai = gko::preconditioner::UpperIsai<value_type, index_type>;
    using ir = gko::solver::Ir<value_type>;
    using ilu_prec_with_isai_relxation_steps = gko::preconditioner::Ilu<ir, ir>;
    using ilu_prec_with_isai_spmv =
        gko::preconditioner::Ilu<lower_isai, upper_isai>;

    BatchIluIsai() : exec(gko::ReferenceExecutor::create()), mtx(get_matrix())
    {}

    const size_t nbatch = 2;
    const index_type nrows = 4;
    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::shared_ptr<const Mtx> mtx;

    std::unique_ptr<Mtx> get_matrix()
    {
        auto mat = Mtx::create(exec, nbatch, gko::dim<2>(nrows, nrows), 8);
        index_type* const row_ptrs = mat->get_row_ptrs();
        index_type* const col_idxs = mat->get_col_idxs();
        value_type* const vals = mat->get_values();
        // clang-format off
		row_ptrs[0] = 0; row_ptrs[1] = 2; row_ptrs[2] = 4; row_ptrs[3] = 6; row_ptrs[4] = 8;
		col_idxs[0] = 0; col_idxs[1] = 2; col_idxs[2] = 0; col_idxs[3] = 1;
		col_idxs[4] = 0; col_idxs[5] = 2; col_idxs[6] = 1, col_idxs[7] = 3;
		vals[0] = 2.0; vals[1] = 0.25; vals[2] = -1.0; vals[3] = -3.0;
		vals[4] = 2.0; vals[5] = 0.2;
		vals[6] = -1.5; vals[7] = 0.55; vals[8] = -1.0; vals[9] = 4.0;
		vals[10] = 2.0; vals[11] = -0.25;
        vals[12] = -1.45; vals[13] = 0.45; vals[14] = -5.0; vals[15] = 8.0;

        // clang-format on
        return mat;
    }

    // TODO: Add tests for non-sorted input matrix
    void test_batch_ilu_isai_generation_is_eqvt_to_unbatched(
        const gko::preconditioner::batch_ilu_type ilu_type,
        const int parilu_num_sweeps, const int lower_spy_power,
        const int upper_spy_power)
    {
        using unbatch_type = typename Mtx::unbatch_type;

        auto umtxs = gko::test::share(mtx->unbatch());

        std::vector<std::shared_ptr<const unbatch_type>>
            check_lower_factor_unbatched(umtxs.size());
        std::vector<std::shared_ptr<const unbatch_type>>
            check_upper_factor_unbatched(umtxs.size());
        std::vector<std::shared_ptr<const unbatch_type>>
            check_lower_factor_isai_unbatched(umtxs.size());
        std::vector<std::shared_ptr<const unbatch_type>>
            check_upper_factor_isai_unbatched(umtxs.size());

        auto lower_isai_factory = gko::share(
            lower_isai::build().with_sparsity_power(lower_spy_power).on(exec));

        auto upper_isai_factory = gko::share(
            upper_isai::build().with_sparsity_power(upper_spy_power).on(exec));

        if (ilu_type == gko::preconditioner::batch_ilu_type::exact_ilu) {
            auto ilu_factorization_factory =
                exact_ilu_factorization::build().with_skip_sorting(true).on(
                    exec);

            for (size_t i = 0; i < umtxs.size(); i++) {
                auto ilu_factorization =
                    gko::share(ilu_factorization_factory->generate(umtxs[i]));

                check_lower_factor_unbatched[i] =
                    ilu_factorization->get_l_factor();
                check_upper_factor_unbatched[i] =
                    ilu_factorization->get_u_factor();
            }
        } else if (ilu_type == gko::preconditioner::batch_ilu_type::parilu) {
            auto ilu_factorization_factory =
                parilu_factorization::build()
                    .with_skip_sorting(true)
                    .with_iterations(
                        static_cast<gko::size_type>(parilu_num_sweeps))
                    .on(exec);

            for (size_t i = 0; i < umtxs.size(); i++) {
                auto ilu_factorization =
                    gko::share(ilu_factorization_factory->generate(umtxs[i]));

                check_lower_factor_unbatched[i] =
                    ilu_factorization->get_l_factor();
                check_upper_factor_unbatched[i] =
                    ilu_factorization->get_u_factor();
            }
        }

        for (size_t i = 0; i < umtxs.size(); i++) {
            auto lower_isai =
                lower_isai_factory->generate(check_lower_factor_unbatched[i]);
            auto upper_isai =
                upper_isai_factory->generate(check_upper_factor_unbatched[i]);
            check_lower_factor_isai_unbatched[i] =
                lower_isai->get_approximate_inverse();
            check_upper_factor_isai_unbatched[i] =
                upper_isai->get_approximate_inverse();
        }


        auto prec_fact =
            prec_type::build()
                .with_skip_sorting(true)
                .with_ilu_type(ilu_type)
                .with_parilu_num_sweeps(parilu_num_sweeps)
                .with_lower_factor_isai_sparsity_power(lower_spy_power)
                .with_upper_factor_isai_sparsity_power(upper_spy_power)
                .on(exec);

        auto prec = prec_fact->generate(mtx);

        const auto l_batch_vec = prec->get_const_lower_factor()->unbatch();
        const auto u_batch_vec = prec->get_const_upper_factor()->unbatch();
        const auto l_isai_batch_vec =
            prec->get_const_lower_factor_isai()->unbatch();
        const auto u_isai_batch_vec =
            prec->get_const_upper_factor_isai()->unbatch();

        for (size_t i = 0; i < umtxs.size(); i++) {
            GKO_ASSERT_MTX_NEAR(check_lower_factor_unbatched[i], l_batch_vec[i],
                                r<value_type>::value);
            GKO_ASSERT_MTX_NEAR(check_upper_factor_unbatched[i], u_batch_vec[i],
                                r<value_type>::value);
            GKO_ASSERT_MTX_NEAR(check_lower_factor_isai_unbatched[i],
                                l_isai_batch_vec[i], r<value_type>::value);
            GKO_ASSERT_MTX_NEAR(check_upper_factor_isai_unbatched[i],
                                u_isai_batch_vec[i], r<value_type>::value);
        }
    }

    // TODO: Add tests for non-sorted input matrix
    void test_batch_ilu_isai_apply_to_single_vector_is_eqvt_to_unbatched(
        const gko::preconditioner::batch_ilu_type ilu_type,
        const int parilu_num_sweeps, const int lower_spy_power,
        const int upper_spy_power, const int num_relaxation_steps,
        const gko::preconditioner::batch_ilu_isai_apply apply_type)
    {
        using unbatch_type = typename Mtx::unbatch_type;

        auto umtxs = gko::test::share(mtx->unbatch());

        auto b = gko::batch_initialize<BDense>(
            {{-2.0, 9.0, 4.0, 7.0}, {-3.0, 5.0, 3.0, 10.0}}, exec);
        auto x = BDense::create(
            exec, gko::batch_dim<>(2, gko::dim<2>(this->nrows, 1)));
        auto ub = b->unbatch();
        auto ux = x->unbatch();

        auto lower_isai_factory = gko::share(
            lower_isai::build().with_sparsity_power(lower_spy_power).on(exec));

        auto upper_isai_factory = gko::share(
            upper_isai::build().with_sparsity_power(upper_spy_power).on(exec));

        if (apply_type == gko::preconditioner::batch_ilu_isai_apply::
                              relaxation_steps_isai_simple ||
            apply_type == gko::preconditioner::batch_ilu_isai_apply::
                              relaxation_steps_isai_with_spgemm) {
            auto lower_trisolve_factory =
                ir::build()
                    .with_solver(lower_isai_factory)
                    .with_criteria(
                        gko::stop::Iteration::build()
                            .with_max_iters(static_cast<gko::size_type>(
                                num_relaxation_steps))
                            .on(exec))
                    .on(exec);

            auto upper_trisolve_factory =
                ir::build()
                    .with_solver(upper_isai_factory)
                    .with_criteria(
                        gko::stop::Iteration::build()
                            .with_max_iters(static_cast<gko::size_type>(
                                num_relaxation_steps))
                            .on(exec))
                    .on(exec);

            // Generate an ILU preconditioner factory by setting lower and upper
            // triangular solver - in this case the previously defined iterative
            // refinement method.
            auto ilu_prec_factory =
                ilu_prec_with_isai_relxation_steps::build()
                    .with_l_solver_factory(gko::clone(lower_trisolve_factory))
                    .with_u_solver_factory(gko::clone(upper_trisolve_factory))
                    .on(exec);

            if (ilu_type == gko::preconditioner::batch_ilu_type::exact_ilu) {
                auto ilu_factorization_factory =
                    exact_ilu_factorization::build().with_skip_sorting(true).on(
                        exec);

                for (size_t i = 0; i < umtxs.size(); i++) {
                    auto ilu_factorization = gko::share(
                        ilu_factorization_factory->generate(umtxs[i]));
                    auto ilu_prec = gko::share(
                        ilu_prec_factory->generate(ilu_factorization));
                    ilu_prec->apply(ub[i].get(), ux[i].get());
                }
            } else if (ilu_type ==
                       gko::preconditioner::batch_ilu_type::parilu) {
                auto ilu_factorization_factory =
                    parilu_factorization::build()
                        .with_skip_sorting(true)
                        .with_iterations(
                            static_cast<gko::size_type>(parilu_num_sweeps))
                        .on(exec);

                for (size_t i = 0; i < umtxs.size(); i++) {
                    auto ilu_factorization = gko::share(
                        ilu_factorization_factory->generate(umtxs[i]));
                    auto ilu_prec = gko::share(
                        ilu_prec_factory->generate(ilu_factorization));
                    ilu_prec->apply(ub[i].get(), ux[i].get());
                }
            }

        } else if (apply_type == gko::preconditioner::batch_ilu_isai_apply::
                                     spmv_isai_simple ||
                   apply_type == gko::preconditioner::batch_ilu_isai_apply::
                                     spmv_isai_with_spgemm) {
            auto lower_trisolve_factory = lower_isai_factory;
            auto upper_trisolve_factory = upper_isai_factory;

            // Generate an ILU preconditioner factory by setting lower and upper
            // triangular solver
            auto ilu_prec_factory =
                ilu_prec_with_isai_spmv::build()
                    .with_l_solver_factory(gko::clone(lower_trisolve_factory))
                    .with_u_solver_factory(gko::clone(upper_trisolve_factory))
                    .on(exec);

            if (ilu_type == gko::preconditioner::batch_ilu_type::exact_ilu) {
                using exact_ilu_factorization =
                    gko::factorization::Ilu<value_type>;
                auto ilu_factorization_factory =
                    exact_ilu_factorization::build().with_skip_sorting(true).on(
                        exec);

                for (size_t i = 0; i < umtxs.size(); i++) {
                    auto ilu_factorization = gko::share(
                        ilu_factorization_factory->generate(umtxs[i]));
                    auto ilu_prec = gko::share(
                        ilu_prec_factory->generate(ilu_factorization));
                    ilu_prec->apply(ub[i].get(), ux[i].get());
                }
            } else if (ilu_type ==
                       gko::preconditioner::batch_ilu_type::parilu) {
                using parilu_factorization =
                    gko::factorization::ParIlu<value_type>;
                auto ilu_factorization_factory =
                    parilu_factorization::build()
                        .with_skip_sorting(true)
                        .with_iterations(
                            static_cast<gko::size_type>(parilu_num_sweeps))
                        .on(exec);

                for (size_t i = 0; i < umtxs.size(); i++) {
                    auto ilu_factorization = gko::share(
                        ilu_factorization_factory->generate(umtxs[i]));
                    auto ilu_prec = gko::share(
                        ilu_prec_factory->generate(ilu_factorization));
                    ilu_prec->apply(ub[i].get(), ux[i].get());
                }
            }
        }

        auto prec_fact =
            prec_type::build()
                .with_skip_sorting(true)
                .with_ilu_type(ilu_type)
                .with_parilu_num_sweeps(parilu_num_sweeps)
                .with_lower_factor_isai_sparsity_power(lower_spy_power)
                .with_upper_factor_isai_sparsity_power(upper_spy_power)
                .with_apply_type(apply_type)
                .with_num_relaxation_steps(num_relaxation_steps)
                .on(exec);

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

        gko::kernels::reference::batch_ilu_isai::apply_ilu_isai(
            exec, mtx.get(), l, u, l_isai, u_isai, mult_inv,
            iter_mat_lower_solve, iter_mat_upper_solve, prec->get_apply_type(),
            prec->get_num_relaxation_steps(), b.get(), x.get());

        auto xs = x->unbatch();
        for (size_t i = 0; i < umtxs.size(); i++) {
            GKO_ASSERT_MTX_NEAR(ux[i], xs[i], r<value_type>::value);
        }
    }
};


TYPED_TEST_SUITE(BatchIluIsai, gko::test::ValueTypes);


TYPED_TEST(BatchIluIsai, BatchIluIsaiGenerationIsEquivalentToUnbatched)
{
    this->test_batch_ilu_isai_generation_is_eqvt_to_unbatched(
        gko::preconditioner::batch_ilu_type::exact_ilu, 10, 2, 3);

    this->test_batch_ilu_isai_generation_is_eqvt_to_unbatched(
        gko::preconditioner::batch_ilu_type::exact_ilu, 10, 1, 2);

    this->test_batch_ilu_isai_generation_is_eqvt_to_unbatched(
        gko::preconditioner::batch_ilu_type::parilu, 10, 2, 3);

    this->test_batch_ilu_isai_generation_is_eqvt_to_unbatched(
        gko::preconditioner::batch_ilu_type::parilu, 10, 1, 2);
}

TYPED_TEST(BatchIluIsai,
           BatchIluIsaiWithApplyTypeRelxationStepsSimpleIsEquivalentToUnbatched)
{
    this->test_batch_ilu_isai_apply_to_single_vector_is_eqvt_to_unbatched(
        gko::preconditioner::batch_ilu_type::exact_ilu, 10, 2, 3, 3,
        gko::preconditioner::batch_ilu_isai_apply::
            relaxation_steps_isai_simple);

    this->test_batch_ilu_isai_apply_to_single_vector_is_eqvt_to_unbatched(
        gko::preconditioner::batch_ilu_type::exact_ilu, 10, 1, 2, 3,
        gko::preconditioner::batch_ilu_isai_apply::
            relaxation_steps_isai_simple);

    this->test_batch_ilu_isai_apply_to_single_vector_is_eqvt_to_unbatched(
        gko::preconditioner::batch_ilu_type::parilu, 10, 2, 3, 3,
        gko::preconditioner::batch_ilu_isai_apply::
            relaxation_steps_isai_simple);

    this->test_batch_ilu_isai_apply_to_single_vector_is_eqvt_to_unbatched(
        gko::preconditioner::batch_ilu_type::parilu, 10, 1, 2, 3,
        gko::preconditioner::batch_ilu_isai_apply::
            relaxation_steps_isai_simple);
}

// TODO: Implement BatchCsr Spegmm
// TYPED_TEST(BatchIluIsai,
//            BatchIluIsaiWithApplyTypeRelxationStepsSpgemmIsEquivalentToUnbatched)
// {
//     this->test_batch_ilu_isai_apply_to_single_vector_is_eqvt_to_unbatched(
//         gko::preconditioner::batch_ilu_type::exact_ilu, 10, 2, 3, 3,
//         gko::preconditioner::batch_ilu_isai_apply::
//             relaxation_steps_isai_with_spgemm);

//     this->test_batch_ilu_isai_apply_to_single_vector_is_eqvt_to_unbatched(
//         gko::preconditioner::batch_ilu_type::exact_ilu, 10, 1, 2, 3,
//         gko::preconditioner::batch_ilu_isai_apply::
//             relaxation_steps_isai_with_spgemm);

//     this->test_batch_ilu_isai_apply_to_single_vector_is_eqvt_to_unbatched(
//         gko::preconditioner::batch_ilu_type::parilu, 10, 2, 3, 3,
//         gko::preconditioner::batch_ilu_isai_apply::
//             relaxation_steps_isai_with_spgemm);

//     this->test_batch_ilu_isai_apply_to_single_vector_is_eqvt_to_unbatched(
//         gko::preconditioner::batch_ilu_type::parilu, 10, 1, 2, 3,
//         gko::preconditioner::batch_ilu_isai_apply::
//             relaxation_steps_isai_with_spgemm);
// }


TYPED_TEST(BatchIluIsai,
           BatchIluIsaiWithApplyTypeSpmvSimpleIsEquivalentToUnbatched)
{
    this->test_batch_ilu_isai_apply_to_single_vector_is_eqvt_to_unbatched(
        gko::preconditioner::batch_ilu_type::exact_ilu, 10, 2, 3, 3,
        gko::preconditioner::batch_ilu_isai_apply::spmv_isai_simple);

    this->test_batch_ilu_isai_apply_to_single_vector_is_eqvt_to_unbatched(
        gko::preconditioner::batch_ilu_type::exact_ilu, 10, 1, 2, 3,
        gko::preconditioner::batch_ilu_isai_apply::spmv_isai_simple);

    this->test_batch_ilu_isai_apply_to_single_vector_is_eqvt_to_unbatched(
        gko::preconditioner::batch_ilu_type::parilu, 10, 2, 3, 3,
        gko::preconditioner::batch_ilu_isai_apply::spmv_isai_simple);

    this->test_batch_ilu_isai_apply_to_single_vector_is_eqvt_to_unbatched(
        gko::preconditioner::batch_ilu_type::parilu, 10, 1, 2, 3,
        gko::preconditioner::batch_ilu_isai_apply::spmv_isai_simple);
}

// TODO: Implement BatchCsr Spegmm
// TYPED_TEST(BatchIluIsai,
//            BatchIluIsaiWithApplyTypeSpmvSpgemmIsEquivalentToUnbatched)
// {
//     this->test_batch_ilu_isai_apply_to_single_vector_is_eqvt_to_unbatched(
//         gko::preconditioner::batch_ilu_type::exact_ilu, 10, 2, 3, 3,
//         gko::preconditioner::batch_ilu_isai_apply::spmv_isai_with_spgemm);

//     this->test_batch_ilu_isai_apply_to_single_vector_is_eqvt_to_unbatched(
//         gko::preconditioner::batch_ilu_type::exact_ilu, 10, 1, 2, 3,
//         gko::preconditioner::batch_ilu_isai_apply::spmv_isai_with_spgemm);

//     this->test_batch_ilu_isai_apply_to_single_vector_is_eqvt_to_unbatched(
//         gko::preconditioner::batch_ilu_type::parilu, 10, 2, 3, 3,
//         gko::preconditioner::batch_ilu_isai_apply::spmv_isai_with_spgemm);

//     this->test_batch_ilu_isai_apply_to_single_vector_is_eqvt_to_unbatched(
//         gko::preconditioner::batch_ilu_type::parilu, 10, 1, 2, 3,
//         gko::preconditioner::batch_ilu_isai_apply::spmv_isai_with_spgemm);
// }


}  // namespace
