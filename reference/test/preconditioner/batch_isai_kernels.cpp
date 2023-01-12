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

#include <ginkgo/core/preconditioner/batch_isai.hpp>


#include <limits>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/factorization/ilu.hpp>
#include <ginkgo/core/preconditioner/ilu.hpp>
#include <ginkgo/core/preconditioner/isai.hpp>
#include <ginkgo/core/solver/upper_trs.hpp>


#include "core/factorization/factorization_kernels.hpp"
#include "core/factorization/ilu_kernels.hpp"
#include "core/preconditioner/batch_isai_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"

namespace {


template <typename T>
class BatchIsai : public ::testing::Test {
protected:
    using value_type = T;
    using index_type = int;
    using real_type = gko::remove_complex<value_type>;
    using Mtx = gko::matrix::BatchCsr<value_type, index_type>;
    using ubatched_mat_type = typename Mtx::unbatch_type;
    using BDense = gko::matrix::BatchDense<value_type>;
    using RBDense = gko::matrix::BatchDense<real_type>;
    using prec_type = gko::preconditioner::BatchIsai<value_type, index_type>;
    using unbatch_lower_prec_type =
        gko::preconditioner::Isai<gko::preconditioner::isai_type::lower,
                                  value_type, index_type>;
    using unbatch_gen_prec_type =
        gko::preconditioner::Isai<gko::preconditioner::isai_type::general,
                                  value_type, index_type>;
    using unbatch_upper_prec_type =
        gko::preconditioner::Isai<gko::preconditioner::isai_type::upper,
                                  value_type, index_type>;

    BatchIsai()
        : exec(gko::ReferenceExecutor::create()),
          general_mtx_small(get_general_matrix(false)),
          lower_tri_mtx_small(get_lower_matrix(false)),
          upper_tri_mtx_small(get_upper_matrix(false)),
          general_mtx_big(get_general_matrix(true)),
          lower_tri_mtx_big(get_lower_matrix(true)),
          upper_tri_mtx_big(get_upper_matrix(true))
    {}

    std::ranlux48 rand_engine;

    const size_t nbatch = 3;
    const index_type nrows_big = 70;
    const index_type nrows_small = 10;
    const index_type min_nnz_row_small = 3;
    const index_type min_nnz_row_big = 30;

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::shared_ptr<const Mtx> general_mtx_small;
    std::shared_ptr<const Mtx> lower_tri_mtx_small;
    std::shared_ptr<const Mtx> upper_tri_mtx_small;
    std::shared_ptr<const Mtx> general_mtx_big;
    std::shared_ptr<const Mtx> lower_tri_mtx_big;
    std::shared_ptr<const Mtx> upper_tri_mtx_big;

    std::unique_ptr<Mtx> get_general_matrix(bool is_big = false)
    {
        const auto nrows = is_big == true ? nrows_big : nrows_small;
        const auto min_nnz_row =
            is_big == true ? min_nnz_row_big : min_nnz_row_small;

        return gko::test::generate_uniform_batch_random_matrix<Mtx>(
            nbatch, nrows, nrows,
            std::uniform_int_distribution<>(min_nnz_row, nrows),
            std::normal_distribution<real_type>(0.0, 1.0), rand_engine, true,
            exec);
    }

    std::unique_ptr<Mtx> get_lower_matrix(bool is_big = false)
    {
        const auto nrows = is_big == true ? nrows_big : nrows_small;

        auto unbatch_mat =
            gko::test::generate_random_triangular_matrix<ubatched_mat_type>(
                nrows, false, true,
                std::uniform_int_distribution<>(nrows, nrows),
                std::normal_distribution<real_type>(0.0, 1.0), rand_engine,
                exec);

        return Mtx::create(exec, nbatch, unbatch_mat.get());
    }

    std::unique_ptr<Mtx> get_upper_matrix(bool is_big = false)
    {
        const auto nrows = is_big == true ? nrows_big : nrows_small;

        auto unbatch_mat =
            gko::test::generate_random_triangular_matrix<ubatched_mat_type>(
                nrows, false, false,
                std::uniform_int_distribution<>(nrows, nrows),
                std::normal_distribution<real_type>(0.0, 1.0), rand_engine,
                exec);

        return Mtx::create(exec, nbatch, unbatch_mat.get());
    }


    // TODO: Add tests for non-sorted input matrix
    void test_batch_isai_generation_is_eqvt_to_unbatched(
        const int spy_power, std::string type, bool test_extension = false)
    {
        using unbatch_type = typename Mtx::unbatch_type;

        auto batch_isai_type =
            gko::preconditioner::batch_isai_input_matrix_type::general;
        auto mtx = general_mtx_small;
        auto mtxs = gko::test::share(mtx->unbatch());
        std::vector<std::shared_ptr<const unbatch_type>> check_isai(
            this->nbatch);


        if (type == std::string("lower")) {
            mtx = test_extension == true ? lower_tri_mtx_big
                                         : lower_tri_mtx_small;

            mtxs = gko::test::share(mtx->unbatch());
            batch_isai_type =
                gko::preconditioner::batch_isai_input_matrix_type::lower_tri;
            auto unbatch_prec_fact = unbatch_lower_prec_type::build()
                                         .with_skip_sorting(true)
                                         .with_sparsity_power(spy_power)
                                         .on(exec);
            for (size_t i = 0; i < mtxs.size(); i++) {
                auto unbatch_prec = unbatch_prec_fact->generate(mtxs[i]);
                check_isai[i] = unbatch_prec->get_approximate_inverse();
            }

        } else if (type == std::string("upper")) {
            mtx = test_extension == true ? upper_tri_mtx_big
                                         : upper_tri_mtx_small;

            mtxs = gko::test::share(mtx->unbatch());
            batch_isai_type =
                gko::preconditioner::batch_isai_input_matrix_type::upper_tri;

            auto unbatch_prec_fact = unbatch_upper_prec_type::build()
                                         .with_skip_sorting(true)
                                         .with_sparsity_power(spy_power)
                                         .on(exec);

            for (size_t i = 0; i < mtxs.size(); i++) {
                auto unbatch_prec = unbatch_prec_fact->generate(mtxs[i]);
                check_isai[i] = unbatch_prec->get_approximate_inverse();
            }

        } else if (type == std::string("general")) {
            mtx = test_extension == true ? general_mtx_big : general_mtx_small;

            mtxs = gko::test::share(mtx->unbatch());
            batch_isai_type =
                gko::preconditioner::batch_isai_input_matrix_type::general;

            auto unbatch_prec_fact = unbatch_gen_prec_type::build()
                                         .with_skip_sorting(true)
                                         .with_sparsity_power(spy_power)
                                         .on(exec);

            for (size_t i = 0; i < mtxs.size(); i++) {
                auto unbatch_prec = unbatch_prec_fact->generate(mtxs[i]);
                check_isai[i] = unbatch_prec->get_approximate_inverse();
            }
        } else {
            GKO_NOT_IMPLEMENTED;
        }

        auto prec_fact = prec_type::build()
                             .with_skip_sorting(true)
                             .with_sparsity_power(spy_power)
                             .with_isai_input_matrix_type(batch_isai_type)
                             .on(exec);
        auto prec = prec_fact->generate(mtx);

        auto approx_inv = prec->get_const_approximate_inverse();
        auto approx_inv_vec = approx_inv->unbatch();

        for (size_t i = 0; i < nbatch; i++) {
            GKO_ASSERT_MTX_NEAR(approx_inv_vec[i], check_isai[i],
                                50 * r<value_type>::value);
        }
    }

    // TODO: Add tests for non-sorted input matrix
    void test_batch_isai_apply_to_single_vector_is_eqvt_to_unbatched(
        const int spy_power, std::string type, bool test_extension = false)
    {
        using unbatch_type = typename Mtx::unbatch_type;

        auto batch_isai_type =
            gko::preconditioner::batch_isai_input_matrix_type::general;
        auto mtx = general_mtx_small;
        auto umtxs = gko::test::share(mtx->unbatch());

        auto nrows = test_extension == true ? nrows_big : nrows_small;

        auto b = gko::test::generate_uniform_batch_random_matrix<BDense>(
            nbatch, nrows, 1, std::uniform_int_distribution<>(1, 1),
            std::normal_distribution<real_type>(0.0, 1.0), rand_engine, false,
            exec);

        auto x = BDense::create(
            exec, gko::batch_dim<>(nbatch, gko::dim<2>(nrows, 1)));

        auto ub = b->unbatch();
        auto ux = x->unbatch();

        if (type == std::string("lower")) {
            mtx = test_extension == true ? lower_tri_mtx_big
                                         : lower_tri_mtx_small;

            umtxs = gko::test::share(mtx->unbatch());
            batch_isai_type =
                gko::preconditioner::batch_isai_input_matrix_type::lower_tri;

            auto unbatch_prec_fact = unbatch_lower_prec_type::build()
                                         .with_skip_sorting(true)
                                         .with_sparsity_power(spy_power)
                                         .on(exec);

            for (size_t i = 0; i < umtxs.size(); i++) {
                auto unbatch_prec = unbatch_prec_fact->generate(umtxs[i]);
                unbatch_prec->apply(ub[i].get(), ux[i].get());
            }

        } else if (type == std::string("upper")) {
            mtx = test_extension == true ? upper_tri_mtx_big
                                         : upper_tri_mtx_small;

            umtxs = gko::test::share(mtx->unbatch());
            batch_isai_type =
                gko::preconditioner::batch_isai_input_matrix_type::upper_tri;

            auto unbatch_prec_fact = unbatch_upper_prec_type::build()
                                         .with_skip_sorting(true)
                                         .with_sparsity_power(spy_power)
                                         .on(exec);

            for (size_t i = 0; i < umtxs.size(); i++) {
                auto unbatch_prec = unbatch_prec_fact->generate(umtxs[i]);
                unbatch_prec->apply(ub[i].get(), ux[i].get());
            }


        } else if (type == std::string("general")) {
            mtx = test_extension == true ? general_mtx_big : general_mtx_small;

            umtxs = gko::test::share(mtx->unbatch());
            batch_isai_type =
                gko::preconditioner::batch_isai_input_matrix_type::general;

            auto unbatch_prec_fact = unbatch_gen_prec_type::build()
                                         .with_skip_sorting(true)
                                         .with_sparsity_power(spy_power)
                                         .on(exec);

            for (size_t i = 0; i < umtxs.size(); i++) {
                auto unbatch_prec = unbatch_prec_fact->generate(umtxs[i]);
                unbatch_prec->apply(ub[i].get(), ux[i].get());
            }

        } else {
            GKO_NOT_IMPLEMENTED;
        }


        auto prec_fact = prec_type::build()
                             .with_skip_sorting(true)
                             .with_sparsity_power(spy_power)
                             .with_isai_input_matrix_type(batch_isai_type)
                             .on(exec);
        auto prec = prec_fact->generate(mtx);
        auto approx_inv = prec->get_const_approximate_inverse();

        gko::kernels::reference::batch_isai::apply_isai(
            exec, mtx.get(), approx_inv.get(), b.get(), x.get());

        auto xs = x->unbatch();
        for (size_t i = 0; i < umtxs.size(); i++) {
            GKO_ASSERT_MTX_NEAR(ux[i], xs[i], 50 * r<value_type>::value);
        }
    }
};

TYPED_TEST_SUITE(BatchIsai, gko::test::ValueTypes);


TYPED_TEST(BatchIsai, GeneralBatchIsaiGenerationIsEquivalentToUnbatchedWithSpy1)
{
    this->test_batch_isai_generation_is_eqvt_to_unbatched(
        1, std::string("general"), false);
}


// NOTE: Test fails because the batched iterative solver does not converge for
// some systems
// Note: To ensure that the general isai extension implementation is correct, I
// tested the kernels for cases where the iterative solver converges for all the
// batched systems produced in the inverse generation process. To get such
// cases, I reduced the row_size_limit to 2 and used a very small matrix as
// input.
// TYPED_TEST(BatchIsai,
// ExtendedGeneralBatchIsaiGenerationIsEquivalentToUnbatchedWithSpy1)
// {
//     this->test_batch_isai_generation_is_eqvt_to_unbatched(
//         1, std::string("general"), true);
// }


// TODO: Fix bug in normal isai
// TYPED_TEST(BatchIsai,
// GeneralBatchIsaiGenerationIsEquivalentToUnbatchedWithSpy2)
// {
//     this->test_batch_isai_generation_is_eqvt_to_unbatched(
//         2, std::string("general"), false);
// }


// TYPED_TEST(BatchIsai,
// ExtendedGeneralBatchIsaiGenerationIsEquivalentToUnbatchedWithSpy2)
// {
//     this->test_batch_isai_generation_is_eqvt_to_unbatched(
//         2, std::string("general"), true);
// }

// TODO: Fix bug in normal isai
// TYPED_TEST(BatchIsai,
// GeneralBatchIsaiGenerationIsEquivalentToUnbatchedWithSpy3)
// {
//     this->test_batch_isai_generation_is_eqvt_to_unbatched(
//         3, std::string("general"), false);
// }

// TYPED_TEST(BatchIsai,
// ExtendedGeneralBatchIsaiGenerationIsEquivalentToUnbatchedWithSpy3)
// {
//     this->test_batch_isai_generation_is_eqvt_to_unbatched(
//         3, std::string("general"), true);
// }


TYPED_TEST(BatchIsai, LowerBatchIsaiGenerationIsEquivalentToUnbatchedWithSpy1)
{
    this->test_batch_isai_generation_is_eqvt_to_unbatched(
        1, std::string("lower"), false);
}


TYPED_TEST(BatchIsai,
           ExtendedLowerBatchIsaiGenerationIsEquivalentToUnbatchedWithSpy1)
{
    this->test_batch_isai_generation_is_eqvt_to_unbatched(
        1, std::string("lower"), true);
}


TYPED_TEST(BatchIsai, LowerBatchIsaiGenerationIsEquivalentToUnbatchedWithSpy2)
{
    this->test_batch_isai_generation_is_eqvt_to_unbatched(
        2, std::string("lower"), false);
}


TYPED_TEST(BatchIsai,
           ExtendedLowerBatchIsaiGenerationIsEquivalentToUnbatchedWithSpy2)
{
    this->test_batch_isai_generation_is_eqvt_to_unbatched(
        2, std::string("lower"), true);
}


TYPED_TEST(BatchIsai, LowerBatchIsaiGenerationIsEquivalentToUnbatchedWithSpy3)
{
    this->test_batch_isai_generation_is_eqvt_to_unbatched(
        3, std::string("lower"), false);
}


TYPED_TEST(BatchIsai,
           ExtendedLowerBatchIsaiGenerationIsEquivalentToUnbatchedWithSpy3)
{
    this->test_batch_isai_generation_is_eqvt_to_unbatched(
        3, std::string("lower"), true);
}

TYPED_TEST(BatchIsai, UpperBatchIsaiGenerationIsEquivalentToUnbatchedWithSpy1)
{
    this->test_batch_isai_generation_is_eqvt_to_unbatched(
        1, std::string("upper"), false);
}


TYPED_TEST(BatchIsai,
           ExtendedUpperBatchIsaiGenerationIsEquivalentToUnbatchedWithSpy1)
{
    this->test_batch_isai_generation_is_eqvt_to_unbatched(
        1, std::string("upper"), true);
}


TYPED_TEST(BatchIsai, UpperBatchIsaiGenerationIsEquivalentToUnbatchedWithSpy2)
{
    this->test_batch_isai_generation_is_eqvt_to_unbatched(
        2, std::string("upper"), false);
}

TYPED_TEST(BatchIsai,
           ExtendedUpperBatchIsaiGenerationIsEquivalentToUnbatchedWithSpy2)
{
    this->test_batch_isai_generation_is_eqvt_to_unbatched(
        2, std::string("upper"), true);
}

TYPED_TEST(BatchIsai, UpperBatchIsaiGenerationIsEquivalentToUnbatchedWithSpy3)
{
    this->test_batch_isai_generation_is_eqvt_to_unbatched(
        3, std::string("upper"), false);
}

TYPED_TEST(BatchIsai,
           ExtendedUpperBatchIsaiGenerationIsEquivalentToUnbatchedWithSpy3)
{
    this->test_batch_isai_generation_is_eqvt_to_unbatched(
        3, std::string("upper"), true);
}


TYPED_TEST(BatchIsai,
           GeneralBatchIsaiApplyToSingleVectorIsEquivalentToUnbatchedWithSpy1)
{
    this->test_batch_isai_apply_to_single_vector_is_eqvt_to_unbatched(
        1, std::string("general"), false);
}


// NOTE: Test fails because the batched iterative solver does not converge for
// some systems
// TYPED_TEST(BatchIsai,
//            ExtendedGeneralBatchIsaiApplyToSingleVectorIsEquivalentToUnbatchedWithSpy1)
// {
//     this->test_batch_isai_apply_to_single_vector_is_eqvt_to_unbatched(
//         1, std::string("general"), true);
// }


// TODO: Fix bug in normal isai
// TYPED_TEST(BatchIsai,
//            GeneralBatchIsaiApplyToSingleVectorIsEquivalentToUnbatchedWithSpy2)
// {
//     this->test_batch_isai_apply_to_single_vector_is_eqvt_to_unbatched(
//         2, std::string("general"), false);
// }

// TYPED_TEST(BatchIsai,
//            ExtendedGeneralBatchIsaiApplyToSingleVectorIsEquivalentToUnbatchedWithSpy2)
// {
//     this->test_batch_isai_apply_to_single_vector_is_eqvt_to_unbatched(
//         2, std::string("general"), true);
// }

// TODO: Fix bug in normal isai
// TYPED_TEST(BatchIsai,
//            GeneralBatchIsaiApplyToSingleVectorIsEquivalentToUnbatchedWithSpy3)
// {
//     this->test_batch_isai_apply_to_single_vector_is_eqvt_to_unbatched(
//         3, std::string("general"), false);
// }

// TYPED_TEST(BatchIsai,
//            ExtendedGeneralBatchIsaiApplyToSingleVectorIsEquivalentToUnbatchedWithSpy3)
// {
//     this->test_batch_isai_apply_to_single_vector_is_eqvt_to_unbatched(
//         3, std::string("general"), true);
// }


TYPED_TEST(BatchIsai,
           LowerBatchIsaiApplyToSingleVectorIsEquivalentToUnbatchedWithSpy1)
{
    this->test_batch_isai_apply_to_single_vector_is_eqvt_to_unbatched(
        1, std::string("lower"), false);
}


TYPED_TEST(
    BatchIsai,
    ExtendedLowerBatchIsaiApplyToSingleVectorIsEquivalentToUnbatchedWithSpy1)
{
    this->test_batch_isai_apply_to_single_vector_is_eqvt_to_unbatched(
        1, std::string("lower"), true);
}


TYPED_TEST(BatchIsai,
           LowerBatchIsaiApplyToSingleVectorIsEquivalentToUnbatchedWithSpy2)
{
    this->test_batch_isai_apply_to_single_vector_is_eqvt_to_unbatched(
        2, std::string("lower"), false);
}

TYPED_TEST(
    BatchIsai,
    ExtendedLowerBatchIsaiApplyToSingleVectorIsEquivalentToUnbatchedWithSpy2)
{
    this->test_batch_isai_apply_to_single_vector_is_eqvt_to_unbatched(
        2, std::string("lower"), true);
}

TYPED_TEST(BatchIsai,
           LowerBatchIsaiApplyToSingleVectorIsEquivalentToUnbatchedWithSpy3)
{
    this->test_batch_isai_apply_to_single_vector_is_eqvt_to_unbatched(
        3, std::string("lower"), false);
}

TYPED_TEST(
    BatchIsai,
    ExtendedLowerBatchIsaiApplyToSingleVectorIsEquivalentToUnbatchedWithSpy3)
{
    this->test_batch_isai_apply_to_single_vector_is_eqvt_to_unbatched(
        3, std::string("lower"), true);
}


TYPED_TEST(BatchIsai,
           UpperBatchIsaiApplyToSingleVectorIsEquivalentToUnbatchedWithSpy1)
{
    this->test_batch_isai_apply_to_single_vector_is_eqvt_to_unbatched(
        1, std::string("upper"), false);
}


TYPED_TEST(
    BatchIsai,
    ExtendedUpperBatchIsaiApplyToSingleVectorIsEquivalentToUnbatchedWithSpy1)
{
    this->test_batch_isai_apply_to_single_vector_is_eqvt_to_unbatched(
        1, std::string("upper"), true);
}


TYPED_TEST(BatchIsai,
           UpperBatchIsaiApplyToSingleVectorIsEquivalentToUnbatchedWithSpy2)
{
    this->test_batch_isai_apply_to_single_vector_is_eqvt_to_unbatched(
        2, std::string("upper"), false);
}


TYPED_TEST(
    BatchIsai,
    ExtendedUpperBatchIsaiApplyToSingleVectorIsEquivalentToUnbatchedWithSpy2)
{
    this->test_batch_isai_apply_to_single_vector_is_eqvt_to_unbatched(
        2, std::string("upper"), true);
}

TYPED_TEST(BatchIsai,
           UpperBatchIsaiApplyToSingleVectorIsEquivalentToUnbatchedWithSpy3)
{
    this->test_batch_isai_apply_to_single_vector_is_eqvt_to_unbatched(
        3, std::string("upper"), false);
}


TYPED_TEST(
    BatchIsai,
    ExtendedUpperBatchIsaiApplyToSingleVectorIsEquivalentToUnbatchedWithSpy3)
{
    this->test_batch_isai_apply_to_single_vector_is_eqvt_to_unbatched(
        3, std::string("upper"), true);
}


}  // namespace
