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
          general_mtx(get_general_matrix()),
          lower_tri_mtx(get_lower_matrix()),
          upper_tri_mtx(get_upper_matrix())
    {}

    const size_t nbatch = 2;
    const index_type nrows = 4;
    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::shared_ptr<const Mtx> general_mtx;
    std::shared_ptr<const Mtx> lower_tri_mtx;
    std::shared_ptr<const Mtx> upper_tri_mtx;


    std::unique_ptr<Mtx> get_general_matrix()
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

    std::unique_ptr<Mtx> get_lower_matrix()
    {
        auto mat = Mtx::create(exec, nbatch, gko::dim<2>(nrows, nrows), 7);
        index_type* const row_ptrs = mat->get_row_ptrs();
        index_type* const col_idxs = mat->get_col_idxs();
        value_type* const vals = mat->get_values();
        // clang-format off
		row_ptrs[0] = 0; row_ptrs[1] = 1; row_ptrs[2] = 3; row_ptrs[3] = 5; row_ptrs[4] = 7;
		col_idxs[0] = 0; col_idxs[1] = 0; col_idxs[2] = 1; col_idxs[3] = 1;
		col_idxs[4] = 2; col_idxs[5] = 0, col_idxs[6] = 3;
		vals[0] = 2.0; vals[1] = 0.25; vals[2] = -1.0; vals[3] = -3.0;
		vals[4] = 2.0; vals[5] = 0.2;
		vals[6] = -1.5; vals[7] = 0.55; vals[8] = -1.0; vals[9] = 4.0;
        vals[10] = -3.5; vals[11] = 0.45; vals[12] = -5.0; vals[13] = 12;
        // clang-format on
        return mat;
    }

    std::unique_ptr<Mtx> get_upper_matrix()
    {
        auto mat = Mtx::create(exec, nbatch, gko::dim<2>(nrows, nrows), 7);
        index_type* const row_ptrs = mat->get_row_ptrs();
        index_type* const col_idxs = mat->get_col_idxs();
        value_type* const vals = mat->get_values();
        // clang-format off
		row_ptrs[0] = 0; row_ptrs[1] = 2; row_ptrs[2] = 4; row_ptrs[3] = 6; row_ptrs[4] = 7;
		col_idxs[0] = 0; col_idxs[1] = 3; col_idxs[2] = 1; col_idxs[3] = 2;
		col_idxs[4] = 2; col_idxs[5] = 3; col_idxs[6] = 3;
		vals[0] = 2.0; vals[1] = 0.25; vals[2] = -1.0; vals[3] = -3.0;
		vals[4] = 2.0; vals[5] = 0.2;
		vals[6] = -1.5; vals[7] = 0.55; vals[8] = -1.0; vals[9] = 4.0;
		vals[10] = -3.5; vals[11] = 0.45; vals[12] = -5.0; vals[13] = 12;
        // clang-format on
        return mat;
    }

    // TODO: Add tests for non-sorted input matrix
    void test_batch_isai_generation_is_eqvt_to_unbatched(const int spy_power,
                                                         std::string type)
    {
        using unbatch_type = typename Mtx::unbatch_type;

        auto batch_isai_type =
            gko::preconditioner::batch_isai_input_matrix_type::general;
        auto mtx = general_mtx;
        auto mtxs = gko::test::share(mtx->unbatch());
        std::vector<std::shared_ptr<const unbatch_type>> check_isai(
            this->nbatch);


        if (type == std::string("lower")) {
            mtx = lower_tri_mtx;
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
            mtx = upper_tri_mtx;
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
            mtx = general_mtx;
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
                                r<value_type>::value);
        }
    }

    // TODO: Add tests for non-sorted input matrix
    void test_batch_isai_apply_to_single_vector_is_eqvt_to_unbatched(
        const int spy_power, std::string type)
    {
        using unbatch_type = typename Mtx::unbatch_type;

        auto batch_isai_type =
            gko::preconditioner::batch_isai_input_matrix_type::general;
        auto mtx = general_mtx;
        auto umtxs = gko::test::share(mtx->unbatch());
        auto b = gko::batch_initialize<BDense>(
            {{-2.0, 9.0, 4.0, 7.0}, {-3.0, 5.0, 3.0, 10.0}}, exec);
        auto x = BDense::create(
            exec, gko::batch_dim<>(2, gko::dim<2>(this->nrows, 1)));
        auto ub = b->unbatch();
        auto ux = x->unbatch();

        if (type == std::string("lower")) {
            mtx = lower_tri_mtx;
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
            mtx = upper_tri_mtx;
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
            mtx = general_mtx;
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
            GKO_ASSERT_MTX_NEAR(ux[i], xs[i], r<value_type>::value);
        }
    }
};

TYPED_TEST_SUITE(BatchIsai, gko::test::ValueTypes);


TYPED_TEST(BatchIsai, GeneralBatchIsaiGenerationIsEquivalentToUnbatchedWithSpy1)
{
    this->test_batch_isai_generation_is_eqvt_to_unbatched(
        1, std::string("general"));
}

// TODO: Fix bug in normal isai
// TYPED_TEST(BatchIsai,
// GeneralBatchIsaiGenerationIsEquivalentToUnbatchedWithSpy2)
// {
//     this->test_batch_isai_generation_is_eqvt_to_unbatched(
//         2, std::string("general"));
// }


TYPED_TEST(BatchIsai, LowerBatchIsaiGenerationIsEquivalentToUnbatchedWithSpy1)
{
    this->test_batch_isai_generation_is_eqvt_to_unbatched(1,
                                                          std::string("lower"));
}


TYPED_TEST(BatchIsai, LowerBatchIsaiGenerationIsEquivalentToUnbatchedWithSpy2)
{
    this->test_batch_isai_generation_is_eqvt_to_unbatched(2,
                                                          std::string("lower"));
}


TYPED_TEST(BatchIsai, UpperBatchIsaiGenerationIsEquivalentToUnbatchedWithSpy1)
{
    this->test_batch_isai_generation_is_eqvt_to_unbatched(1,
                                                          std::string("upper"));
}


TYPED_TEST(BatchIsai, UpperBatchIsaiGenerationIsEquivalentToUnbatchedWithSpy2)
{
    this->test_batch_isai_generation_is_eqvt_to_unbatched(2,
                                                          std::string("upper"));
}


TYPED_TEST(BatchIsai,
           GeneralBatchIsaiApplyToSingleVectorIsEquivalentToUnbatchedWithSpy1)
{
    this->test_batch_isai_apply_to_single_vector_is_eqvt_to_unbatched(
        1, std::string("general"));
}

// TODO: Fix bug in normal isai
// TYPED_TEST(BatchIsai,
//            GeneralBatchIsaiApplyToSingleVectorIsEquivalentToUnbatchedWithSpy2)
// {
//     this->test_batch_isai_apply_to_single_vector_is_eqvt_to_unbatched(
//         2, std::string("general"));
// }


TYPED_TEST(BatchIsai,
           LowerBatchIsaiApplyToSingleVectorIsEquivalentToUnbatchedWithSpy1)
{
    this->test_batch_isai_apply_to_single_vector_is_eqvt_to_unbatched(
        1, std::string("lower"));
}


TYPED_TEST(BatchIsai,
           LowerBatchIsaiApplyToSingleVectorIsEquivalentToUnbatchedWithSpy2)
{
    this->test_batch_isai_apply_to_single_vector_is_eqvt_to_unbatched(
        2, std::string("lower"));
}


TYPED_TEST(BatchIsai,
           UpperBatchIsaiApplyToSingleVectorIsEquivalentToUnbatchedWithSpy1)
{
    this->test_batch_isai_apply_to_single_vector_is_eqvt_to_unbatched(
        1, std::string("upper"));
}


TYPED_TEST(BatchIsai,
           UpperBatchIsaiApplyToSingleVectorIsEquivalentToUnbatchedWithSpy2)
{
    this->test_batch_isai_apply_to_single_vector_is_eqvt_to_unbatched(
        2, std::string("upper"));
}

}  // namespace
