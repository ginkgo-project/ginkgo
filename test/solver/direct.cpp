// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/factorization/lu.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>
#include <ginkgo/core/solver/direct.hpp>


#include "core/components/fill_array_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/factorization/cholesky_kernels.hpp"
#include "core/factorization/elimination_forest.hpp"
#include "core/factorization/lu_kernels.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/matrix/csr_lookup.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"
#include "matrices/config.hpp"
#include "test/utils/executor.hpp"


namespace {


template <typename ValueIndexType>
class Direct : public CommonTestFixture {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using factorization_type =
        gko::experimental::factorization::Lu<value_type, index_type>;
    using solver_type =
        gko::experimental::solver::Direct<value_type, index_type>;
    using matrix_type = typename factorization_type::matrix_type;
    using vector_type = gko::matrix::Dense<value_type>;

    Direct() : rand_engine(633) {}

    std::unique_ptr<vector_type> gen_mtx(gko::size_type num_rows,
                                         gko::size_type num_cols)
    {
        return gko::test::generate_random_matrix<vector_type>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<gko::remove_complex<value_type>>(-1.0,
                                                                      1.0),
            rand_engine, ref);
    }

    void initialize_data(const char* mtx_filename, int nrhs)
    {
        std::ifstream s_mtx{mtx_filename};
        mtx = gko::read<matrix_type>(s_mtx, ref);
        dmtx = gko::clone(exec, mtx);
        const auto num_rows = mtx->get_size()[0];
        factory = solver_type::build()
                      .with_factorization(
                          factorization_type::build().with_symbolic_algorithm(
                              gko::experimental::factorization::symbolic_type::
                                  symmetric))
                      .with_num_rhs(static_cast<gko::size_type>(nrhs))
                      .on(ref);
        alpha = gen_mtx(1, 1);
        beta = gen_mtx(1, 1);
        input = gen_mtx(num_rows, nrhs);
        output = gen_mtx(num_rows, nrhs);
        dfactory = solver_type::build()
                       .with_factorization(
                           factorization_type::build().with_symbolic_algorithm(
                               gko::experimental::factorization::symbolic_type::
                                   symmetric))
                       .with_num_rhs(static_cast<gko::size_type>(nrhs))
                       .on(exec);
        dalpha = gko::clone(exec, alpha);
        dbeta = gko::clone(exec, beta);
        dinput = gko::clone(exec, input);
        doutput = gko::clone(exec, output);
    }

    std::default_random_engine rand_engine;
    std::unique_ptr<typename solver_type::Factory> factory;
    std::shared_ptr<matrix_type> mtx;
    std::shared_ptr<vector_type> alpha;
    std::shared_ptr<vector_type> beta;
    std::shared_ptr<vector_type> input;
    std::shared_ptr<vector_type> output;
    std::unique_ptr<typename solver_type::Factory> dfactory;
    std::shared_ptr<matrix_type> dmtx;
    std::shared_ptr<vector_type> dalpha;
    std::shared_ptr<vector_type> dbeta;
    std::shared_ptr<vector_type> dinput;
    std::shared_ptr<vector_type> doutput;
};

#ifdef GKO_COMPILING_OMP
using Types = gko::test::ValueIndexTypes;
#elif defined(GKO_COMPILING_CUDA)
// CUDA don't support long indices for sorting, and the triangular solvers
// seem broken
using Types = ::testing::Types<std::tuple<float, gko::int32>,
                               std::tuple<double, gko::int32>,
                               std::tuple<std::complex<float>, gko::int32>,
                               std::tuple<std::complex<double>, gko::int32>>;
#else
// HIP only supports real types and int32
using Types = ::testing::Types<std::tuple<float, gko::int32>,
                               std::tuple<double, gko::int32>>;
#endif

TYPED_TEST_SUITE(Direct, Types, PairTypenameNameGenerator);


TYPED_TEST(Direct, ApplyToSingleRhsIsEquivalentToRef)
{
    using value_type = typename TestFixture::value_type;
    this->initialize_data(gko::matrices::location_ani4_amd_mtx, 1);
    auto solver = this->factory->generate(this->mtx);
    auto dsolver = this->dfactory->generate(this->dmtx);

    solver->apply(this->input, this->output);
    dsolver->apply(this->dinput, this->doutput);

    GKO_ASSERT_MTX_NEAR(this->output, this->doutput,
                        100 * r<value_type>::value);
}


TYPED_TEST(Direct, ApplyToMultipleRhsIsEquivalentToRef)
{
    using value_type = typename TestFixture::value_type;
    this->initialize_data(gko::matrices::location_ani4_amd_mtx, 6);
    auto solver = this->factory->generate(this->mtx);
    auto dsolver = this->dfactory->generate(this->dmtx);

    solver->apply(this->input, this->output);
    dsolver->apply(this->dinput, this->doutput);

    GKO_ASSERT_MTX_NEAR(this->output, this->doutput,
                        100 * r<value_type>::value);
}


TYPED_TEST(Direct, AdvancedApplyToSingleRhsIsEquivalentToRef)
{
    using value_type = typename TestFixture::value_type;
    this->initialize_data(gko::matrices::location_ani4_amd_mtx, 1);
    auto solver = this->factory->generate(this->mtx);
    auto dsolver = this->dfactory->generate(this->dmtx);

    solver->apply(this->alpha, this->input, this->beta, this->output);
    dsolver->apply(this->dalpha, this->dinput, this->dbeta, this->doutput);

    GKO_ASSERT_MTX_NEAR(this->output, this->doutput,
                        100 * r<value_type>::value);
}


TYPED_TEST(Direct, AdvancedApplyToMultipleRhsIsEquivalentToRef)
{
    using value_type = typename TestFixture::value_type;
    this->initialize_data(gko::matrices::location_ani4_amd_mtx, 6);
    auto solver = this->factory->generate(this->mtx);
    auto dsolver = this->dfactory->generate(this->dmtx);

    solver->apply(this->alpha, this->input, this->beta, this->output);
    dsolver->apply(this->dalpha, this->dinput, this->dbeta, this->doutput);

    GKO_ASSERT_MTX_NEAR(this->output, this->doutput,
                        100 * r<value_type>::value);
}


}  // namespace
