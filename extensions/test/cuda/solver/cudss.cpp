// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <fstream>
#include <memory>
#include <random>

#include <gtest/gtest.h>

#include <ginkgo/ginkgo.hpp>

#include <ginkgo/extensions/cuda/solver/cudss.hpp>

#include "core/test/utils.hpp"
#include "matrices/config.hpp"


namespace {


class CuDss : public ::testing::Test {
protected:
    using value_type = double;
    using index_type = gko::int32;
    using CuDssSolver = gko::ext::cuda::solver::CuDss<value_type, index_type>;
    using Direct = gko::experimental::solver::Direct<value_type, index_type>;
    using Lu = gko::experimental::factorization::Lu<value_type, index_type>;
    using matrix_type = gko::matrix::Csr<value_type, index_type>;
    using vector_type = gko::matrix::Dense<value_type>;

    CuDss()
        : ref(gko::ReferenceExecutor::create()),
          exec(gko::CudaExecutor::create(0, ref)),
          rand_engine(633)
    {}

    std::unique_ptr<vector_type> gen_mtx(gko::size_type num_rows,
                                         gko::size_type num_cols)
    {
        return gko::test::generate_random_matrix<vector_type>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    void initialize_data(const char* mtx_filename, int nrhs)
    {
        std::ifstream s_mtx{mtx_filename};
        mtx = gko::read<matrix_type>(s_mtx, ref);
        dmtx = gko::clone(exec, mtx);
        const auto num_rows = mtx->get_size()[0];

        ref_factory =
            Direct::build()
                .with_factorization(Lu::build().with_symbolic_algorithm(
                    gko::experimental::factorization::symbolic_type::symmetric))
                .with_num_rhs(static_cast<gko::size_type>(nrhs))
                .on(ref);

        cudss_factory = CuDssSolver::build().on(exec);

        alpha = gen_mtx(1, 1);
        beta = gen_mtx(1, 1);
        input = gen_mtx(num_rows, nrhs);
        output = gen_mtx(num_rows, nrhs);
        dalpha = gko::clone(exec, alpha);
        dbeta = gko::clone(exec, beta);
        dinput = gko::clone(exec, input);
        doutput = gko::clone(exec, output);
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::CudaExecutor> exec;
    std::default_random_engine rand_engine;
    std::unique_ptr<typename Direct::Factory> ref_factory;
    std::unique_ptr<typename CuDssSolver::Factory> cudss_factory;
    std::shared_ptr<matrix_type> mtx;
    std::shared_ptr<matrix_type> dmtx;
    std::shared_ptr<vector_type> alpha;
    std::shared_ptr<vector_type> beta;
    std::shared_ptr<vector_type> input;
    std::shared_ptr<vector_type> output;
    std::shared_ptr<vector_type> dalpha;
    std::shared_ptr<vector_type> dbeta;
    std::shared_ptr<vector_type> dinput;
    std::shared_ptr<vector_type> doutput;
};


TEST_F(CuDss, ApplyToSingleRhsMatchesRef)
{
    this->initialize_data(gko::matrices::location_ani4_amd_mtx, 1);
    auto ref_solver = this->ref_factory->generate(this->mtx);
    auto cudss_solver = this->cudss_factory->generate(this->dmtx);

    ref_solver->apply(this->input, this->output);
    cudss_solver->apply(this->dinput, this->doutput);

    GKO_ASSERT_MTX_NEAR(this->output, this->doutput, 100 * r<double>::value);
}


TEST_F(CuDss, ApplyToMultipleRhsMatchesRef)
{
    this->initialize_data(gko::matrices::location_ani4_amd_mtx, 6);
    auto ref_solver = this->ref_factory->generate(this->mtx);
    auto cudss_solver = this->cudss_factory->generate(this->dmtx);

    ref_solver->apply(this->input, this->output);
    cudss_solver->apply(this->dinput, this->doutput);

    GKO_ASSERT_MTX_NEAR(this->output, this->doutput, 100 * r<double>::value);
}


TEST_F(CuDss, AdvancedApplyMatchesRef)
{
    this->initialize_data(gko::matrices::location_ani4_amd_mtx, 1);
    auto ref_solver = this->ref_factory->generate(this->mtx);
    auto cudss_solver = this->cudss_factory->generate(this->dmtx);

    ref_solver->apply(this->alpha, this->input, this->beta, this->output);
    cudss_solver->apply(this->dalpha, this->dinput, this->dbeta, this->doutput);

    GKO_ASSERT_MTX_NEAR(this->output, this->doutput, 100 * r<double>::value);
}


}  // namespace
