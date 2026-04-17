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

    GKO_ASSERT_MTX_NEAR(this->output, this->doutput,
                        100 * r<value_type>::value);
}


TEST_F(CuDss, ApplyToMultipleRhsMatchesRef)
{
    this->initialize_data(gko::matrices::location_ani4_amd_mtx, 6);
    auto ref_solver = this->ref_factory->generate(this->mtx);
    auto cudss_solver = this->cudss_factory->generate(this->dmtx);

    ref_solver->apply(this->input, this->output);
    cudss_solver->apply(this->dinput, this->doutput);

    GKO_ASSERT_MTX_NEAR(this->output, this->doutput,
                        100 * r<value_type>::value);
}


TEST_F(CuDss, AdvancedApplyMatchesRef)
{
    this->initialize_data(gko::matrices::location_ani4_amd_mtx, 1);
    auto ref_solver = this->ref_factory->generate(this->mtx);
    auto cudss_solver = this->cudss_factory->generate(this->dmtx);

    ref_solver->apply(this->alpha, this->input, this->beta, this->output);
    cudss_solver->apply(this->dalpha, this->dinput, this->dbeta, this->doutput);

    GKO_ASSERT_MTX_NEAR(this->output, this->doutput,
                        100 * r<value_type>::value);
}


TEST_F(CuDss, RefactorizeWithUpdatedValuesMatchesRef)
{
    this->initialize_data(gko::matrices::location_ani4_amd_mtx, 1);
    auto cudss_solver = this->cudss_factory->generate(this->dmtx);
    // First solve with original matrix
    cudss_solver->apply(this->dinput, this->doutput);
    // Scale all matrix values by 2 — same sparsity, different numerics
    auto scaled_mtx = gko::share(gko::clone(this->ref, this->mtx));
    for (gko::size_type i = 0; i < scaled_mtx->get_num_stored_elements(); ++i) {
        scaled_mtx->get_values()[i] *= 2.0;
    }
    auto d_scaled_mtx = gko::share(gko::clone(this->exec, scaled_mtx));

    // Reference: generate a fresh solver with the scaled matrix
    auto ref_solver = this->ref_factory->generate(scaled_mtx);
    ref_solver->apply(this->input, this->output);
    // CuDss: refactorize with the scaled matrix, then solve
    cudss_solver->refactorize(d_scaled_mtx);
    cudss_solver->apply(this->dinput, this->doutput);

    GKO_ASSERT_MTX_NEAR(this->output, this->doutput,
                        100 * r<value_type>::value);
}


TEST_F(CuDss, ApplyToStridedSingleRhsMatchesRef)
{
    this->initialize_data(gko::matrices::location_ani4_amd_mtx, 1);
    auto ref_solver = this->ref_factory->generate(this->mtx);
    auto cudss_solver = this->cudss_factory->generate(this->dmtx);
    const auto nrows = this->mtx->get_size()[0];
    // Create a wider dense matrix and take a column view to get strided vectors
    auto wide_input = vector_type::create(this->ref, gko::dim<2>{nrows, 3});
    auto wide_output = vector_type::create(this->ref, gko::dim<2>{nrows, 3});
    // Copy input into column 1 (middle column, stride = 3)
    for (gko::size_type i = 0; i < nrows; ++i) {
        wide_input->at(i, 1) = this->input->at(i, 0);
        wide_output->at(i, 1) = this->output->at(i, 0);
    }
    auto d_wide_input = gko::clone(this->exec, wide_input);
    auto d_wide_output = gko::clone(this->exec, wide_output);
    // Create strided submatrix views (column 1 of the 3-column matrix)
    auto strided_input =
        d_wide_input->create_submatrix(gko::span{0, nrows}, gko::span{1, 2});
    auto strided_output =
        d_wide_output->create_submatrix(gko::span{0, nrows}, gko::span{1, 2});

    // Reference solve with non-strided vectors
    ref_solver->apply(this->input, this->output);

    // CuDss solve with strided vectors
    cudss_solver->apply(strided_input, strided_output);

    GKO_ASSERT_MTX_NEAR(this->output, strided_output,
                        100 * r<value_type>::value);
}


TEST_F(CuDss, ApplyToStridedMultipleRhsMatchesRef)
{
    this->initialize_data(gko::matrices::location_ani4_amd_mtx, 3);
    auto ref_solver = this->ref_factory->generate(this->mtx);
    auto cudss_solver = this->cudss_factory->generate(this->dmtx);
    const auto nrows = this->mtx->get_size()[0];
    // Create a wider matrix (6 cols) and take columns 1..3 as a strided view
    auto wide_input = vector_type::create(this->ref, gko::dim<2>{nrows, 6});
    auto wide_output = vector_type::create(this->ref, gko::dim<2>{nrows, 6});
    for (gko::size_type i = 0; i < nrows; ++i) {
        for (gko::size_type j = 0; j < 3; ++j) {
            wide_input->at(i, j + 1) = this->input->at(i, j);
            wide_output->at(i, j + 1) = this->output->at(i, j);
        }
    }
    auto d_wide_input = gko::clone(this->exec, wide_input);
    auto d_wide_output = gko::clone(this->exec, wide_output);
    auto strided_input =
        d_wide_input->create_submatrix(gko::span{0, nrows}, gko::span{1, 4});
    auto strided_output =
        d_wide_output->create_submatrix(gko::span{0, nrows}, gko::span{1, 4});

    ref_solver->apply(this->input, this->output);
    cudss_solver->apply(strided_input, strided_output);

    GKO_ASSERT_MTX_NEAR(this->output, strided_output,
                        100 * r<value_type>::value);
}


TEST_F(CuDss, ParseConfigCreatesCorrectFactory)
{
    auto config_map = CuDssSolver::get_config_map();
    auto reg = gko::config::registry{config_map};
    gko::config::pnode::map_type conf_map;
    conf_map["type"] = gko::config::pnode{"ext::cuda::solver::CuDss"};
    conf_map["matrix_type"] = gko::config::pnode{3};
    conf_map["matrix_view"] = gko::config::pnode{2};
    conf_map["reordering_alg"] = gko::config::pnode{1};
    auto conf = gko::config::pnode{conf_map};

    auto params = CuDssSolver::parse(conf, reg);

    ASSERT_EQ(params.matrix_type, 3);
    ASSERT_EQ(params.matrix_view, 2);
    ASSERT_EQ(params.reordering_alg, 1);
    ASSERT_EQ(params.hybrid_execute, false);
    ASSERT_EQ(params.hybrid_memory, false);
}


TEST_F(CuDss, ParseConfigThrowsOnUnknownKey)
{
    auto config_map = CuDssSolver::get_config_map();
    auto reg = gko::config::registry{config_map};
    gko::config::pnode::map_type conf_map;
    conf_map["type"] = gko::config::pnode{"ext::cuda::solver::CuDss"};
    conf_map["invalid_key"] = gko::config::pnode{42};
    auto conf = gko::config::pnode{conf_map};

    ASSERT_THROW(CuDssSolver::parse(conf, reg), gko::InvalidStateError);
}


}  // namespace
