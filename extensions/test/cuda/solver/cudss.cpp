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


class Cudss : public ::testing::Test {
protected:
    using value_type = double;
    using index_type = gko::int32;
    using CudssSolver = gko::ext::cuda::solver::Cudss<value_type, index_type>;
    using Direct = gko::experimental::solver::Direct<value_type, index_type>;
    using Lu = gko::experimental::factorization::Lu<value_type, index_type>;
    using matrix_type = gko::matrix::Csr<value_type, index_type>;
    using vector_type = gko::matrix::Dense<value_type>;

    Cudss()
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

        cudss_factory = CudssSolver::build().on(exec);

        alpha = gen_mtx(1, 1);
        beta = gen_mtx(1, 1);
        // input = A * solution
        solution = gen_mtx(num_rows, nrhs);
        input = vector_type::create(
            ref, gko::dim<2>{num_rows, static_cast<gko::size_type>(nrhs)});
        mtx->apply(solution, input);
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
    std::unique_ptr<typename CudssSolver::Factory> cudss_factory;
    std::shared_ptr<matrix_type> mtx;
    std::shared_ptr<matrix_type> dmtx;
    std::shared_ptr<vector_type> alpha;
    std::shared_ptr<vector_type> beta;
    std::shared_ptr<vector_type> solution;
    std::shared_ptr<vector_type> input;
    std::shared_ptr<vector_type> output;
    std::shared_ptr<vector_type> dalpha;
    std::shared_ptr<vector_type> dbeta;
    std::shared_ptr<vector_type> dinput;
    std::shared_ptr<vector_type> doutput;
};


TEST_F(Cudss, ApplyToSingleRhsMatchesRef)
{
    this->initialize_data(gko::matrices::location_ani4_amd_mtx, 1);
    auto cudss_solver = this->cudss_factory->generate(this->dmtx);

    cudss_solver->apply(this->dinput, this->doutput);

    GKO_ASSERT_MTX_NEAR(this->doutput, this->solution,
                        100 * r<value_type>::value);
}


TEST_F(Cudss, ApplyToMultipleRhsMatchesRef)
{
    this->initialize_data(gko::matrices::location_ani4_amd_mtx, 6);
    auto cudss_solver = this->cudss_factory->generate(this->dmtx);

    cudss_solver->apply(this->dinput, this->doutput);

    GKO_ASSERT_MTX_NEAR(this->doutput, this->solution,
                        100 * r<value_type>::value);
}


TEST_F(Cudss, AdvancedApplyMatchesRef)
{
    this->initialize_data(gko::matrices::location_ani4_amd_mtx, 1);
    auto ref_solver = this->ref_factory->generate(this->mtx);
    auto cudss_solver = this->cudss_factory->generate(this->dmtx);

    ref_solver->apply(this->alpha, this->input, this->beta, this->output);
    cudss_solver->apply(this->dalpha, this->dinput, this->dbeta, this->doutput);

    GKO_ASSERT_MTX_NEAR(this->output, this->doutput,
                        100 * r<value_type>::value);
}


TEST_F(Cudss, RefactorizeWithUpdatedValuesMatchesRef)
{
    this->initialize_data(gko::matrices::location_ani4_amd_mtx, 1);
    auto scaled_mtx = gko::share(gko::clone(this->ref, this->mtx));
    for (gko::size_type i = 0; i < scaled_mtx->get_num_stored_elements(); ++i) {
        scaled_mtx->get_values()[i] *= 2.0;
    }
    auto d_scaled_mtx = gko::share(gko::clone(this->exec, scaled_mtx));
    auto ref_solver = this->ref_factory->generate(scaled_mtx);
    ref_solver->apply(this->input, this->output);
    auto cudss_solver = this->cudss_factory->generate(this->dmtx);

    cudss_solver->refactorize(d_scaled_mtx);
    cudss_solver->apply(this->dinput, this->doutput);

    GKO_ASSERT_MTX_NEAR(this->output, this->doutput,
                        100 * r<value_type>::value);
}


TEST_F(Cudss, ApplyToStridedSingleRhsMatchesRef)
{
    this->initialize_data(gko::matrices::location_ani4_amd_mtx, 1);
    auto cudss_solver = this->cudss_factory->generate(this->dmtx);
    const auto nrows = this->mtx->get_size()[0];
    auto wide_input = this->gen_mtx(nrows, 3);
    auto wide_output = this->gen_mtx(nrows, 3);
    for (gko::size_type i = 0; i < nrows; ++i) {
        wide_input->at(i, 1) = this->input->at(i, 0);
        wide_output->at(i, 1) = this->output->at(i, 0);
    }
    auto d_wide_input = gko::clone(this->exec, wide_input);
    auto d_wide_output = gko::clone(this->exec, wide_output);
    auto strided_input =
        d_wide_input->create_submatrix(gko::span{0, nrows}, gko::span{1, 2});
    auto strided_output =
        d_wide_output->create_submatrix(gko::span{0, nrows}, gko::span{1, 2});
    const auto input_stride_before = strided_input->get_stride();
    const auto output_stride_before = strided_output->get_stride();
    auto wide_input_before = gko::clone(d_wide_input);
    auto wide_output_before = gko::clone(d_wide_output);

    cudss_solver->apply(strided_input, strided_output);

    ASSERT_EQ(strided_input->get_stride(), input_stride_before);
    ASSERT_EQ(strided_output->get_stride(), output_stride_before);
    GKO_ASSERT_MTX_NEAR(d_wide_input, wide_input_before, 0);
    GKO_ASSERT_MTX_NEAR(
        d_wide_output->create_submatrix(gko::span{0, nrows}, gko::span{0, 1}),
        wide_output_before->create_submatrix(gko::span{0, nrows},
                                             gko::span{0, 1}),
        0);
    GKO_ASSERT_MTX_NEAR(
        d_wide_output->create_submatrix(gko::span{0, nrows}, gko::span{2, 3}),
        wide_output_before->create_submatrix(gko::span{0, nrows},
                                             gko::span{2, 3}),
        0);
    GKO_ASSERT_MTX_NEAR(strided_output, this->solution,
                        100 * r<value_type>::value);
}


TEST_F(Cudss, ApplyToStridedMultipleRhsMatchesRef)
{
    this->initialize_data(gko::matrices::location_ani4_amd_mtx, 3);
    auto cudss_solver = this->cudss_factory->generate(this->dmtx);
    const auto nrows = this->mtx->get_size()[0];
    auto wide_input = this->gen_mtx(nrows, 6);
    auto wide_output = this->gen_mtx(nrows, 6);
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
    const auto input_stride_before = strided_input->get_stride();
    const auto output_stride_before = strided_output->get_stride();
    auto wide_input_before = gko::clone(d_wide_input);
    auto wide_output_before = gko::clone(d_wide_output);

    cudss_solver->apply(strided_input, strided_output);

    ASSERT_EQ(strided_input->get_stride(), input_stride_before);
    ASSERT_EQ(strided_output->get_stride(), output_stride_before);
    GKO_ASSERT_MTX_NEAR(d_wide_input, wide_input_before, 0);
    GKO_ASSERT_MTX_NEAR(
        d_wide_output->create_submatrix(gko::span{0, nrows}, gko::span{0, 1}),
        wide_output_before->create_submatrix(gko::span{0, nrows},
                                             gko::span{0, 1}),
        0);
    GKO_ASSERT_MTX_NEAR(
        d_wide_output->create_submatrix(gko::span{0, nrows}, gko::span{4, 6}),
        wide_output_before->create_submatrix(gko::span{0, nrows},
                                             gko::span{4, 6}),
        0);
    GKO_ASSERT_MTX_NEAR(strided_output, this->solution,
                        100 * r<value_type>::value);
}


TEST_F(Cudss, ParseConfigCreatesCorrectFactory)
{
    auto config_map = CudssSolver::get_config_map();
    auto reg = gko::config::registry{config_map};
    gko::config::pnode::map_type conf_map;
    conf_map["type"] = gko::config::pnode{"ext::cuda::solver::Cudss"};
    conf_map["matrix_type"] = gko::config::pnode{3};
    conf_map["matrix_view"] = gko::config::pnode{2};
    conf_map["reordering_alg"] = gko::config::pnode{1};
    auto conf = gko::config::pnode{conf_map};

    auto params = CudssSolver::parse(conf, reg);

    ASSERT_EQ(params.matrix_type, 3);
    ASSERT_EQ(params.matrix_view, 2);
    ASSERT_EQ(params.reordering_alg, 1);
    ASSERT_EQ(params.hybrid_execute, false);
    ASSERT_EQ(params.hybrid_memory, false);
}


TEST_F(Cudss, ParseConfigThrowsOnUnknownKey)
{
    auto config_map = CudssSolver::get_config_map();
    auto reg = gko::config::registry{config_map};
    gko::config::pnode::map_type conf_map;
    conf_map["type"] = gko::config::pnode{"ext::cuda::solver::Cudss"};
    conf_map["invalid_key"] = gko::config::pnode{42};
    auto conf = gko::config::pnode{conf_map};

    ASSERT_THROW(CudssSolver::parse(conf, reg), gko::InvalidStateError);
}
