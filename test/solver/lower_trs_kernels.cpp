// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/triangular.hpp>


#include "core/test/utils.hpp"
#include "core/utils/matrix_utils.hpp"
#include "test/utils/executor.hpp"


class LowerTrs : public CommonTestFixture {
protected:
    using mtx_type = gko::matrix::Csr<value_type, index_type>;
    using vec_type = gko::matrix::Dense<>;
    using solver_type = gko::solver::LowerTrs<value_type, index_type>;

    LowerTrs() : rand_engine(30) {}

    std::unique_ptr<vec_type> gen_vec(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<vec_type>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    std::unique_ptr<mtx_type> gen_l_mtx(int size, int row_nnz)
    {
        return gko::test::generate_random_lower_triangular_matrix<mtx_type>(
            size, false, std::uniform_int_distribution<>(row_nnz, size),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    std::unique_ptr<mtx_type> gen_mtx(int size, int row_nnz)
    {
        auto data =
            gko::test::generate_random_matrix_data<value_type, index_type>(
                size, size, std::uniform_int_distribution<>(row_nnz, size),
                std::normal_distribution<>(-1.0, 1.0), rand_engine);
        gko::utils::make_diag_dominant(data);
        auto result = mtx_type::create(ref);
        result->read(data);
        return result;
    }

    void initialize_data(int m, int n, int row_nnz)
    {
        b = gen_vec(m, n);
        x = gen_vec(m, n);
        mtx = gen_mtx(m, row_nnz);
        mtx_l = gen_l_mtx(m, row_nnz);
        dx = gko::clone(exec, x);
        db = gko::clone(exec, b);
        dmtx = gko::clone(exec, mtx);
        dmtx_l = gko::clone(exec, mtx_l);
    }

    std::shared_ptr<vec_type> b;
    std::shared_ptr<vec_type> x;
    std::shared_ptr<mtx_type> mtx;
    std::shared_ptr<mtx_type> mtx_l;
    std::shared_ptr<vec_type> db;
    std::shared_ptr<vec_type> dx;
    std::shared_ptr<mtx_type> dmtx;
    std::shared_ptr<mtx_type> dmtx_l;
    std::default_random_engine rand_engine;
};


TEST_F(LowerTrs, ApplyFullDenseMtxIsEquivalentToRef)
{
    initialize_data(50, 1, 50);
    auto lower_trs_factory = solver_type::build().on(ref);
    auto d_lower_trs_factory = solver_type::build().on(exec);
    auto solver = lower_trs_factory->generate(mtx);
    auto d_solver = d_lower_trs_factory->generate(dmtx);

    solver->apply(b, x);
    d_solver->apply(db, dx);

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(LowerTrs, ApplyFullDenseMtxUnitDiagIsEquivalentToRef)
{
    initialize_data(50, 1, 50);
    auto lower_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(ref);
    auto d_lower_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(exec);
    auto solver = lower_trs_factory->generate(mtx);
    auto d_solver = d_lower_trs_factory->generate(dmtx);

    solver->apply(b, x);
    d_solver->apply(db, dx);

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(LowerTrs, ApplyFullSparseMtxIsEquivalentToRef)
{
    initialize_data(50, 1, 5);
    auto lower_trs_factory = solver_type::build().on(ref);
    auto d_lower_trs_factory = solver_type::build().on(exec);
    auto solver = lower_trs_factory->generate(mtx);
    auto d_solver = d_lower_trs_factory->generate(dmtx);

    solver->apply(b, x);
    d_solver->apply(db, dx);

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(LowerTrs, ApplyFullSparseMtxUnitDiagIsEquivalentToRef)
{
    initialize_data(50, 1, 5);
    auto lower_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(ref);
    auto d_lower_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(exec);
    auto solver = lower_trs_factory->generate(mtx);
    auto d_solver = d_lower_trs_factory->generate(dmtx);

    solver->apply(b, x);
    d_solver->apply(db, dx);

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(LowerTrs, ApplyTriangularDenseMtxIsEquivalentToRef)
{
    initialize_data(50, 1, 50);
    auto lower_trs_factory = solver_type::build().on(ref);
    auto d_lower_trs_factory = solver_type::build().on(exec);
    auto solver = lower_trs_factory->generate(mtx_l);
    auto d_solver = d_lower_trs_factory->generate(dmtx_l);

    solver->apply(b, x);
    d_solver->apply(db, dx);

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(LowerTrs, ApplyTriangularDenseMtxUnitDiagIsEquivalentToRef)
{
    initialize_data(50, 1, 50);
    auto lower_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(ref);
    auto d_lower_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(exec);
    auto solver = lower_trs_factory->generate(mtx_l);
    auto d_solver = d_lower_trs_factory->generate(dmtx_l);

    solver->apply(b, x);
    d_solver->apply(db, dx);

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(LowerTrs, ApplyTriangularSparseMtxIsEquivalentToRef)
{
    initialize_data(50, 1, 5);
    auto lower_trs_factory = solver_type::build().on(ref);
    auto d_lower_trs_factory = solver_type::build().on(exec);
    auto solver = lower_trs_factory->generate(mtx_l);
    auto d_solver = d_lower_trs_factory->generate(dmtx_l);

    solver->apply(b, x);
    d_solver->apply(db, dx);

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(LowerTrs, ApplyTriangularSparseMtxUnitDiagIsEquivalentToRef)
{
    initialize_data(50, 1, 5);
    auto lower_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(ref);
    auto d_lower_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(exec);
    auto solver = lower_trs_factory->generate(mtx_l);
    auto d_solver = d_lower_trs_factory->generate(dmtx_l);

    solver->apply(b, x);
    d_solver->apply(db, dx);

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(LowerTrs, ApplyFullDenseMtxMultipleRhsIsEquivalentToRef)
{
    initialize_data(50, 4, 50);
    auto lower_trs_factory = solver_type::build().with_num_rhs(4u).on(ref);
    auto d_lower_trs_factory = solver_type::build().with_num_rhs(4u).on(exec);
    auto solver = lower_trs_factory->generate(mtx);
    auto d_solver = d_lower_trs_factory->generate(dmtx);

    solver->apply(b, x);
    d_solver->apply(db, dx);

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(LowerTrs, ApplyFullDenseMtxUnitDiagMultipleRhsIsEquivalentToRef)
{
    initialize_data(50, 5, 50);
    auto lower_trs_factory =
        solver_type::build().with_num_rhs(5u).with_unit_diagonal(true).on(ref);
    auto d_lower_trs_factory =
        solver_type::build().with_num_rhs(5u).with_unit_diagonal(true).on(exec);
    auto solver = lower_trs_factory->generate(mtx);
    auto d_solver = d_lower_trs_factory->generate(dmtx);

    solver->apply(b, x);
    d_solver->apply(db, dx);

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(LowerTrs, ApplyFullSparseMtxMultipleRhsIsEquivalentToRef)
{
    initialize_data(50, 6, 5);
    auto lower_trs_factory = solver_type::build().with_num_rhs(6u).on(ref);
    auto d_lower_trs_factory = solver_type::build().with_num_rhs(6u).on(exec);
    auto solver = lower_trs_factory->generate(mtx);
    auto d_solver = d_lower_trs_factory->generate(dmtx);

    solver->apply(b, x);
    d_solver->apply(db, dx);

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(LowerTrs, ApplyFullSparseMtxUnitDiagMultipleRhsIsEquivalentToRef)
{
    initialize_data(50, 7, 5);
    auto lower_trs_factory =
        solver_type::build().with_num_rhs(7u).with_unit_diagonal(true).on(ref);
    auto d_lower_trs_factory =
        solver_type::build().with_num_rhs(7u).with_unit_diagonal(true).on(exec);
    auto solver = lower_trs_factory->generate(mtx);
    auto d_solver = d_lower_trs_factory->generate(dmtx);

    solver->apply(b, x);
    d_solver->apply(db, dx);

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(LowerTrs, ApplyTriangularDenseMtxMultipleRhsIsEquivalentToRef)
{
    initialize_data(50, 8, 50);
    auto lower_trs_factory = solver_type::build().with_num_rhs(8u).on(ref);
    auto d_lower_trs_factory = solver_type::build().with_num_rhs(8u).on(exec);
    auto solver = lower_trs_factory->generate(mtx_l);
    auto d_solver = d_lower_trs_factory->generate(dmtx_l);

    solver->apply(b, x);
    d_solver->apply(db, dx);

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(LowerTrs, ApplyTriangularDenseMtxUnitDiagMultipleRhsIsEquivalentToRef)
{
    initialize_data(50, 9, 50);
    auto lower_trs_factory =
        solver_type::build().with_num_rhs(9u).with_unit_diagonal(true).on(ref);
    auto d_lower_trs_factory =
        solver_type::build().with_num_rhs(9u).with_unit_diagonal(true).on(exec);
    auto solver = lower_trs_factory->generate(mtx_l);
    auto d_solver = d_lower_trs_factory->generate(dmtx_l);

    solver->apply(b, x);
    d_solver->apply(db, dx);

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(LowerTrs, ApplyTriangularSparseMtxMultipleRhsIsEquivalentToRef)
{
    initialize_data(50, 10, 5);
    auto lower_trs_factory = solver_type::build().with_num_rhs(10u).on(ref);
    auto d_lower_trs_factory = solver_type::build().with_num_rhs(10u).on(exec);
    auto solver = lower_trs_factory->generate(mtx_l);
    auto d_solver = d_lower_trs_factory->generate(dmtx_l);

    solver->apply(b, x);
    d_solver->apply(db, dx);

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(LowerTrs, ApplyTriangularSparseMtxUnitDiagMultipleRhsIsEquivalentToRef)
{
    initialize_data(50, 11, 5);
    auto lower_trs_factory =
        solver_type::build().with_num_rhs(11u).with_unit_diagonal(true).on(ref);
    auto d_lower_trs_factory =
        solver_type::build().with_num_rhs(11u).with_unit_diagonal(true).on(
            exec);
    auto solver = lower_trs_factory->generate(mtx_l);
    auto d_solver = d_lower_trs_factory->generate(dmtx_l);

    solver->apply(b, x);
    d_solver->apply(db, dx);

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


#ifdef GKO_COMPILING_CUDA


TEST_F(LowerTrs, ClassicalApplyFullDenseMtxIsEquivalentToRef)
{
    initialize_data(50, 1, 50);
    dmtx->set_strategy(std::make_shared<mtx_type::classical>());
    auto lower_trs_factory = solver_type::build().on(ref);
    auto d_lower_trs_factory = solver_type::build().on(exec);
    auto solver = lower_trs_factory->generate(mtx);
    auto d_solver = d_lower_trs_factory->generate(dmtx);

    solver->apply(b, x);
    d_solver->apply(db, dx);

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(LowerTrs, ClassicalApplyFullDenseMtxUnitDiagIsEquivalentToRef)
{
    initialize_data(50, 1, 50);
    dmtx->set_strategy(std::make_shared<mtx_type::classical>());
    auto lower_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(ref);
    auto d_lower_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(exec);
    auto solver = lower_trs_factory->generate(mtx);
    auto d_solver = d_lower_trs_factory->generate(dmtx);

    solver->apply(b, x);
    d_solver->apply(db, dx);

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(LowerTrs, ClassicalApplyFullSparseMtxIsEquivalentToRef)
{
    initialize_data(50, 1, 5);
    dmtx->set_strategy(std::make_shared<mtx_type::classical>());
    auto lower_trs_factory = solver_type::build().on(ref);
    auto d_lower_trs_factory = solver_type::build().on(exec);
    auto solver = lower_trs_factory->generate(mtx);
    auto d_solver = d_lower_trs_factory->generate(dmtx);

    solver->apply(b, x);
    d_solver->apply(db, dx);

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(LowerTrs, ClassicalApplyFullSparseMtxUnitDiagIsEquivalentToRef)
{
    initialize_data(50, 1, 5);
    dmtx->set_strategy(std::make_shared<mtx_type::classical>());
    auto lower_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(ref);
    auto d_lower_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(exec);
    auto solver = lower_trs_factory->generate(mtx);
    auto d_solver = d_lower_trs_factory->generate(dmtx);

    solver->apply(b, x);
    d_solver->apply(db, dx);

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(LowerTrs, ClassicalApplyTriangularDenseMtxIsEquivalentToRef)
{
    initialize_data(50, 1, 50);
    dmtx_l->set_strategy(std::make_shared<mtx_type::classical>());
    auto lower_trs_factory = solver_type::build().on(ref);
    auto d_lower_trs_factory = solver_type::build().on(exec);
    auto solver = lower_trs_factory->generate(mtx_l);
    auto d_solver = d_lower_trs_factory->generate(dmtx_l);

    solver->apply(b, x);
    d_solver->apply(db, dx);

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(LowerTrs, ClassicalApplyTriangularDenseMtxUnitDiagIsEquivalentToRef)
{
    initialize_data(50, 1, 50);
    dmtx_l->set_strategy(std::make_shared<mtx_type::classical>());
    auto lower_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(ref);
    auto d_lower_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(exec);
    auto solver = lower_trs_factory->generate(mtx_l);
    auto d_solver = d_lower_trs_factory->generate(dmtx_l);

    solver->apply(b, x);
    d_solver->apply(db, dx);

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(LowerTrs, ClassicalApplyTriangularSparseMtxIsEquivalentToRef)
{
    initialize_data(50, 1, 5);
    dmtx_l->set_strategy(std::make_shared<mtx_type::classical>());
    auto lower_trs_factory = solver_type::build().on(ref);
    auto d_lower_trs_factory = solver_type::build().on(exec);
    auto solver = lower_trs_factory->generate(mtx_l);
    auto d_solver = d_lower_trs_factory->generate(dmtx_l);

    solver->apply(b, x);
    d_solver->apply(db, dx);

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(LowerTrs, ClassicalApplyTriangularSparseMtxUnitDiagIsEquivalentToRef)
{
    initialize_data(50, 1, 5);
    dmtx_l->set_strategy(std::make_shared<mtx_type::classical>());
    auto lower_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(ref);
    auto d_lower_trs_factory =
        solver_type::build().with_unit_diagonal(true).on(exec);
    auto solver = lower_trs_factory->generate(mtx_l);
    auto d_solver = d_lower_trs_factory->generate(dmtx_l);

    solver->apply(b, x);
    d_solver->apply(db, dx);

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(LowerTrs, ClassicalApplyFullDenseMtxMultipleRhsIsEquivalentToRef)
{
    initialize_data(50, 4, 50);
    dmtx->set_strategy(std::make_shared<mtx_type::classical>());
    auto lower_trs_factory = solver_type::build().with_num_rhs(4u).on(ref);
    auto d_lower_trs_factory = solver_type::build().with_num_rhs(4u).on(exec);
    auto solver = lower_trs_factory->generate(mtx);
    auto d_solver = d_lower_trs_factory->generate(dmtx);

    solver->apply(b, x);
    d_solver->apply(db, dx);

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(LowerTrs, ClassicalApplyFullDenseMtxUnitDiagMultipleRhsIsEquivalentToRef)
{
    initialize_data(50, 5, 50);
    dmtx->set_strategy(std::make_shared<mtx_type::classical>());
    auto lower_trs_factory =
        solver_type::build().with_num_rhs(5u).with_unit_diagonal(true).on(ref);
    auto d_lower_trs_factory =
        solver_type::build().with_num_rhs(5u).with_unit_diagonal(true).on(exec);
    auto solver = lower_trs_factory->generate(mtx);
    auto d_solver = d_lower_trs_factory->generate(dmtx);

    solver->apply(b, x);
    d_solver->apply(db, dx);

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(LowerTrs, ClassicalApplyFullSparseMtxMultipleRhsIsEquivalentToRef)
{
    initialize_data(50, 6, 5);
    dmtx->set_strategy(std::make_shared<mtx_type::classical>());
    auto lower_trs_factory = solver_type::build().with_num_rhs(6u).on(ref);
    auto d_lower_trs_factory = solver_type::build().with_num_rhs(6u).on(exec);
    auto solver = lower_trs_factory->generate(mtx);
    auto d_solver = d_lower_trs_factory->generate(dmtx);

    solver->apply(b, x);
    d_solver->apply(db, dx);

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(LowerTrs,
       ClassicalApplyFullSparseMtxUnitDiagMultipleRhsIsEquivalentToRef)
{
    initialize_data(50, 7, 5);
    dmtx->set_strategy(std::make_shared<mtx_type::classical>());
    auto lower_trs_factory =
        solver_type::build().with_num_rhs(7u).with_unit_diagonal(true).on(ref);
    auto d_lower_trs_factory =
        solver_type::build().with_num_rhs(7u).with_unit_diagonal(true).on(exec);
    auto solver = lower_trs_factory->generate(mtx);
    auto d_solver = d_lower_trs_factory->generate(dmtx);

    solver->apply(b, x);
    d_solver->apply(db, dx);

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(LowerTrs, ClassicalApplyTriangularDenseMtxMultipleRhsIsEquivalentToRef)
{
    initialize_data(50, 8, 50);
    dmtx_l->set_strategy(std::make_shared<mtx_type::classical>());
    auto lower_trs_factory = solver_type::build().with_num_rhs(8u).on(ref);
    auto d_lower_trs_factory = solver_type::build().with_num_rhs(8u).on(exec);
    auto solver = lower_trs_factory->generate(mtx_l);
    auto d_solver = d_lower_trs_factory->generate(dmtx_l);

    solver->apply(b, x);
    d_solver->apply(db, dx);

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(LowerTrs,
       ClassicalApplyTriangularDenseMtxUnitDiagMultipleRhsIsEquivalentToRef)
{
    initialize_data(50, 9, 50);
    dmtx_l->set_strategy(std::make_shared<mtx_type::classical>());
    auto lower_trs_factory =
        solver_type::build().with_num_rhs(9u).with_unit_diagonal(true).on(ref);
    auto d_lower_trs_factory =
        solver_type::build().with_num_rhs(9u).with_unit_diagonal(true).on(exec);
    auto solver = lower_trs_factory->generate(mtx_l);
    auto d_solver = d_lower_trs_factory->generate(dmtx_l);

    solver->apply(b, x);
    d_solver->apply(db, dx);

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(LowerTrs, ClassicalApplyTriangularSparseMtxMultipleRhsIsEquivalentToRef)
{
    initialize_data(50, 10, 5);
    dmtx_l->set_strategy(std::make_shared<mtx_type::classical>());
    auto lower_trs_factory = solver_type::build().with_num_rhs(10u).on(ref);
    auto d_lower_trs_factory = solver_type::build().with_num_rhs(10u).on(exec);
    auto solver = lower_trs_factory->generate(mtx_l);
    auto d_solver = d_lower_trs_factory->generate(dmtx_l);

    solver->apply(b, x);
    d_solver->apply(db, dx);

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(LowerTrs,
       ClassicalApplyTriangularSparseMtxUnitDiagMultipleRhsIsEquivalentToRef)
{
    initialize_data(50, 11, 5);
    dmtx_l->set_strategy(std::make_shared<mtx_type::classical>());
    auto lower_trs_factory =
        solver_type::build().with_num_rhs(11u).with_unit_diagonal(true).on(ref);
    auto d_lower_trs_factory =
        solver_type::build().with_num_rhs(11u).with_unit_diagonal(true).on(
            exec);
    auto solver = lower_trs_factory->generate(mtx_l);
    auto d_solver = d_lower_trs_factory->generate(dmtx_l);

    solver->apply(b, x);
    d_solver->apply(db, dx);

    GKO_ASSERT_MTX_NEAR(dx, x, 1e-14);
}


#endif
