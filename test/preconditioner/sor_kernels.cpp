// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/preconditioner/sor_kernels.hpp"

#include <random>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/test/utils.hpp"
#include "core/utils/matrix_utils.hpp"
#include "test/utils/common_fixture.hpp"
#include "test/utils/executor.hpp"


class Sor : public CommonTestFixture {
protected:
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using Dense = gko::matrix::Dense<value_type>;

    Sor()
    {
        gko::size_type n = 133;
        index_type row_limit = 15;
        auto nz_dist = std::uniform_int_distribution<index_type>(1, row_limit);
        auto val_dist = std::uniform_real_distribution<value_type>(-1., 1.);
        auto md =
            gko::test::generate_random_matrix_data<value_type, index_type>(
                n, n, nz_dist, val_dist, rand_engine);
        auto md_l = md;
        auto md_u = md;
        // make_upper/lower_triangular also removes the diagonal, so it is
        // added back with make_unit_diagonal
        gko::utils::make_lower_triangular(md_l);
        gko::utils::make_unit_diagonal(md_l);
        gko::utils::make_upper_triangular(md_u);
        gko::utils::make_unit_diagonal(md_u);

        mtx->read(md);
        d_mtx->read(md);

        result_l->read(md_l);
        result_l->scale(gko::initialize<Dense>({0.0}, ref));
        d_result_l = gko::clone(exec, result_l);

        result_u->read(md_u);
        result_u->scale(gko::initialize<Dense>({0.0}, ref));
        d_result_u = gko::clone(exec, result_u);
    }

    std::default_random_engine rand_engine{42};

    std::unique_ptr<Csr> mtx = Csr::create(ref);
    std::unique_ptr<Csr> d_mtx = Csr::create(exec);

    std::unique_ptr<Csr> result_l = Csr::create(ref);
    std::unique_ptr<Csr> d_result_l = Csr::create(exec);
    std::unique_ptr<Csr> result_u = Csr::create(ref);
    std::unique_ptr<Csr> d_result_u = Csr::create(exec);
};


TEST_F(Sor, InitializeWeightedLFactorIsSameAsReference)
{
    gko::kernels::reference::sor::initialize_weighted_l(ref, mtx.get(), 1.24,
                                                        result_l.get());
    gko::kernels::GKO_DEVICE_NAMESPACE::sor::initialize_weighted_l(
        exec, d_mtx.get(), 1.24, d_result_l.get());

    GKO_ASSERT_MTX_NEAR(result_l, d_result_l, r<value_type>::value);
}


TEST_F(Sor, InitializeWeightedLAndUFactorIsSameAsReference)
{
    gko::kernels::reference::sor::initialize_weighted_l_u(
        ref, mtx.get(), 1.24, result_l.get(), result_u.get());
    gko::kernels::GKO_DEVICE_NAMESPACE::sor::initialize_weighted_l_u(
        exec, d_mtx.get(), 1.24, d_result_l.get(), d_result_u.get());

    GKO_ASSERT_MTX_NEAR(result_l, d_result_l, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(result_u, d_result_u, r<value_type>::value);
}
