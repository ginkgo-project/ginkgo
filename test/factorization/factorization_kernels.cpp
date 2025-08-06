// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/factorization_kernels.hpp"

#include <fstream>
#include <memory>
#include <random>
#include <string>

#include <gtest/gtest.h>

#include <core/utils/matrix_utils.hpp>

#include <ginkgo/core/base/executor.hpp>

#include "core/test/utils.hpp"
#include "core/test/utils/unsort_matrix.hpp"
#include "test/utils/common_fixture.hpp"


class Factorization : public CommonTestFixture {
protected:
    using Csr = gko::matrix::Csr<value_type, index_type>;

    Factorization()
    {
        mtx = gko::test::generate_random_matrix<Csr>(
            52, 52, std::uniform_int_distribution<>(4, 40),
            std::uniform_real_distribution<>(1, 2), rand_engine, ref);
        gko::utils::ensure_all_diagonal_entries(mtx.get());
        dmtx = gko::clone(exec, mtx);
    }

    std::default_random_engine rand_engine{6794};
    std::shared_ptr<Csr> mtx;
    std::shared_ptr<Csr> dmtx;
};


TEST_F(Factorization, InitializeRowPtrsLSameAsRef)
{
    gko::array<index_type> l_ptrs{ref, mtx->get_size()[0] + 1};
    gko::array<index_type> dl_ptrs{exec, mtx->get_size()[0] + 1};

    gko::kernels::reference::factorization::initialize_row_ptrs_l(
        ref, mtx.get(), l_ptrs.get_data());
    gko::kernels::GKO_DEVICE_NAMESPACE::factorization::initialize_row_ptrs_l(
        exec, dmtx.get(), dl_ptrs.get_data());

    GKO_ASSERT_ARRAY_EQ(l_ptrs, dl_ptrs);
}


TEST_F(Factorization, InitializeLSameAsRef)
{
    gko::array<index_type> l_ptrs{ref, mtx->get_size()[0] + 1};
    gko::kernels::reference::factorization::initialize_row_ptrs_l(
        ref, mtx.get(), l_ptrs.get_data());
    auto nnz =
        static_cast<gko::size_type>(l_ptrs.get_data()[mtx->get_size()[0]]);
    auto l_mtx =
        Csr::create(ref, mtx->get_size(), gko::array<value_type>(ref, nnz),
                    gko::array<index_type>(ref, nnz), l_ptrs);
    auto dl_mtx = gko::clone(exec, l_mtx);

    for (auto diag_sqrt : {false, true}) {
        SCOPED_TRACE("diag_sqrt: " + std::to_string(diag_sqrt));

        gko::kernels::reference::factorization::initialize_l(
            ref, mtx.get(), l_mtx.get(), diag_sqrt);
        gko::kernels::GKO_DEVICE_NAMESPACE::factorization::initialize_l(
            exec, dmtx.get(), dl_mtx.get(), diag_sqrt);

        GKO_ASSERT_MTX_NEAR(l_mtx, dl_mtx,
                            diag_sqrt ? r<value_type>::value : 0.0);
    }
}
