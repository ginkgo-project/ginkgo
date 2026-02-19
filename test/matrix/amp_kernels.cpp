// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/amp_kernels.hpp"

#include <random>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/amp.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/ell.hpp>

#include "core/matrix/amp_helpers.hpp"
#include "core/test/utils.hpp"
#include "test/utils/common_fixture.hpp"


class Amp : public CommonTestFixture {
protected:
    using Mtx = gko::matrix::Ell<value_type>;
    using Vec = gko::matrix::Dense<value_type>;

    Amp() : rand_engine(42) {}

    std::unique_ptr<Mtx> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<Mtx>(
            num_rows, num_cols, std::uniform_int_distribution<>(1, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    std::default_random_engine rand_engine;
};


TEST_F(Amp, GenerateEllScatterBinsIsEquivalentToRef)
{
    SKIP_IF_SINGLE_MODE;
    using T = value_type;
    using IndexType = index_type;
    constexpr int q = gko::matrix::AMP<T, IndexType>::num_precisions;
    const float tol = 1e-10;
    auto mtx = gen_mtx(532, 231);
    auto dmtx = gko::clone(exec, mtx);

    // Compute max_nnz per bin using reference kernel
    gko::amp::precision_array<int, T> max_nnz;
    gko::array<gko::remove_complex<T>> rownorms(ref, mtx->get_size()[0]);
    gko::kernels::reference::amp::generate_ell_rownorms_storage(
        ref, mtx.get(), tol, max_nnz, rownorms);

    // Allocate bins on ref and exec with the same max_nnz
    auto ref_bins =
        gko::amp::allocate_bins<T, IndexType>(ref, mtx->get_size(), max_nnz);
    auto exec_bins =
        gko::amp::allocate_bins<T, IndexType>(exec, dmtx->get_size(), max_nnz);
    constexpr auto num_bins = std::tuple_size<decltype(ref_bins)>::value;
    static_assert(num_bins == q, "Wrong number of bins!");

    gko::amp::precision_array<gko::LinOp*, T> ref_amat;
    gko::amp::precision_array<gko::LinOp*, T> exec_amat;
    for (int k = 0; k < num_bins; k++) {
        ref_amat[k] = ref_bins[k].get();
        exec_amat[k] = exec_bins[k].get();
    }

    // Run kernel on ref and exec
    gko::kernels::reference::amp::generate_ell_scatter_bins(ref, mtx.get(), tol,
                                                            ref_amat);
    gko::kernels::GKO_DEVICE_NAMESPACE::amp::generate_ell_scatter_bins(
        exec, dmtx.get(), tol, exec_amat);

    // Compare each bin
    using types_list = typename gko::amp::narrow_types<T>::type;
    gko::constexpr_for<0, num_bins, 1>([&](auto k) {
        using vt = typename std::tuple_element<k, types_list>::type;
        auto ref_ell =
            dynamic_cast<gko::matrix::Ell<vt, IndexType>*>(ref_amat[k]);
        auto exec_ell =
            dynamic_cast<gko::matrix::Ell<vt, IndexType>*>(exec_amat[k]);
        ASSERT_TRUE(ref_ell);
        ASSERT_TRUE(exec_ell);
        GKO_ASSERT_MTX_NEAR(ref_ell, exec_ell, 0);
    });
}
