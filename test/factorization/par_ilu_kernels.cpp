// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/par_ilu_kernels.hpp"


#include <algorithm>
#include <fstream>
#include <memory>
#include <random>
#include <string>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/factorization/factorization_kernels.hpp"
#include "core/test/utils.hpp"
#include "matrices/config.hpp"
#include "test/utils/executor.hpp"


template <typename ValueIndexType>
class ParIlu : public CommonTestFixture {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Dense = gko::matrix::Dense<value_type>;
    using Coo = gko::matrix::Coo<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;

    std::default_random_engine rand_engine;
    std::shared_ptr<const Csr> mtx;
    std::shared_ptr<const Csr> dmtx;

    ParIlu() : rand_engine(18)
    {
        std::string file_name(gko::matrices::location_ani4_mtx);
        auto input_file = std::ifstream(file_name, std::ios::in);
        auto mtx_temp = gko::read<Csr>(input_file, ref);
        // Make sure there are diagonal elements present
        gko::kernels::reference::factorization::add_diagonal_elements(
            ref, mtx_temp.get(), false);
        auto dmtx_temp = gko::clone(exec, mtx_temp);
        mtx = gko::give(mtx_temp);
        dmtx = gko::give(dmtx_temp);
    }

    template <typename Mtx>
    std::unique_ptr<Mtx> gen_mtx(index_type num_rows, index_type num_cols)
    {
        return gko::test::generate_random_matrix<Mtx>(
            num_rows, num_cols,
            std::uniform_int_distribution<index_type>(0, num_cols - 1),
            std::normal_distribution<gko::remove_complex<value_type>>(0.0, 1.0),
            rand_engine, ref);
    }

    std::unique_ptr<Csr> gen_unsorted_mtx(index_type num_rows,
                                          index_type num_cols)
    {
        using std::swap;
        auto mtx = gen_mtx<Csr>(num_rows, num_cols);
        auto values = mtx->get_values();
        auto col_idxs = mtx->get_col_idxs();
        const auto row_ptrs = mtx->get_const_row_ptrs();
        for (int row = 0; row < num_rows; ++row) {
            const auto row_start = row_ptrs[row];
            const auto row_end = row_ptrs[row + 1];
            const int num_row_elements = row_end - row_start;
            auto idx_dist = std::uniform_int_distribution<index_type>(
                row_start, row_end - 1);
            for (int i = 0; i < num_row_elements / 2; ++i) {
                auto idx1 = idx_dist(rand_engine);
                auto idx2 = idx_dist(rand_engine);
                if (idx1 != idx2) {
                    swap(values[idx1], values[idx2]);
                    swap(col_idxs[idx1], col_idxs[idx2]);
                }
            }
        }
        return mtx;
    }

    void initialize_row_ptrs(index_type* l_row_ptrs, index_type* u_row_ptrs,
                             index_type* dl_row_ptrs, index_type* du_row_ptrs)
    {
        gko::kernels::reference::factorization::initialize_row_ptrs_l_u(
            ref, mtx.get(), l_row_ptrs, u_row_ptrs);
        gko::kernels::EXEC_NAMESPACE::factorization::initialize_row_ptrs_l_u(
            exec, dmtx.get(), dl_row_ptrs, du_row_ptrs);
    }

    void initialize_lu(std::unique_ptr<Csr>& l, std::unique_ptr<Csr>& u,
                       std::unique_ptr<Csr>& dl, std::unique_ptr<Csr>& du)
    {
        auto num_row_ptrs = mtx->get_size()[0] + 1;
        gko::array<index_type> l_row_ptrs{ref, num_row_ptrs};
        gko::array<index_type> u_row_ptrs{ref, num_row_ptrs};
        gko::array<index_type> dl_row_ptrs{exec, num_row_ptrs};
        gko::array<index_type> du_row_ptrs{exec, num_row_ptrs};

        initialize_row_ptrs(l_row_ptrs.get_data(), u_row_ptrs.get_data(),
                            dl_row_ptrs.get_data(), du_row_ptrs.get_data());
        // Since `initialize_row_ptrs` was already tested, it is expected that
        // `*` and `d*` contain identical values
        auto l_nnz = l_row_ptrs.get_const_data()[num_row_ptrs - 1];
        auto u_nnz = u_row_ptrs.get_const_data()[num_row_ptrs - 1];

        l = Csr::create(ref, mtx->get_size(), l_nnz);
        u = Csr::create(ref, mtx->get_size(), u_nnz);
        dl = Csr::create(exec, dmtx->get_size(), l_nnz);
        du = Csr::create(exec, dmtx->get_size(), u_nnz);
        // Copy the already initialized `row_ptrs` to the new matrices
        ref->copy(num_row_ptrs, l_row_ptrs.get_data(), l->get_row_ptrs());
        ref->copy(num_row_ptrs, u_row_ptrs.get_data(), u->get_row_ptrs());
        exec->copy(num_row_ptrs, dl_row_ptrs.get_data(), dl->get_row_ptrs());
        exec->copy(num_row_ptrs, du_row_ptrs.get_data(), du->get_row_ptrs());

        gko::kernels::reference::factorization::initialize_l_u(
            ref, mtx.get(), l.get(), u.get());
        gko::kernels::EXEC_NAMESPACE::factorization::initialize_l_u(
            exec, dmtx.get(), dl.get(), du.get());
    }

    void compute_lu(std::unique_ptr<Csr>& l, std::unique_ptr<Csr>& u,
                    std::unique_ptr<Csr>& dl, std::unique_ptr<Csr>& du,
                    gko::size_type iterations = 0)
    {
        auto coo = Coo::create(ref);
        mtx->convert_to(coo);
        auto dcoo = Coo::create(exec);
        dmtx->convert_to(dcoo);
        initialize_lu(l, u, dl, du);
        auto u_transpose_mtx = gko::as<Csr>(u->transpose());
        auto u_transpose_dmtx = gko::as<Csr>(du->transpose());

        gko::kernels::reference::par_ilu_factorization::compute_l_u_factors(
            ref, iterations, coo.get(), l.get(), u_transpose_mtx.get());
        gko::kernels::EXEC_NAMESPACE::par_ilu_factorization::
            compute_l_u_factors(exec, iterations, dcoo.get(), dl.get(),
                                u_transpose_dmtx.get());
        auto u_lin_op = u_transpose_mtx->transpose();
        u = gko::as<Csr>(std::move(u_lin_op));
        auto du_lin_op = u_transpose_dmtx->transpose();
        du = gko::as<Csr>(std::move(du_lin_op));
    }
};

TYPED_TEST_SUITE(ParIlu, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(ParIlu, KernelAddDiagonalElementsSortedEquivalentToRef)
{
    using Csr = typename TestFixture::Csr;
    auto mtx = this->template gen_mtx<Csr>(600, 600);
    auto dmtx = gko::clone(this->exec, mtx);

    gko::kernels::reference::factorization::add_diagonal_elements(
        this->ref, mtx.get(), true);
    gko::kernels::EXEC_NAMESPACE::factorization::add_diagonal_elements(
        this->exec, dmtx.get(), true);

    ASSERT_TRUE(mtx->is_sorted_by_column_index());
    GKO_ASSERT_MTX_NEAR(mtx, dmtx, 0.);
    GKO_ASSERT_MTX_EQ_SPARSITY(mtx, dmtx);
}


TYPED_TEST(ParIlu, KernelAddDiagonalElementsUnsortedEquivalentToRef)
{
    auto mtx = this->gen_unsorted_mtx(600, 600);
    auto dmtx = gko::clone(this->exec, mtx);

    gko::kernels::reference::factorization::add_diagonal_elements(
        this->ref, mtx.get(), false);
    gko::kernels::EXEC_NAMESPACE::factorization::add_diagonal_elements(
        this->exec, dmtx.get(), false);

    ASSERT_FALSE(mtx->is_sorted_by_column_index());
    GKO_ASSERT_MTX_NEAR(mtx, dmtx, 0.);
    GKO_ASSERT_MTX_EQ_SPARSITY(mtx, dmtx);
}


TYPED_TEST(ParIlu, KernelAddDiagonalElementsNonSquareEquivalentToRef)
{
    using Csr = typename TestFixture::Csr;
    auto mtx = this->template gen_mtx<Csr>(600, 500);
    auto dmtx = gko::clone(this->exec, mtx);

    gko::kernels::reference::factorization::add_diagonal_elements(
        this->ref, mtx.get(), true);
    gko::kernels::EXEC_NAMESPACE::factorization::add_diagonal_elements(
        this->exec, dmtx.get(), true);

    ASSERT_TRUE(mtx->is_sorted_by_column_index());
    GKO_ASSERT_MTX_NEAR(mtx, dmtx, 0.);
    GKO_ASSERT_MTX_EQ_SPARSITY(mtx, dmtx);
}


TYPED_TEST(ParIlu, KernelInitializeRowPtrsLUEquivalentToRef)
{
    using index_type = typename TestFixture::index_type;
    auto num_row_ptrs = this->mtx->get_size()[0] + 1;
    gko::array<index_type> l_row_ptrs_array(this->ref, num_row_ptrs);
    gko::array<index_type> u_row_ptrs_array(this->ref, num_row_ptrs);
    gko::array<index_type> dl_row_ptrs_array(this->exec, num_row_ptrs);
    gko::array<index_type> du_row_ptrs_array(this->exec, num_row_ptrs);

    this->initialize_row_ptrs(
        l_row_ptrs_array.get_data(), u_row_ptrs_array.get_data(),
        dl_row_ptrs_array.get_data(), du_row_ptrs_array.get_data());

    GKO_ASSERT_ARRAY_EQ(l_row_ptrs_array, dl_row_ptrs_array);
    GKO_ASSERT_ARRAY_EQ(u_row_ptrs_array, du_row_ptrs_array);
}


TYPED_TEST(ParIlu, KernelInitializeParILUIsEquivalentToRef)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    std::unique_ptr<Csr> l_mtx{};
    std::unique_ptr<Csr> u_mtx{};
    std::unique_ptr<Csr> dl_mtx{};
    std::unique_ptr<Csr> du_mtx{};

    this->initialize_lu(l_mtx, u_mtx, dl_mtx, du_mtx);

    GKO_ASSERT_MTX_NEAR(l_mtx, dl_mtx, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(u_mtx, du_mtx, r<value_type>::value);
    GKO_ASSERT_MTX_EQ_SPARSITY(l_mtx, dl_mtx);
    GKO_ASSERT_MTX_EQ_SPARSITY(u_mtx, du_mtx);
}


TYPED_TEST(ParIlu, KernelComputeParILUIsEquivalentToRef)
{
    using Csr = typename TestFixture::Csr;
    std::unique_ptr<Csr> l_mtx{};
    std::unique_ptr<Csr> u_mtx{};
    std::unique_ptr<Csr> dl_mtx{};
    std::unique_ptr<Csr> du_mtx{};

    this->compute_lu(l_mtx, u_mtx, dl_mtx, du_mtx);

    GKO_ASSERT_MTX_NEAR(l_mtx, dl_mtx, 5e-2);
    GKO_ASSERT_MTX_NEAR(u_mtx, du_mtx, 5e-2);
    GKO_ASSERT_MTX_EQ_SPARSITY(l_mtx, dl_mtx);
    GKO_ASSERT_MTX_EQ_SPARSITY(u_mtx, du_mtx);
}


TYPED_TEST(ParIlu, KernelComputeParILUWithMoreIterationsIsEquivalentToRef)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    std::unique_ptr<Csr> l_mtx{};
    std::unique_ptr<Csr> u_mtx{};
    std::unique_ptr<Csr> dl_mtx{};
    std::unique_ptr<Csr> du_mtx{};
    gko::size_type iterations{200};

    this->compute_lu(l_mtx, u_mtx, dl_mtx, du_mtx, iterations);

    GKO_ASSERT_MTX_NEAR(l_mtx, dl_mtx, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(u_mtx, du_mtx, r<value_type>::value);
    GKO_ASSERT_MTX_EQ_SPARSITY(l_mtx, dl_mtx);
    GKO_ASSERT_MTX_EQ_SPARSITY(u_mtx, du_mtx);
}
