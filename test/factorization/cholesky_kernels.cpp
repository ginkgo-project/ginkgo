// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/cholesky_kernels.hpp"


#include <algorithm>
#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/factorization/cholesky.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/identity.hpp>


#include "core/components/fill_array_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/factorization/elimination_forest.hpp"
#include "core/factorization/symbolic.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/matrix/csr_lookup.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"
#include "core/utils/matrix_utils.hpp"
#include "matrices/config.hpp"
#include "test/utils/executor.hpp"


namespace {


template <typename ValueIndexType>
class CholeskySymbolic : public CommonTestFixture {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using matrix_type = gko::matrix::Csr<value_type, index_type>;
    using elimination_forest =
        gko::factorization::elimination_forest<index_type>;

    CholeskySymbolic() : tmp{ref}, dtmp{exec}
    {
        matrices.emplace_back(
            "example small",
            gko::initialize<matrix_type>(
                {{1, 0, 1, 0}, {0, 1, 0, 1}, {1, 0, 1, 0}, {0, 0, 0, 1}}, ref));
        matrices.emplace_back("example", gko::initialize<matrix_type>(
                                             {{1, 0, 1, 0, 0, 0, 0, 1, 0, 0},
                                              {0, 1, 0, 1, 0, 0, 0, 0, 0, 1},
                                              {1, 0, 1, 0, 0, 0, 0, 0, 0, 0},
                                              {0, 0, 0, 1, 0, 0, 0, 0, 1, 1},
                                              {0, 1, 0, 0, 1, 0, 0, 0, 1, 1},
                                              {0, 0, 0, 0, 0, 1, 0, 1, 0, 0},
                                              {0, 0, 1, 0, 0, 1, 1, 0, 0, 0},
                                              {1, 0, 0, 0, 0, 1, 0, 1, 1, 1},
                                              {0, 0, 0, 1, 1, 0, 0, 1, 1, 0},
                                              {0, 1, 0, 1, 1, 0, 0, 1, 0, 1}},
                                             ref));
        matrices.emplace_back("separable", gko::initialize<matrix_type>(
                                               {{1, 0, 1, 0, 0, 0, 0, 0, 0, 0},
                                                {0, 1, 1, 0, 0, 0, 0, 0, 0, 0},
                                                {1, 1, 1, 0, 0, 0, 0, 0, 0, 0},
                                                {0, 0, 0, 1, 1, 0, 0, 0, 0, 0},
                                                {0, 0, 0, 1, 1, 1, 0, 0, 0, 1},
                                                {0, 0, 0, 0, 1, 1, 0, 0, 0, 0},
                                                {0, 0, 0, 0, 0, 0, 1, 0, 0, 1},
                                                {0, 0, 0, 0, 0, 0, 1, 1, 0, 0},
                                                {0, 0, 0, 0, 0, 0, 0, 0, 1, 1},
                                                {0, 0, 0, 0, 1, 0, 1, 0, 1, 1}},
                                               ref));
        matrices.emplace_back(
            "missing diagonal",
            gko::initialize<matrix_type>({{1, 0, 1, 0, 0, 0, 0, 0, 0, 0},
                                          {0, 1, 1, 0, 0, 0, 0, 0, 0, 0},
                                          {1, 1, 0, 1, 0, 0, 0, 0, 0, 0},
                                          {0, 0, 1, 1, 1, 0, 0, 0, 0, 0},
                                          {0, 0, 0, 1, 0, 1, 0, 0, 0, 0},
                                          {0, 0, 0, 0, 1, 1, 1, 0, 0, 0},
                                          {0, 0, 0, 0, 0, 1, 1, 1, 0, 1},
                                          {0, 0, 0, 0, 0, 0, 1, 1, 0, 0},
                                          {0, 0, 0, 0, 0, 0, 0, 0, 1, 1},
                                          {0, 0, 0, 0, 0, 0, 1, 0, 1, 0}},
                                         ref));
        std::ifstream ani1_stream{gko::matrices::location_ani1_mtx};
        matrices.emplace_back("ani1", gko::read<matrix_type>(ani1_stream, ref));
        std::ifstream ani1_amd_stream{gko::matrices::location_ani1_amd_mtx};
        matrices.emplace_back("ani1_amd",
                              gko::read<matrix_type>(ani1_amd_stream, ref));
    }

    void assert_equal_forests(elimination_forest& lhs, elimination_forest& rhs,
                              bool check_postorder = false)
    {
        GKO_ASSERT_ARRAY_EQ(lhs.parents, rhs.parents);
        GKO_ASSERT_ARRAY_EQ(lhs.children, rhs.children);
        GKO_ASSERT_ARRAY_EQ(lhs.child_ptrs, rhs.child_ptrs);
        if (check_postorder) {
            GKO_ASSERT_ARRAY_EQ(lhs.postorder, rhs.postorder);
            GKO_ASSERT_ARRAY_EQ(lhs.postorder_parents, rhs.postorder_parents);
            GKO_ASSERT_ARRAY_EQ(lhs.inv_postorder, rhs.inv_postorder);
        }
    }

    std::vector<std::pair<std::string, std::unique_ptr<const matrix_type>>>
        matrices;
    gko::array<index_type> tmp;
    gko::array<index_type> dtmp;
};

#ifdef GKO_COMPILING_OMP
using Types = gko::test::ValueIndexTypes;
#elif defined(GKO_COMPILING_CUDA)
// CUDA doesn't support long indices for sorting, and the triangular solvers
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

TYPED_TEST_SUITE(CholeskySymbolic, Types, PairTypenameNameGenerator);


TYPED_TEST(CholeskySymbolic, KernelSymbolicCount)
{
    using matrix_type = typename TestFixture::matrix_type;
    using index_type = typename TestFixture::index_type;
    using elimination_forest = typename TestFixture::elimination_forest;
    for (const auto& pair : this->matrices) {
        SCOPED_TRACE(pair.first);
        const auto& mtx = pair.second;
        const auto dmtx = gko::clone(this->exec, mtx);
        std::unique_ptr<elimination_forest> forest;
        std::unique_ptr<elimination_forest> dforest;
        gko::factorization::compute_elim_forest(mtx.get(), forest);
        gko::factorization::compute_elim_forest(dmtx.get(), dforest);
        gko::array<index_type> row_nnz{this->ref, mtx->get_size()[0]};
        gko::array<index_type> drow_nnz{this->exec, mtx->get_size()[0]};

        gko::kernels::reference::cholesky::symbolic_count(
            this->ref, mtx.get(), *forest, row_nnz.get_data(), this->tmp);
        gko::kernels::EXEC_NAMESPACE::cholesky::symbolic_count(
            this->exec, dmtx.get(), *dforest, drow_nnz.get_data(), this->dtmp);

        GKO_ASSERT_ARRAY_EQ(drow_nnz, row_nnz);
    }
}


TYPED_TEST(CholeskySymbolic, KernelSymbolicFactorize)
{
    using matrix_type = typename TestFixture::matrix_type;
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    using elimination_forest = typename TestFixture::elimination_forest;
    for (const auto& pair : this->matrices) {
        SCOPED_TRACE(pair.first);
        const auto& mtx = pair.second;
        const auto dmtx = gko::clone(this->exec, mtx);
        const auto num_rows = mtx->get_size()[0];
        std::unique_ptr<elimination_forest> forest;
        gko::factorization::compute_elim_forest(mtx.get(), forest);
        gko::array<index_type> row_ptrs{this->ref, num_rows + 1};
        gko::kernels::reference::cholesky::symbolic_count(
            this->ref, mtx.get(), *forest, row_ptrs.get_data(), this->tmp);
        gko::kernels::reference::components::prefix_sum_nonnegative(
            this->ref, row_ptrs.get_data(), num_rows + 1);
        const auto nnz =
            static_cast<gko::size_type>(row_ptrs.get_const_data()[num_rows]);
        auto l_factor = matrix_type::create(
            this->ref, mtx->get_size(), gko::array<value_type>{this->ref, nnz},
            gko::array<index_type>{this->ref, nnz}, row_ptrs);
        auto dl_factor = matrix_type::create(
            this->exec, mtx->get_size(),
            gko::array<value_type>{this->exec, nnz},
            gko::array<index_type>{this->exec, nnz}, row_ptrs);
        // need to call the device kernels to initialize dtmp
        std::unique_ptr<elimination_forest> dforest;
        gko::factorization::compute_elim_forest(dmtx.get(), dforest);
        gko::array<index_type> dtmp_ptrs{this->exec, num_rows + 1};
        gko::kernels::EXEC_NAMESPACE::cholesky::symbolic_count(
            this->exec, dmtx.get(), *dforest, dtmp_ptrs.get_data(), this->dtmp);

        gko::kernels::reference::cholesky::symbolic_factorize(
            this->ref, mtx.get(), *forest, l_factor.get(), this->tmp);
        gko::kernels::EXEC_NAMESPACE::cholesky::symbolic_factorize(
            this->exec, dmtx.get(), *dforest, dl_factor.get(), this->dtmp);

        GKO_ASSERT_MTX_EQ_SPARSITY(dl_factor, l_factor);
    }
}


TYPED_TEST(CholeskySymbolic, SymbolicFactorize)
{
    using matrix_type = typename TestFixture::matrix_type;
    using elimination_forest = typename TestFixture::elimination_forest;
    for (const auto& pair : this->matrices) {
        SCOPED_TRACE(pair.first);
        const auto& mtx = pair.second;
        const auto dmtx = gko::clone(this->exec, mtx);
        std::unique_ptr<matrix_type> factors;
        std::unique_ptr<matrix_type> dfactors;
        std::unique_ptr<elimination_forest> forest;
        std::unique_ptr<elimination_forest> dforest;
        gko::factorization::symbolic_cholesky(mtx.get(), true, factors, forest);
        gko::factorization::symbolic_cholesky(mtx.get(), true, dfactors,
                                              dforest);

        GKO_ASSERT_MTX_EQ_SPARSITY(dfactors, factors);
        this->assert_equal_forests(*forest, *dforest, true);
    }
}


TYPED_TEST(CholeskySymbolic, KernelForestFromFactorWorks)
{
    using matrix_type = typename TestFixture::matrix_type;
    using index_type = typename TestFixture::index_type;
    using elimination_forest = typename TestFixture::elimination_forest;
    for (const auto& pair : this->matrices) {
        SCOPED_TRACE(pair.first);
        const auto& mtx = pair.second;
        std::unique_ptr<matrix_type> factors;
        std::unique_ptr<elimination_forest> forest;
        gko::factorization::symbolic_cholesky(mtx.get(), true, factors, forest);
        const auto dfactors = gko::clone(this->exec, factors);
        elimination_forest dforest{this->exec,
                                   static_cast<index_type>(mtx->get_size()[0])};

        gko::kernels::EXEC_NAMESPACE::cholesky::forest_from_factor(
            this->exec, dfactors.get(), dforest);

        this->assert_equal_forests(*forest, dforest);
    }
}


template <typename ValueIndexType>
class Cholesky : public CommonTestFixture {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using factory_type =
        gko::experimental::factorization::Cholesky<value_type, index_type>;
    using matrix_type = typename factory_type::matrix_type;
    using sparsity_pattern_type = typename factory_type::sparsity_pattern_type;
    using elimination_forest =
        gko::factorization::elimination_forest<index_type>;

    Cholesky()
        : storage_offsets{ref},
          dstorage_offsets{exec},
          storage{ref},
          dstorage{exec},
          row_descs{ref},
          drow_descs{exec}
    {}

    void initialize_data(const char* mtx_filename,
                         const char* mtx_chol_filename)
    {
        std::ifstream s_mtx{mtx_filename};
        mtx = gko::read<matrix_type>(s_mtx, ref);
        dmtx = gko::clone(exec, mtx);
        num_rows = mtx->get_size()[0];
        std::ifstream s_mtx_chol{mtx_chol_filename};
        auto mtx_chol_data = gko::read_raw<value_type, index_type>(s_mtx_chol);
        auto nnz = mtx_chol_data.nonzeros.size();
        // add missing upper diagonal entries
        // (values not important, only pattern important)
        gko::utils::make_symmetric(mtx_chol_data);
        mtx_chol_data.sort_row_major();
        mtx_chol = matrix_type::create(ref);
        mtx_chol->read(mtx_chol_data);
        storage_offsets.resize_and_reset(num_rows + 1);
        row_descs.resize_and_reset(num_rows);

        const auto allowed = gko::matrix::csr::sparsity_type::bitmap |
                             gko::matrix::csr::sparsity_type::full |
                             gko::matrix::csr::sparsity_type::hash;
        gko::kernels::reference::csr::build_lookup_offsets(
            ref, mtx_chol->get_const_row_ptrs(), mtx_chol->get_const_col_idxs(),
            num_rows, allowed, storage_offsets.get_data());
        storage.resize_and_reset(storage_offsets.get_const_data()[num_rows]);
        gko::kernels::reference::csr::build_lookup(
            ref, mtx_chol->get_const_row_ptrs(), mtx_chol->get_const_col_idxs(),
            num_rows, allowed, storage_offsets.get_const_data(),
            row_descs.get_data(), storage.get_data());
        dstorage_offsets = storage_offsets;
        dstorage = storage;
        drow_descs = row_descs;
        dmtx_chol = gko::clone(exec, mtx_chol);
        mtx_chol_sparsity = sparsity_pattern_type::create(ref);
        mtx_chol_sparsity->copy_from(mtx_chol.get());
        dmtx_chol_sparsity = sparsity_pattern_type::create(exec);
        dmtx_chol_sparsity->copy_from(mtx_chol_sparsity.get());
        gko::factorization::compute_elim_forest(mtx_chol.get(), forest);
        gko::factorization::compute_elim_forest(dmtx_chol.get(), dforest);
    }

    void forall_matrices(std::function<void()> fn)
    {
        {
            SCOPED_TRACE("ani1");
            this->initialize_data(gko::matrices::location_ani1_mtx,
                                  gko::matrices::location_ani1_chol_mtx);
            fn();
        }
        {
            SCOPED_TRACE("ani1_amd");
            this->initialize_data(gko::matrices::location_ani1_amd_mtx,
                                  gko::matrices::location_ani1_amd_chol_mtx);
            fn();
        }
        {
            SCOPED_TRACE("ani4");
            this->initialize_data(gko::matrices::location_ani4_mtx,
                                  gko::matrices::location_ani4_chol_mtx);
            fn();
        }
        {
            SCOPED_TRACE("ani4_amd");
            this->initialize_data(gko::matrices::location_ani4_amd_mtx,
                                  gko::matrices::location_ani4_amd_chol_mtx);
            fn();
        }
    }

    gko::size_type num_rows;
    std::shared_ptr<matrix_type> mtx;
    std::shared_ptr<matrix_type> mtx_chol;
    std::unique_ptr<elimination_forest> forest;
    std::shared_ptr<sparsity_pattern_type> mtx_chol_sparsity;
    std::shared_ptr<matrix_type> dmtx;
    std::shared_ptr<matrix_type> dmtx_chol;
    std::unique_ptr<elimination_forest> dforest;
    std::shared_ptr<sparsity_pattern_type> dmtx_chol_sparsity;
    gko::array<index_type> storage_offsets;
    gko::array<index_type> dstorage_offsets;
    gko::array<gko::int32> storage;
    gko::array<gko::int32> dstorage;
    gko::array<gko::int64> row_descs;
    gko::array<gko::int64> drow_descs;
};

TYPED_TEST_SUITE(Cholesky, Types, PairTypenameNameGenerator);


TYPED_TEST(Cholesky, KernelInitializeIsEquivalentToRef)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    this->forall_matrices([this] {
        const auto nnz = this->mtx_chol->get_num_stored_elements();
        std::fill_n(this->mtx_chol->get_values(), nnz, gko::zero<value_type>());
        gko::kernels::EXEC_NAMESPACE::components::fill_array(
            this->exec, this->dmtx_chol->get_values(), nnz,
            gko::zero<value_type>());
        gko::array<index_type> diag_idxs{this->ref, this->num_rows};
        gko::array<index_type> ddiag_idxs{this->exec, this->num_rows};
        gko::array<index_type> transpose_idxs{this->ref, nnz};
        gko::array<index_type> dtranspose_idxs{this->exec, nnz};

        gko::kernels::reference::cholesky::initialize(
            this->ref, this->mtx.get(), this->storage_offsets.get_const_data(),
            this->row_descs.get_const_data(), this->storage.get_const_data(),
            diag_idxs.get_data(), transpose_idxs.get_data(),
            this->mtx_chol.get());
        gko::kernels::EXEC_NAMESPACE::cholesky::initialize(
            this->exec, this->dmtx.get(),
            this->dstorage_offsets.get_const_data(),
            this->drow_descs.get_const_data(), this->dstorage.get_const_data(),
            ddiag_idxs.get_data(), dtranspose_idxs.get_data(),
            this->dmtx_chol.get());

        GKO_ASSERT_MTX_NEAR(this->dmtx_chol, this->dmtx_chol, 0.0);
        GKO_ASSERT_ARRAY_EQ(diag_idxs, ddiag_idxs);
    });
}


TYPED_TEST(Cholesky, KernelFactorizeIsEquivalentToRef)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    this->forall_matrices([this] {
        const auto nnz = this->mtx_chol->get_num_stored_elements();
        gko::array<index_type> diag_idxs{this->ref, this->num_rows};
        gko::array<index_type> ddiag_idxs{this->exec, this->num_rows};
        gko::array<index_type> transpose_idxs{this->ref, nnz};
        gko::array<index_type> dtranspose_idxs{this->exec, nnz};
        gko::array<int> tmp{this->ref};
        gko::array<int> dtmp{this->exec};
        gko::kernels::reference::cholesky::initialize(
            this->ref, this->mtx.get(), this->storage_offsets.get_const_data(),
            this->row_descs.get_const_data(), this->storage.get_const_data(),
            diag_idxs.get_data(), transpose_idxs.get_data(),
            this->mtx_chol.get());
        gko::kernels::EXEC_NAMESPACE::cholesky::initialize(
            this->exec, this->dmtx.get(),
            this->dstorage_offsets.get_const_data(),
            this->drow_descs.get_const_data(), this->dstorage.get_const_data(),
            ddiag_idxs.get_data(), dtranspose_idxs.get_data(),
            this->dmtx_chol.get());

        gko::kernels::reference::cholesky::factorize(
            this->ref, this->storage_offsets.get_const_data(),
            this->row_descs.get_const_data(), this->storage.get_const_data(),
            diag_idxs.get_const_data(), transpose_idxs.get_const_data(),
            *this->forest, this->mtx_chol.get(), tmp);
        gko::kernels::EXEC_NAMESPACE::cholesky::factorize(
            this->exec, this->dstorage_offsets.get_const_data(),
            this->drow_descs.get_const_data(), this->dstorage.get_const_data(),
            ddiag_idxs.get_const_data(), dtranspose_idxs.get_const_data(),
            *this->dforest, this->dmtx_chol.get(), dtmp);

        GKO_ASSERT_MTX_NEAR(this->mtx_chol, this->dmtx_chol,
                            r<value_type>::value);
    });
}


TYPED_TEST(Cholesky, GenerateWithUnknownSparsityIsEquivalentToRef)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    this->forall_matrices([this] {
        auto factory =
            gko::experimental::factorization::Cholesky<value_type,
                                                       index_type>::build()
                .on(this->ref);
        auto dfactory =
            gko::experimental::factorization::Cholesky<value_type,
                                                       index_type>::build()
                .on(this->exec);

        auto factors = factory->generate(this->mtx);
        auto dfactors = dfactory->generate(this->dmtx);

        GKO_ASSERT_MTX_EQ_SPARSITY(factors->get_combined(),
                                   dfactors->get_combined());
        GKO_ASSERT_MTX_NEAR(factors->get_combined(), dfactors->get_combined(),
                            r<value_type>::value);
    });
}


TYPED_TEST(Cholesky, GenerateWithKnownSparsityIsEquivalentToRef)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    this->forall_matrices([this] {
        auto factory =
            gko::experimental::factorization::Cholesky<value_type,
                                                       index_type>::build()
                .with_symbolic_factorization(this->mtx_chol_sparsity)
                .on(this->ref);
        auto dfactory =
            gko::experimental::factorization::Cholesky<value_type,
                                                       index_type>::build()
                .with_symbolic_factorization(this->dmtx_chol_sparsity)
                .on(this->exec);

        auto factors = factory->generate(this->mtx);
        auto dfactors = dfactory->generate(this->dmtx);

        GKO_ASSERT_MTX_EQ_SPARSITY(this->dmtx_chol_sparsity,
                                   dfactors->get_combined());
        GKO_ASSERT_MTX_NEAR(factors->get_combined(), dfactors->get_combined(),
                            r<value_type>::value);
    });
}


}  // namespace
