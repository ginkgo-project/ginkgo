// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/lu_kernels.hpp"


#include <algorithm>
#include <fstream>
#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/factorization/lu.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/components/fill_array_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/factorization/cholesky_kernels.hpp"
#include "core/factorization/elimination_forest.hpp"
#include "core/factorization/symbolic.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/matrix/csr_lookup.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"
#include "matrices/config.hpp"
#include "test/utils/executor.hpp"


template <typename ValueIndexType>
class Lu : public CommonTestFixture {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using factory_type =
        gko::experimental::factorization::Lu<value_type, index_type>;
    using matrix_type = typename factory_type::matrix_type;
    using sparsity_pattern_type = typename factory_type::sparsity_pattern_type;

    Lu()
        : storage_offsets{ref},
          dstorage_offsets{exec},
          storage{ref},
          dstorage{exec},
          row_descs{ref},
          drow_descs{exec}
    {}

    void initialize_data(const char* mtx_filename, const char* mtx_lu_filename)
    {
        std::ifstream s_mtx{mtx_filename};
        mtx = gko::read<matrix_type>(s_mtx, ref);
        dmtx = gko::clone(exec, mtx);
        num_rows = mtx->get_size()[0];
        std::ifstream s_mtx_lu{mtx_lu_filename};
        mtx_lu = gko::read<matrix_type>(s_mtx_lu, ref);
        storage_offsets.resize_and_reset(num_rows + 1);
        row_descs.resize_and_reset(num_rows);

        const auto allowed = gko::matrix::csr::sparsity_type::bitmap |
                             gko::matrix::csr::sparsity_type::full |
                             gko::matrix::csr::sparsity_type::hash;
        gko::kernels::reference::csr::build_lookup_offsets(
            ref, mtx_lu->get_const_row_ptrs(), mtx_lu->get_const_col_idxs(),
            num_rows, allowed, storage_offsets.get_data());
        storage.resize_and_reset(storage_offsets.get_const_data()[num_rows]);
        gko::kernels::reference::csr::build_lookup(
            ref, mtx_lu->get_const_row_ptrs(), mtx_lu->get_const_col_idxs(),
            num_rows, allowed, storage_offsets.get_const_data(),
            row_descs.get_data(), storage.get_data());
        dstorage_offsets = storage_offsets;
        dstorage = storage;
        drow_descs = row_descs;
        dmtx_lu = gko::clone(exec, mtx_lu);
        mtx_lu_sparsity = sparsity_pattern_type::create(ref);
        mtx_lu_sparsity->copy_from(mtx_lu);
        dmtx_lu_sparsity = sparsity_pattern_type::create(exec);
        dmtx_lu_sparsity->copy_from(mtx_lu_sparsity);
    }

    void forall_matrices(std::function<void()> fn)
    {
        {
            SCOPED_TRACE("ani1");
            this->initialize_data(gko::matrices::location_ani1_mtx,
                                  gko::matrices::location_ani1_lu_mtx);
            fn();
        }
        {
            SCOPED_TRACE("ani1_amd");
            this->initialize_data(gko::matrices::location_ani1_amd_mtx,
                                  gko::matrices::location_ani1_amd_lu_mtx);
            fn();
        }
        {
            SCOPED_TRACE("ani4");
            this->initialize_data(gko::matrices::location_ani4_mtx,
                                  gko::matrices::location_ani4_lu_mtx);
            fn();
        }
        {
            SCOPED_TRACE("ani4_amd");
            this->initialize_data(gko::matrices::location_ani4_amd_mtx,
                                  gko::matrices::location_ani4_amd_lu_mtx);
            fn();
        }
    }

    gko::size_type num_rows;
    std::shared_ptr<matrix_type> mtx;
    std::shared_ptr<matrix_type> mtx_lu;
    std::shared_ptr<sparsity_pattern_type> mtx_lu_sparsity;
    std::shared_ptr<matrix_type> dmtx;
    std::shared_ptr<matrix_type> dmtx_lu;
    std::shared_ptr<sparsity_pattern_type> dmtx_lu_sparsity;
    gko::array<index_type> storage_offsets;
    gko::array<index_type> dstorage_offsets;
    gko::array<gko::int32> storage;
    gko::array<gko::int32> dstorage;
    gko::array<gko::int64> row_descs;
    gko::array<gko::int64> drow_descs;
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

TYPED_TEST_SUITE(Lu, Types, PairTypenameNameGenerator);


TYPED_TEST(Lu, KernelInitializeIsEquivalentToRef)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    this->forall_matrices([this] {
        std::fill_n(this->mtx_lu->get_values(),
                    this->mtx_lu->get_num_stored_elements(),
                    gko::zero<value_type>());
        gko::kernels::EXEC_NAMESPACE::components::fill_array(
            this->exec, this->dmtx_lu->get_values(),
            this->dmtx_lu->get_num_stored_elements(), gko::zero<value_type>());
        gko::array<index_type> diag_idxs{this->ref, this->num_rows};
        gko::array<index_type> ddiag_idxs{this->exec, this->num_rows};

        gko::kernels::reference::lu_factorization::initialize(
            this->ref, this->mtx.get(), this->storage_offsets.get_const_data(),
            this->row_descs.get_const_data(), this->storage.get_const_data(),
            diag_idxs.get_data(), this->mtx_lu.get());
        gko::kernels::EXEC_NAMESPACE::lu_factorization::initialize(
            this->exec, this->dmtx.get(),
            this->dstorage_offsets.get_const_data(),
            this->drow_descs.get_const_data(), this->dstorage.get_const_data(),
            ddiag_idxs.get_data(), this->dmtx_lu.get());

        GKO_ASSERT_MTX_NEAR(this->dmtx_lu, this->dmtx_lu, 0.0);
        GKO_ASSERT_ARRAY_EQ(diag_idxs, ddiag_idxs);
    });
}


TYPED_TEST(Lu, KernelFactorizeIsEquivalentToRef)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    this->forall_matrices([this] {
        gko::array<index_type> diag_idxs{this->ref, this->num_rows};
        gko::array<index_type> ddiag_idxs{this->exec, this->num_rows};
        gko::array<int> tmp{this->ref};
        gko::array<int> dtmp{this->exec};
        gko::kernels::reference::lu_factorization::initialize(
            this->ref, this->mtx.get(), this->storage_offsets.get_const_data(),
            this->row_descs.get_const_data(), this->storage.get_const_data(),
            diag_idxs.get_data(), this->mtx_lu.get());
        gko::kernels::EXEC_NAMESPACE::lu_factorization::initialize(
            this->exec, this->dmtx.get(),
            this->dstorage_offsets.get_const_data(),
            this->drow_descs.get_const_data(), this->dstorage.get_const_data(),
            ddiag_idxs.get_data(), this->dmtx_lu.get());

        gko::kernels::reference::lu_factorization::factorize(
            this->ref, this->storage_offsets.get_const_data(),
            this->row_descs.get_const_data(), this->storage.get_const_data(),
            diag_idxs.get_const_data(), this->mtx_lu.get(), tmp);
        gko::kernels::EXEC_NAMESPACE::lu_factorization::factorize(
            this->exec, this->dstorage_offsets.get_const_data(),
            this->drow_descs.get_const_data(), this->dstorage.get_const_data(),
            ddiag_idxs.get_const_data(), this->dmtx_lu.get(), dtmp);

        GKO_ASSERT_MTX_NEAR(this->mtx_lu, this->dmtx_lu, r<value_type>::value);
    });
}


TYPED_TEST(Lu, SymbolicCholeskyWorks)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    this->forall_matrices([this] {
        std::unique_ptr<gko::matrix::Csr<value_type, index_type>> dlu;
        std::unique_ptr<gko::factorization::elimination_forest<index_type>>
            forest;
        gko::factorization::symbolic_cholesky(this->dmtx.get(), true, dlu,
                                              forest);

        GKO_ASSERT_MTX_EQ_SPARSITY(dlu, this->dmtx_lu);
    });
}


TYPED_TEST(Lu, SymbolicLUWorks)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    this->forall_matrices([this] {
        std::unique_ptr<gko::matrix::Csr<value_type, index_type>> dlu;
        gko::factorization::symbolic_lu(this->dmtx.get(), dlu);

        GKO_ASSERT_MTX_EQ_SPARSITY(dlu, this->dmtx_lu);
    });
}


TYPED_TEST(Lu, SymbolicLUNearSymmWorks)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    this->forall_matrices([this] {
        std::unique_ptr<gko::matrix::Csr<value_type, index_type>> dlu;
        gko::factorization::symbolic_lu_near_symm(this->dmtx.get(), dlu);

        GKO_ASSERT_MTX_EQ_SPARSITY(dlu, this->dmtx_lu);
    });
}


TYPED_TEST(Lu, GenerateSymmWithUnknownSparsityIsEquivalentToRef)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    this->forall_matrices([this] {
        auto factory =
            gko::experimental::factorization::Lu<value_type,
                                                 index_type>::build()
                .with_symbolic_algorithm(
                    gko::experimental::factorization::symbolic_type::symmetric)
                .on(this->ref);
        auto dfactory =
            gko::experimental::factorization::Lu<value_type,
                                                 index_type>::build()
                .with_symbolic_algorithm(
                    gko::experimental::factorization::symbolic_type::symmetric)
                .on(this->exec);

        auto lu = factory->generate(this->mtx);
        auto dlu = dfactory->generate(this->dmtx);

        GKO_ASSERT_MTX_EQ_SPARSITY(lu->get_combined(), dlu->get_combined());
        GKO_ASSERT_MTX_NEAR(lu->get_combined(), dlu->get_combined(),
                            r<value_type>::value);
    });
}


TYPED_TEST(Lu, GenerateNearSymmWithUnknownSparsityIsEquivalentToRef)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    this->forall_matrices([this] {
        auto factory =
            gko::experimental::factorization::Lu<value_type,
                                                 index_type>::build()
                .with_symbolic_algorithm(gko::experimental::factorization::
                                             symbolic_type::near_symmetric)
                .on(this->ref);
        auto dfactory =
            gko::experimental::factorization::Lu<value_type,
                                                 index_type>::build()
                .with_symbolic_algorithm(gko::experimental::factorization::
                                             symbolic_type::near_symmetric)
                .on(this->exec);

        auto lu = factory->generate(this->mtx);
        auto dlu = dfactory->generate(this->dmtx);

        GKO_ASSERT_MTX_EQ_SPARSITY(lu->get_combined(), dlu->get_combined());
        GKO_ASSERT_MTX_NEAR(lu->get_combined(), dlu->get_combined(),
                            r<value_type>::value);
    });
}


TYPED_TEST(Lu, GenerateWithKnownSparsityIsEquivalentToRef)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    this->forall_matrices([this] {
        auto factory = gko::experimental::factorization::Lu<value_type,
                                                            index_type>::build()
                           .with_symbolic_factorization(this->mtx_lu_sparsity)
                           .on(this->ref);
        auto dfactory =
            gko::experimental::factorization::Lu<value_type,
                                                 index_type>::build()
                .with_symbolic_factorization(this->dmtx_lu_sparsity)
                .on(this->exec);

        auto lu = factory->generate(this->mtx);
        auto dlu = dfactory->generate(this->dmtx);

        GKO_ASSERT_MTX_EQ_SPARSITY(this->dmtx_lu_sparsity, dlu->get_combined());
        GKO_ASSERT_MTX_NEAR(lu->get_combined(), dlu->get_combined(),
                            r<value_type>::value);
    });
}


TYPED_TEST(Lu, GenerateUnsymmWithUnknownSparsityIsEquivalentToRef)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    this->forall_matrices([this] {
        auto factory = gko::experimental::factorization::Lu<value_type,
                                                            index_type>::build()
                           .on(this->ref);
        auto dfactory =
            gko::experimental::factorization::Lu<value_type,
                                                 index_type>::build()
                .on(this->exec);

        auto lu = factory->generate(this->mtx);
        auto dlu = dfactory->generate(this->dmtx);

        GKO_ASSERT_MTX_EQ_SPARSITY(lu->get_combined(), dlu->get_combined());
        GKO_ASSERT_MTX_NEAR(lu->get_combined(), dlu->get_combined(),
                            r<value_type>::value);
    });
}
