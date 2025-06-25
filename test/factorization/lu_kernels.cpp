// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/lu_kernels.hpp"

#include <algorithm>
#include <fstream>
#include <memory>

#include <gtest/gtest.h>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/base/mtx_io.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/factorization/lu.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>

#include "core/base/index_range.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/factorization/cholesky_kernels.hpp"
#include "core/factorization/elimination_forest.hpp"
#include "core/factorization/factorization_kernels.hpp"
#include "core/factorization/symbolic.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/matrix/csr_lookup.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"
#include "matrices/config.hpp"
#include "test/utils/common_fixture.hpp"


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

    Lu() : lookup{ref}, dlookup{exec} {}

    void initialize_data(const char* mtx_filename, const char* mtx_lu_filename)
    {
        std::ifstream s_mtx{mtx_filename};
        mtx = gko::read<matrix_type>(s_mtx, ref);
        dmtx = gko::clone(exec, mtx);
        num_rows = mtx->get_size()[0];
        std::ifstream s_mtx_lu{mtx_lu_filename};
        mtx_lu = gko::read<matrix_type>(s_mtx_lu, ref);
        lookup = gko::matrix::csr::build_lookup(mtx_lu.get());
        dlookup = lookup;
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
#ifndef GINKGO_FAST_TESTS
            SCOPED_TRACE("ani4");
            this->initialize_data(gko::matrices::location_ani4_mtx,
                                  gko::matrices::location_ani4_lu_mtx);
            fn();
#endif
        }
        {
#ifndef GINKGO_FAST_TESTS
            SCOPED_TRACE("ani4_amd");
            this->initialize_data(gko::matrices::location_ani4_amd_mtx,
                                  gko::matrices::location_ani4_amd_lu_mtx);
            fn();
#endif
        }
    }

    gko::size_type num_rows;
    std::shared_ptr<matrix_type> mtx;
    std::shared_ptr<matrix_type> mtx_lu;
    std::shared_ptr<sparsity_pattern_type> mtx_lu_sparsity;
    std::shared_ptr<matrix_type> dmtx;
    std::shared_ptr<matrix_type> dmtx_lu;
    std::shared_ptr<sparsity_pattern_type> dmtx_lu_sparsity;
    gko::matrix::csr::lookup_data<index_type> lookup;
    gko::matrix::csr::lookup_data<index_type> dlookup;
};

#ifdef GKO_COMPILING_OMP
using Types = gko::test::ValueIndexTypes;
#elif defined(GKO_COMPILING_CUDA)
// CUDA don't support long indices for sorting, and the triangular solvers
// seem broken
using Types = gko::test::cartesian_type_product_t<gko::test::ValueTypes,
                                                  ::testing::Types<gko::int32>>;
#else
// HIP only supports real types and int32
using Types = gko::test::cartesian_type_product_t<gko::test::RealValueTypesBase,
                                                  ::testing::Types<gko::int32>>;
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
        gko::kernels::GKO_DEVICE_NAMESPACE::components::fill_array(
            this->exec, this->dmtx_lu->get_values(),
            this->dmtx_lu->get_num_stored_elements(), gko::zero<value_type>());
        gko::array<index_type> diag_idxs{this->ref, this->num_rows};
        gko::array<index_type> ddiag_idxs{this->exec, this->num_rows};

        gko::kernels::reference::lu_factorization::initialize(
            this->ref, this->mtx.get(),
            this->lookup.storage_offsets.get_const_data(),
            this->lookup.row_descs.get_const_data(),
            this->lookup.storage.get_const_data(), diag_idxs.get_data(),
            this->mtx_lu.get());
        gko::kernels::GKO_DEVICE_NAMESPACE::lu_factorization::initialize(
            this->exec, this->dmtx.get(),
            this->dlookup.storage_offsets.get_const_data(),
            this->dlookup.row_descs.get_const_data(),
            this->dlookup.storage.get_const_data(), ddiag_idxs.get_data(),
            this->dmtx_lu.get());

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
            this->ref, this->mtx.get(),
            this->lookup.storage_offsets.get_const_data(),
            this->lookup.row_descs.get_const_data(),
            this->lookup.storage.get_const_data(), diag_idxs.get_data(),
            this->mtx_lu.get());
        gko::kernels::GKO_DEVICE_NAMESPACE::lu_factorization::initialize(
            this->exec, this->dmtx.get(),
            this->dlookup.storage_offsets.get_const_data(),
            this->dlookup.row_descs.get_const_data(),
            this->dlookup.storage.get_const_data(), ddiag_idxs.get_data(),
            this->dmtx_lu.get());

        gko::kernels::reference::lu_factorization::factorize(
            this->ref, this->lookup.storage_offsets.get_const_data(),
            this->lookup.row_descs.get_const_data(),
            this->lookup.storage.get_const_data(), diag_idxs.get_const_data(),
            this->mtx_lu.get(), true, tmp);
        gko::kernels::GKO_DEVICE_NAMESPACE::lu_factorization::factorize(
            this->exec, this->dlookup.storage_offsets.get_const_data(),
            this->dlookup.row_descs.get_const_data(),
            this->dlookup.storage.get_const_data(), ddiag_idxs.get_const_data(),
            this->dmtx_lu.get(), true, dtmp);

        GKO_ASSERT_MTX_NEAR(this->mtx_lu, this->dmtx_lu, r<value_type>::value);
    });
}


TYPED_TEST(Lu, KernelValidateValidFactors)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    this->forall_matrices([this] {
        bool valid = false;

        gko::kernels::GKO_DEVICE_NAMESPACE::factorization::symbolic_validate(
            this->exec, this->dmtx.get(), this->dmtx_lu.get(),
            gko::matrix::csr::build_lookup(this->dmtx_lu.get()), valid);

        ASSERT_TRUE(valid);
    });
}


TYPED_TEST(Lu, KernelValidateInvalidFactorsIdentity)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    this->forall_matrices([this] {
        bool valid = true;
        gko::matrix_data<value_type, index_type> data(
            this->dmtx_lu->get_size());
        // an identity matrix is a valid factorization, but doesn't contain the
        // system matrix
        for (auto row : gko::irange{static_cast<index_type>(data.size[0])}) {
            data.nonzeros.emplace_back(row, row, gko::one<value_type>());
        }
        this->dmtx_lu->read(data);

        gko::kernels::GKO_DEVICE_NAMESPACE::factorization::symbolic_validate(
            this->exec, this->dmtx.get(), this->dmtx_lu.get(),
            gko::matrix::csr::build_lookup(this->dmtx_lu.get()), valid);

        ASSERT_FALSE(valid);
    });
}


TYPED_TEST(Lu, KernelValidateInvalidFactorsMissing)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    this->forall_matrices([this] {
        bool valid = true;
        gko::matrix_data<value_type, index_type> data;
        this->dmtx_lu->write(data);
        // delete a random entry somewhere in the middle of the matrix
        data.nonzeros.erase(data.nonzeros.begin() +
                            data.nonzeros.size() * 3 / 4);
        this->dmtx_lu->read(data);

        gko::kernels::GKO_DEVICE_NAMESPACE::factorization::symbolic_validate(
            this->exec, this->dmtx.get(), this->dmtx_lu.get(),
            gko::matrix::csr::build_lookup(this->dmtx_lu.get()), valid);

        ASSERT_FALSE(valid);
    });
}


TYPED_TEST(Lu, KernelValidateInvalidFactorsExtra)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    this->forall_matrices([this] {
        bool valid = true;
        gko::matrix_data<value_type, index_type> data;
        this->dmtx_lu->write(data);
        // insert an entry between two non-adjacent values in a row somewhere
        // not at the beginning
        const auto it = std::adjacent_find(
            data.nonzeros.begin() + data.nonzeros.size() / 5,
            data.nonzeros.end(), [](auto a, auto b) {
                return a.row == b.row && a.column < b.column - 1;
            });
        data.nonzeros.insert(it, {it->row, it->column + 1, it->value});
        this->dmtx_lu->read(data);

        gko::kernels::GKO_DEVICE_NAMESPACE::factorization::symbolic_validate(
            this->exec, this->dmtx.get(), this->dmtx_lu.get(),
            gko::matrix::csr::build_lookup(this->dmtx_lu.get()), valid);

        ASSERT_FALSE(valid);
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
