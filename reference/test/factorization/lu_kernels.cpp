/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <ginkgo/core/factorization/lu.hpp>


#include <algorithm>
#include <fstream>
#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/factorization/factorization.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/components/prefix_sum_kernels.hpp"
#include "core/factorization/cholesky_kernels.hpp"
#include "core/factorization/elimination_forest.hpp"
#include "core/factorization/lu_kernels.hpp"
#include "core/factorization/symbolic.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/matrix/csr_lookup.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"
#include "matrices/config.hpp"


template <typename ValueIndexType>
class Lu : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using matrix_type = gko::matrix::Csr<value_type, index_type>;

    Lu()
        : ref(gko::ReferenceExecutor::create()),
          storage_offsets{ref},
          storage{ref},
          row_descs{ref}
    {}

    void setup(const char* mtx_filename, const char* mtx_lu_filename)
    {
        std::ifstream s_mtx{mtx_filename};
        std::ifstream s_mtx_lu{mtx_lu_filename};
        mtx = gko::read<matrix_type>(s_mtx, ref);
        mtx_lu = gko::read<matrix_type>(s_mtx_lu, ref);
        num_rows = mtx->get_size()[0];
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
    }

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    gko::size_type num_rows;
    std::shared_ptr<matrix_type> mtx;
    std::unique_ptr<matrix_type> mtx_lu;
    gko::array<index_type> storage_offsets;
    gko::array<gko::int32> storage;
    gko::array<gko::int64> row_descs;
};

TYPED_TEST_SUITE(Lu, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(Lu, SymbolicCholeskyWorks)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    this->setup(gko::matrices::location_ani1_mtx,
                gko::matrices::location_ani1_lu_mtx);

    std::unique_ptr<gko::matrix::Csr<value_type, index_type>> lu;
    gko::factorization::symbolic_cholesky(this->mtx.get(), lu);

    GKO_ASSERT_MTX_EQ_SPARSITY(lu, this->mtx_lu);
}


TYPED_TEST(Lu, SymbolicLUWorks)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    this->setup(gko::matrices::location_ani1_nonsymm_mtx,
                gko::matrices::location_ani1_nonsymm_lu_mtx);

    std::unique_ptr<gko::matrix::Csr<value_type, index_type>> lu;
    gko::factorization::symbolic_lu(this->mtx.get(), lu);

    GKO_ASSERT_MTX_EQ_SPARSITY(lu, this->mtx_lu);
}


TYPED_TEST(Lu, SymbolicLUWorksWithMissingDiagonal)
{
    using matrix_type = typename TestFixture::matrix_type;
    auto mtx = gko::initialize<matrix_type>({{1, 1, 1, 0, 0, 0},
                                             {1, 0, 1, 0, 0, 0},
                                             {1, 1, 1, 1, 0, 0},
                                             {0, 0, 1, 1, 1, 0},
                                             {0, 0, 0, 1, 0, 1},
                                             {0, 0, 0, 0, 1, 0}},
                                            this->ref);
    auto expected = gko::initialize<matrix_type>({{1, 1, 1, 0, 0, 0},
                                                  {1, 1, 1, 0, 0, 0},
                                                  {1, 1, 1, 1, 0, 0},
                                                  {0, 0, 1, 1, 1, 0},
                                                  {0, 0, 0, 1, 1, 1},
                                                  {0, 0, 0, 0, 1, 1}},
                                                 this->ref);


    std::unique_ptr<matrix_type> lu;
    gko::factorization::symbolic_lu(mtx.get(), lu);

    GKO_ASSERT_MTX_EQ_SPARSITY(lu, expected);
}


TYPED_TEST(Lu, KernelInitializeWorks)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    this->setup(gko::matrices::location_ani1_mtx,
                gko::matrices::location_ani1_lu_mtx);
    std::fill_n(this->mtx_lu->get_values(),
                this->mtx_lu->get_num_stored_elements(),
                gko::zero<value_type>());
    gko::array<index_type> diag_idxs{this->ref, this->num_rows};

    gko::kernels::reference::lu_factorization::initialize(
        this->ref, this->mtx.get(), this->storage_offsets.get_const_data(),
        this->row_descs.get_const_data(), this->storage.get_const_data(),
        diag_idxs.get_data(), this->mtx_lu.get());

    GKO_ASSERT_MTX_NEAR(this->mtx, this->mtx_lu, 0.0);
    for (gko::size_type row = 0; row < this->num_rows; row++) {
        const auto diag_pos = diag_idxs.get_const_data()[row];
        ASSERT_GE(diag_pos, this->mtx_lu->get_const_row_ptrs()[row]);
        ASSERT_LT(diag_pos, this->mtx_lu->get_const_row_ptrs()[row + 1]);
        ASSERT_EQ(this->mtx_lu->get_const_col_idxs()[diag_pos], row);
    }
}


TYPED_TEST(Lu, KernelInitializeAmdWorks)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    this->setup(gko::matrices::location_ani1_amd_mtx,
                gko::matrices::location_ani1_amd_lu_mtx);
    std::fill_n(this->mtx_lu->get_values(),
                this->mtx_lu->get_num_stored_elements(),
                gko::zero<value_type>());
    gko::array<index_type> diag_idxs{this->ref, this->num_rows};

    gko::kernels::reference::lu_factorization::initialize(
        this->ref, this->mtx.get(), this->storage_offsets.get_const_data(),
        this->row_descs.get_const_data(), this->storage.get_const_data(),
        diag_idxs.get_data(), this->mtx_lu.get());

    GKO_ASSERT_MTX_NEAR(this->mtx, this->mtx_lu, 0.0);
    for (gko::size_type row = 0; row < this->num_rows; row++) {
        const auto diag_pos = diag_idxs.get_const_data()[row];
        ASSERT_GE(diag_pos, this->mtx_lu->get_const_row_ptrs()[row]);
        ASSERT_LT(diag_pos, this->mtx_lu->get_const_row_ptrs()[row + 1]);
        ASSERT_EQ(this->mtx_lu->get_const_col_idxs()[diag_pos], row);
    }
}


TYPED_TEST(Lu, KernelFactorizeWorks)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    this->setup(gko::matrices::location_ani1_mtx,
                gko::matrices::location_ani1_lu_mtx);
    const auto mtx_lu_ref = this->mtx_lu->clone();
    std::fill_n(this->mtx_lu->get_values(),
                this->mtx_lu->get_num_stored_elements(),
                gko::zero<value_type>());
    gko::array<index_type> diag_idxs{this->ref, this->num_rows};
    gko::array<int> tmp{this->ref};
    gko::kernels::reference::lu_factorization::initialize(
        this->ref, this->mtx.get(), this->storage_offsets.get_const_data(),
        this->row_descs.get_const_data(), this->storage.get_const_data(),
        diag_idxs.get_data(), this->mtx_lu.get());

    gko::kernels::reference::lu_factorization::factorize(
        this->ref, this->storage_offsets.get_const_data(),
        this->row_descs.get_const_data(), this->storage.get_const_data(),
        diag_idxs.get_const_data(), this->mtx_lu.get(), tmp);

    GKO_ASSERT_MTX_NEAR(this->mtx_lu, mtx_lu_ref, r<value_type>::value);
}


TYPED_TEST(Lu, FactorizeWorks)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    this->setup(gko::matrices::location_ani1_mtx,
                gko::matrices::location_ani1_lu_mtx);
    auto factory =
        gko::experimental::factorization::Lu<value_type, index_type>::build()
            .with_symmetric_sparsity(true)
            .on(this->ref);

    auto lu = factory->generate(this->mtx);

    GKO_ASSERT_MTX_NEAR(lu->get_combined(), this->mtx_lu, r<value_type>::value);
    ASSERT_EQ(lu->get_storage_type(),
              gko::experimental::factorization::storage_type::combined_lu);
    ASSERT_EQ(lu->get_lower_factor(), nullptr);
    ASSERT_EQ(lu->get_upper_factor(), nullptr);
    ASSERT_EQ(lu->get_diagonal(), nullptr);
}


TYPED_TEST(Lu, FactorizeNonsymmetricWorks)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    this->setup(gko::matrices::location_ani1_nonsymm_mtx,
                gko::matrices::location_ani1_nonsymm_lu_mtx);
    auto factory =
        gko::experimental::factorization::Lu<value_type, index_type>::build()
            .with_symmetric_sparsity(false)
            .on(this->ref);

    auto lu = factory->generate(this->mtx);

    GKO_ASSERT_MTX_EQ_SPARSITY(lu->get_combined(), this->mtx_lu);
    GKO_ASSERT_MTX_NEAR(lu->get_combined(), this->mtx_lu,
                        r<value_type>::value * 10);
    ASSERT_EQ(lu->get_storage_type(),
              gko::experimental::factorization::storage_type::combined_lu);
    ASSERT_EQ(lu->get_lower_factor(), nullptr);
    ASSERT_EQ(lu->get_upper_factor(), nullptr);
    ASSERT_EQ(lu->get_diagonal(), nullptr);
}


TYPED_TEST(Lu, KernelFactorizeAmdWorks)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    this->setup(gko::matrices::location_ani1_amd_mtx,
                gko::matrices::location_ani1_amd_lu_mtx);
    const auto mtx_lu_ref = this->mtx_lu->clone();
    std::fill_n(this->mtx_lu->get_values(),
                this->mtx_lu->get_num_stored_elements(),
                gko::zero<value_type>());
    gko::array<index_type> diag_idxs{this->ref, this->num_rows};
    gko::array<int> tmp{this->ref};
    gko::kernels::reference::lu_factorization::initialize(
        this->ref, this->mtx.get(), this->storage_offsets.get_const_data(),
        this->row_descs.get_const_data(), this->storage.get_const_data(),
        diag_idxs.get_data(), this->mtx_lu.get());

    gko::kernels::reference::lu_factorization::factorize(
        this->ref, this->storage_offsets.get_const_data(),
        this->row_descs.get_const_data(), this->storage.get_const_data(),
        diag_idxs.get_const_data(), this->mtx_lu.get(), tmp);

    GKO_ASSERT_MTX_NEAR(this->mtx_lu, mtx_lu_ref, r<value_type>::value);
}


TYPED_TEST(Lu, FactorizeAmdWorks)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    this->setup(gko::matrices::location_ani1_amd_mtx,
                gko::matrices::location_ani1_amd_lu_mtx);
    auto pattern = gko::share(
        gko::matrix::SparsityCsr<value_type, index_type>::create(this->ref));
    pattern->copy_from(this->mtx_lu.get());
    auto factory =
        gko::experimental::factorization::Lu<value_type, index_type>::build()
            .with_symbolic_factorization(pattern)
            .on(this->ref);

    auto lu = factory->generate(this->mtx);

    GKO_ASSERT_MTX_NEAR(lu->get_combined(), this->mtx_lu, r<value_type>::value);
    ASSERT_EQ(lu->get_storage_type(),
              gko::experimental::factorization::storage_type::combined_lu);
    ASSERT_EQ(lu->get_lower_factor(), nullptr);
    ASSERT_EQ(lu->get_upper_factor(), nullptr);
    ASSERT_EQ(lu->get_diagonal(), nullptr);
}
