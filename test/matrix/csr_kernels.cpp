/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include "core/matrix/csr_kernels.hpp"


#include <algorithm>
#include <numeric>
#include <random>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/components/fill_array_kernels.hpp"
#include "core/test/utils.hpp"
#include "test/utils/executor.hpp"


namespace {


class Csr : public ::testing::Test {
protected:
    using itype = int;
#if GINKGO_COMMON_SINGLE_MODE
    using vtype = float;
#else
    using vtype = double;
#endif
    using Mtx = gko::matrix::Csr<vtype, itype>;
    using Vec = gko::matrix::Dense<vtype>;

    Csr() : rand_engine(15) {}

    void SetUp()
    {
        ref = gko::ReferenceExecutor::create();
        init_executor(ref, exec);
    }

    void TearDown()
    {
        if (exec != nullptr) {
            ASSERT_NO_THROW(exec->synchronize());
        }
    }

    template <typename MtxType>
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<MtxType>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<>(0.0, 1.0), rand_engine, ref);
    }

    void set_up_apply_data()
    {
        x = gen_mtx<Mtx>(40, 25);
        alpha = gko::initialize<Vec>({2.0}, ref);
        dx = Mtx::create(exec);
        dx->copy_from(x.get());
        dalpha = Vec::create(exec);
        dalpha->copy_from(alpha.get());
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::EXEC_TYPE> exec;

    std::default_random_engine rand_engine;

    std::unique_ptr<Mtx> x;
    std::unique_ptr<Vec> alpha;
    std::unique_ptr<Mtx> dx;
    std::unique_ptr<Vec> dalpha;
};


TEST_F(Csr, ScaleIsEquivalentToRef)
{
    set_up_apply_data();

    x->scale(alpha.get());
    dx->scale(dalpha.get());

    GKO_ASSERT_MTX_NEAR(dx, x, r<vtype>::value);
}


TEST_F(Csr, InvScaleIsEquivalentToRef)
{
    set_up_apply_data();

    x->inv_scale(alpha.get());
    dx->inv_scale(dalpha.get());

    GKO_ASSERT_MTX_NEAR(dx, x, r<vtype>::value);
}


template <typename IndexType>
class CsrLookup : public ::testing::Test {
protected:
    using value_type = float;
    using index_type = IndexType;
    using Mtx = gko::matrix::Csr<value_type, index_type>;

    CsrLookup() : rand_engine(15) {}

    void SetUp()
    {
        ref = gko::ReferenceExecutor::create();
        init_executor(ref, exec);
        auto data =
            gko::test::generate_random_matrix_data<value_type, index_type>(
                628, 923, std::uniform_int_distribution<index_type>(10, 300),
                std::normal_distribution<gko::remove_complex<value_type>>(-1.0,
                                                                          1.0),
                rand_engine);
        // create a few empty rows
        data.nonzeros.erase(
            std::remove_if(data.nonzeros.begin(), data.nonzeros.end(),
                           [](auto entry) { return entry.row % 200 == 21; }),
            data.nonzeros.end());
        // insert a full row and a pretty dense row
        for (int i = 0; i < 100; i++) {
            data.nonzeros.emplace_back(221, i + 100, 1.0);
            data.nonzeros.emplace_back(421, i * 3 + 100, 2.0);
        }
        data.ensure_row_major_order();
        // initialize the matrices
        mtx = Mtx::create(ref);
        mtx->read(data);
        dmtx = gko::clone(exec, mtx);
    }

    void TearDown()
    {
        if (exec != nullptr) {
            ASSERT_NO_THROW(exec->synchronize());
        }
    }

    std::default_random_engine rand_engine;
    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::EXEC_TYPE> exec;
    std::unique_ptr<Mtx> mtx;
    std::unique_ptr<Mtx> dmtx;
    index_type invalid_index = gko::invalid_index<index_type>();
};

TYPED_TEST_SUITE(CsrLookup, gko::test::IndexTypes, TypenameNameGenerator);


TYPED_TEST(CsrLookup, BuildLookupWorks)
{
    using index_type = typename TestFixture::index_type;
    using gko::matrix::csr::sparsity_type;
    const auto num_rows = this->mtx->get_size()[0];
    const auto num_cols = this->mtx->get_size()[1];
    gko::array<gko::int64> row_desc_array(this->ref, num_rows);
    gko::array<gko::int64> drow_desc_array(this->exec, num_rows);
    gko::array<index_type> storage_offset_array(this->ref, num_rows + 1);
    gko::array<index_type> dstorage_offset_array(this->exec, num_rows + 1);
    const auto row_descs = row_desc_array.get_data();
    const auto drow_descs = drow_desc_array.get_data();
    const auto row_ptrs = this->mtx->get_const_row_ptrs();
    const auto col_idxs = this->mtx->get_const_col_idxs();
    const auto drow_ptrs = this->dmtx->get_const_row_ptrs();
    const auto dcol_idxs = this->dmtx->get_const_col_idxs();
    const auto storage_offsets = storage_offset_array.get_data();
    const auto dstorage_offsets = dstorage_offset_array.get_data();
    for (auto allowed :
         {sparsity_type::full | sparsity_type::bitmap | sparsity_type::hash,
          sparsity_type::bitmap | sparsity_type::hash,
          sparsity_type::full | sparsity_type::hash, sparsity_type::hash}) {
        // check that storage offsets are calculated correctly
        // otherwise things might crash
        gko::kernels::reference::csr::build_lookup_offsets(
            this->ref, row_ptrs, col_idxs, num_rows, allowed, storage_offsets);
        gko::kernels::EXEC_NAMESPACE::csr::build_lookup_offsets(
            this->exec, drow_ptrs, dcol_idxs, num_rows, allowed,
            dstorage_offsets);

        GKO_ASSERT_ARRAY_EQ(storage_offset_array, dstorage_offset_array);

        gko::array<gko::int32> storage_array(this->ref,
                                             storage_offsets[num_rows]);
        gko::array<gko::int32> dstorage_array(this->exec,
                                              storage_offsets[num_rows]);
        const auto storage = storage_array.get_data();
        const auto dstorage = dstorage_array.get_data();
        const auto bitmap_equivalent =
            csr_lookup_allowed(allowed, sparsity_type::bitmap)
                ? sparsity_type::bitmap
                : sparsity_type::hash;
        const auto full_equivalent =
            csr_lookup_allowed(allowed, sparsity_type::full)
                ? sparsity_type::full
                : bitmap_equivalent;

        gko::kernels::reference::csr::build_lookup(
            this->ref, row_ptrs, col_idxs, num_rows, allowed, storage_offsets,
            row_descs, storage);
        gko::kernels::EXEC_NAMESPACE::csr::build_lookup(
            this->exec, drow_ptrs, dcol_idxs, num_rows, allowed,
            dstorage_offsets, drow_descs, dstorage);

        gko::array<gko::int64> host_row_descs(this->ref, drow_desc_array);
        gko::array<gko::int32> host_storage_array(this->ref, dstorage_array);
        for (int row = 0; row < num_rows; row++) {
            const auto row_begin = row_ptrs[row];
            const auto row_end = row_ptrs[row + 1];
            const auto row_nnz = row_end - row_begin;
            gko::matrix::csr::device_sparsity_lookup<index_type> lookup{
                row_ptrs,
                col_idxs,
                storage_offsets,
                host_storage_array.get_const_data(),
                host_row_descs.get_data(),
                static_cast<gko::size_type>(row)};
            ASSERT_EQ(host_row_descs.get_const_data()[row] & 0xF,
                      row_descs[row] & 0xF);
            for (auto nz = row_begin; nz < row_end; nz++) {
                const auto col = col_idxs[nz];
                ASSERT_EQ(lookup.lookup_unsafe(col) + row_begin, nz);
            }
            auto nz = row_begin;
            for (int col = 0; col < num_cols; col++) {
                auto found_nz = lookup[col];
                if (nz < row_end && col_idxs[nz] == col) {
                    ASSERT_EQ(found_nz, nz - row_begin);
                    nz++;
                } else {
                    ASSERT_EQ(found_nz, this->invalid_index);
                }
            }
        }
    }
}


}  //  namespace
