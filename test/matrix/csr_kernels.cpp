// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/csr_kernels.hpp"


#include <algorithm>
#include <numeric>
#include <random>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "common/unified/base/kernel_launch.hpp"
#include "core/base/array_access.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/test/utils.hpp"
#include "test/utils/executor.hpp"


class Csr : public CommonTestFixture {
protected:
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;

    Csr() : rand_engine(15) {}

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
        dx->copy_from(x);
        dalpha = Vec::create(exec);
        dalpha->copy_from(alpha);
    }

    std::default_random_engine rand_engine;

    std::unique_ptr<Mtx> x;
    std::unique_ptr<Vec> alpha;
    std::unique_ptr<Mtx> dx;
    std::unique_ptr<Vec> dalpha;
};


TEST_F(Csr, ScaleIsEquivalentToRef)
{
    set_up_apply_data();

    x->scale(alpha);
    dx->scale(dalpha);

    GKO_ASSERT_MTX_NEAR(dx, x, r<value_type>::value);
}


TEST_F(Csr, InvScaleIsEquivalentToRef)
{
    set_up_apply_data();

    x->inv_scale(alpha);
    dx->inv_scale(dalpha);

    GKO_ASSERT_MTX_NEAR(dx, x, r<value_type>::value);
}


template <typename IndexType>
class CsrLookup : public CommonTestFixture {
public:
    using value_type = float;
    using index_type = IndexType;
    using Mtx = gko::matrix::Csr<value_type, index_type>;

    CsrLookup() : rand_engine(15)
    {
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
        data.sort_row_major();
        // initialize the matrices
        mtx = Mtx::create(ref);
        mtx->read(data);
        dmtx = gko::clone(exec, mtx);
        const auto num_rows = mtx->get_size()[0];

        row_desc_array = gko::array<gko::int64>{ref, num_rows};
        drow_desc_array = gko::array<gko::int64>{exec, num_rows};
        storage_offset_array = gko::array<index_type>{ref, num_rows + 1};
        dstorage_offset_array = gko::array<index_type>{exec, num_rows + 1};
        storage_array.set_executor(ref);
        dstorage_array.set_executor(exec);
    }

    std::default_random_engine rand_engine;
    std::unique_ptr<Mtx> mtx;
    std::unique_ptr<Mtx> dmtx;
    gko::array<gko::int64> row_desc_array;
    gko::array<gko::int64> drow_desc_array;
    gko::array<index_type> storage_offset_array;
    gko::array<index_type> dstorage_offset_array;
    gko::array<gko::int32> storage_array;
    gko::array<gko::int32> dstorage_array;
    index_type invalid_index = gko::invalid_index<index_type>();
};

TYPED_TEST_SUITE(CsrLookup, gko::test::IndexTypes, TypenameNameGenerator);


template <typename IndexType>
void assert_lookup_correct(std::shared_ptr<const gko::EXEC_TYPE> exec,
                           const typename CsrLookup<IndexType>::Mtx* mtx,
                           const gko::array<IndexType>& storage_offsets,
                           const gko::array<gko::int32>& storage,
                           const gko::array<gko::int64>& row_descs)
{
    const auto num_rows = mtx->get_size()[0];
    const auto num_cols = mtx->get_size()[1];
    const auto row_ptrs = mtx->get_const_row_ptrs();
    const auto col_idxs = mtx->get_const_col_idxs();
    gko::array<bool> correct{exec, {true}};
    gko::kernels::EXEC_NAMESPACE::run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto num_cols, auto row_ptrs, auto col_idxs,
                      auto storage_offsets, auto storage, auto row_descs,
                      auto correct) {
            const auto row_begin = row_ptrs[row];
            const auto row_end = row_ptrs[row + 1];
            const auto row_nnz = row_end - row_begin;
            gko::matrix::csr::device_sparsity_lookup<IndexType> lookup{
                row_ptrs, col_idxs,  storage_offsets,
                storage,  row_descs, static_cast<gko::size_type>(row)};
            // check lookup for existing entries
            for (auto nz = row_begin; nz < row_end; nz++) {
                const auto col = col_idxs[nz];
                if (lookup.lookup_unsafe(col) + row_begin != nz) {
                    *correct = false;
                    return;
                }
            }
            // check generic lookup for all columns
            auto nz = row_begin;
            for (int col = 0; col < num_cols; col++) {
                auto found_nz = lookup[col];
                if (nz < row_end && col_idxs[nz] == col) {
                    if (found_nz != nz - row_begin) {
                        *correct = false;
                        return;
                    }
                    nz++;
                } else {
                    if (found_nz != gko::invalid_index<IndexType>()) {
                        *correct = false;
                        return;
                    }
                }
            }
        },
        num_rows, num_cols, row_ptrs, col_idxs, storage_offsets, storage,
        row_descs, correct);
    ASSERT_TRUE(get_element(correct, 0));
}


TYPED_TEST(CsrLookup, BuildLookupWorks)
{
    using index_type = typename TestFixture::index_type;
    using gko::matrix::csr::sparsity_type;
    const auto num_rows = this->mtx->get_size()[0];
    const auto num_cols = this->mtx->get_size()[1];
    const auto row_descs = this->row_desc_array.get_data();
    const auto drow_descs = this->drow_desc_array.get_data();
    const auto row_ptrs = this->mtx->get_const_row_ptrs();
    const auto col_idxs = this->mtx->get_const_col_idxs();
    const auto drow_ptrs = this->dmtx->get_const_row_ptrs();
    const auto dcol_idxs = this->dmtx->get_const_col_idxs();
    const auto storage_offsets = this->storage_offset_array.get_data();
    const auto dstorage_offsets = this->dstorage_offset_array.get_data();
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

        GKO_ASSERT_ARRAY_EQ(this->storage_offset_array,
                            this->dstorage_offset_array);

        this->storage_array.resize_and_reset(storage_offsets[num_rows]);
        this->dstorage_array.resize_and_reset(storage_offsets[num_rows]);
        const auto storage = this->storage_array.get_data();
        const auto dstorage = this->dstorage_array.get_data();
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

        assert_lookup_correct(this->exec, this->dmtx.get(),
                              this->dstorage_offset_array, this->dstorage_array,
                              this->drow_desc_array);
        // check that all rows use the same lookup type
        gko::array<gko::int64> host_row_desc_array{this->ref,
                                                   this->drow_desc_array};
        const auto host_row_descs = host_row_desc_array.get_const_data();
        for (gko::size_type row = 0; row < num_rows; row++) {
            ASSERT_EQ(host_row_descs[row] & 0xF, row_descs[row] & 0xF);
        }
    }
}
