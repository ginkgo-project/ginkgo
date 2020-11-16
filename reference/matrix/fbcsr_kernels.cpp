/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include "core/matrix/fbcsr_kernels.hpp"


#include <algorithm>
#include <iterator>
#include <numeric>
#include <utility>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/sellp.hpp>


#include "core/base/allocator.hpp"
#include "core/base/iterator_factory.hpp"
#include "core/components/fixed_block.hpp"
#include "core/components/prefix_sum.hpp"
#include "core/matrix/fbcsr_builder.hpp"
#include "reference/components/fbcsr_spgeam.hpp"
#include "reference/components/format_conversion.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The Compressed sparse row matrix format namespace.
 * @ref Fbcsr
 * @ingroup fbcsr
 */
namespace fbcsr {


template <typename ValueType, typename IndexType>
void spmv(const std::shared_ptr<const ReferenceExecutor> exec,
          const matrix::Fbcsr<ValueType, IndexType> *const a,
          const matrix::Dense<ValueType> *const b,
          matrix::Dense<ValueType> *const c)
{
    const int bs = a->get_block_size();
    const IndexType nvecs = static_cast<IndexType>(b->get_size()[1]);
    const IndexType nbrows =
        gko::blockutils::getNumFixedBlocks(bs, a->get_size()[0]);
    auto row_ptrs = a->get_const_row_ptrs();
    auto col_idxs = a->get_const_col_idxs();
    auto vals = a->get_const_values();
    const gko::blockutils::DenseBlocksView<const ValueType, IndexType> avalues(
        vals, bs, bs);

    ValueType *const cvals = c->get_values();

    for (IndexType ibrow = 0; ibrow < nbrows; ++ibrow) {
        const IndexType crowblkend = (ibrow + 1) * bs * nvecs;
        for (IndexType i = ibrow * bs * nvecs; i < crowblkend; i++)
            cvals[i] = zero<ValueType>();

        for (IndexType inz = row_ptrs[ibrow]; inz < row_ptrs[ibrow + 1];
             ++inz) {
            for (int ib = 0; ib < bs; ib++) {
                const IndexType row = ibrow * bs + ib;
                for (int jb = 0; jb < bs; jb++) {
                    const auto val = avalues(inz, ib, jb);
                    const auto col = col_idxs[inz] * bs + jb;
                    for (size_type j = 0; j < nvecs; ++j)
                        c->at(row, j) += val * b->at(col, j);
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_FBCSR_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(const std::shared_ptr<const ReferenceExecutor> exec,
                   const matrix::Dense<ValueType> *const alpha,
                   const matrix::Fbcsr<ValueType, IndexType> *const a,
                   const matrix::Dense<ValueType> *const b,
                   const matrix::Dense<ValueType> *const beta,
                   matrix::Dense<ValueType> *const c)
{
    const int bs = a->get_block_size();
    const IndexType nvecs = static_cast<IndexType>(b->get_size()[1]);
    const IndexType nbrows =
        gko::blockutils::getNumFixedBlocks(bs, a->get_size()[0]);
    auto row_ptrs = a->get_const_row_ptrs();
    auto col_idxs = a->get_const_col_idxs();
    auto vals = a->get_const_values();
    auto valpha = alpha->at(0, 0);
    auto vbeta = beta->at(0, 0);
    const gko::blockutils::DenseBlocksView<const ValueType, IndexType> avalues(
        vals, bs, bs);

    ValueType *const cvals = c->get_values();

    for (IndexType ibrow = 0; ibrow < nbrows; ++ibrow) {
        const IndexType crowblkend = (ibrow + 1) * bs * nvecs;
        for (IndexType i = ibrow * bs * nvecs; i < crowblkend; i++)
            cvals[i] *= vbeta;

        for (IndexType inz = row_ptrs[ibrow]; inz < row_ptrs[ibrow + 1];
             ++inz) {
            for (int ib = 0; ib < bs; ib++) {
                const IndexType row = ibrow * bs + ib;
                for (int jb = 0; jb < bs; jb++) {
                    const auto val = avalues(inz, ib, jb);
                    const auto col = col_idxs[inz] * bs + jb;
                    for (size_type j = 0; j < nvecs; ++j)
                        c->at(row, j) += valpha * val * b->at(col, j);
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void spgemm_insert_row(unordered_set<IndexType> &cols,
                       const matrix::Fbcsr<ValueType, IndexType> *c,
                       size_type row) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    auto row_ptrs = c->get_const_row_ptrs();
//    auto col_idxs = c->get_const_col_idxs();
//    cols.insert(col_idxs + row_ptrs[row], col_idxs + row_ptrs[row + 1]);
//}


template <typename ValueType, typename IndexType>
void spgemm_insert_row2(unordered_set<IndexType> &cols,
                        const matrix::Fbcsr<ValueType, IndexType> *a,
                        const matrix::Fbcsr<ValueType, IndexType> *b,
                        size_type row) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    auto a_row_ptrs = a->get_const_row_ptrs();
//    auto a_col_idxs = a->get_const_col_idxs();
//    auto b_row_ptrs = b->get_const_row_ptrs();
//    auto b_col_idxs = b->get_const_col_idxs();
//    for (size_type a_nz = a_row_ptrs[row];
//         a_nz < size_type(a_row_ptrs[row + 1]); ++a_nz) {
//        auto a_col = a_col_idxs[a_nz];
//        auto b_row = a_col;
//        cols.insert(b_col_idxs + b_row_ptrs[b_row],
//                    b_col_idxs + b_row_ptrs[b_row + 1]);
//    }
//}


template <typename ValueType, typename IndexType>
void spgemm_accumulate_row(map<IndexType, ValueType> &cols,
                           const matrix::Fbcsr<ValueType, IndexType> *c,
                           ValueType scale, size_type row) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    auto row_ptrs = c->get_const_row_ptrs();
//    auto col_idxs = c->get_const_col_idxs();
//    auto vals = c->get_const_values();
//    for (size_type c_nz = row_ptrs[row]; c_nz < size_type(row_ptrs[row + 1]);
//         ++c_nz) {
//        auto c_col = col_idxs[c_nz];
//        auto c_val = vals[c_nz];
//        cols[c_col] += scale * c_val;
//    }
//}


template <typename ValueType, typename IndexType>
void spgemm_accumulate_row2(map<IndexType, ValueType> &cols,
                            const matrix::Fbcsr<ValueType, IndexType> *a,
                            const matrix::Fbcsr<ValueType, IndexType> *b,
                            ValueType scale, size_type row) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    auto a_row_ptrs = a->get_const_row_ptrs();
//    auto a_col_idxs = a->get_const_col_idxs();
//    auto a_vals = a->get_const_values();
//    auto b_row_ptrs = b->get_const_row_ptrs();
//    auto b_col_idxs = b->get_const_col_idxs();
//    auto b_vals = b->get_const_values();
//    for (size_type a_nz = a_row_ptrs[row];
//         a_nz < size_type(a_row_ptrs[row + 1]); ++a_nz) {
//        auto a_col = a_col_idxs[a_nz];
//        auto a_val = a_vals[a_nz];
//        auto b_row = a_col;
//        for (size_type b_nz = b_row_ptrs[b_row];
//             b_nz < size_type(b_row_ptrs[b_row + 1]); ++b_nz) {
//            auto b_col = b_col_idxs[b_nz];
//            auto b_val = b_vals[b_nz];
//            cols[b_col] += scale * a_val * b_val;
//        }
//    }
//}


template <typename ValueType, typename IndexType>
void spgemm(std::shared_ptr<const ReferenceExecutor> exec,
            const matrix::Fbcsr<ValueType, IndexType> *a,
            const matrix::Fbcsr<ValueType, IndexType> *b,
            matrix::Fbcsr<ValueType, IndexType> *c) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    auto num_rows = a->get_size()[0];
//
//    // first sweep: count nnz for each row
//    auto c_row_ptrs = c->get_row_ptrs();
//
//    unordered_set<IndexType> local_col_idxs(exec);
//    for (size_type a_row = 0; a_row < num_rows; ++a_row) {
//        local_col_idxs.clear();
//        spgemm_insert_row2(local_col_idxs, a, b, a_row);
//        c_row_ptrs[a_row] = local_col_idxs.size();
//    }
//
//    // build row pointers
//    components::prefix_sum(exec, c_row_ptrs, num_rows + 1);
//
//    // second sweep: accumulate non-zeros
//    auto new_nnz = c_row_ptrs[num_rows];
//    matrix::FbcsrBuilder<ValueType, IndexType> c_builder{c};
//    auto &c_col_idxs_array = c_builder.get_col_idx_array();
//    auto &c_vals_array = c_builder.get_value_array();
//    c_col_idxs_array.resize_and_reset(new_nnz);
//    c_vals_array.resize_and_reset(new_nnz);
//    auto c_col_idxs = c_col_idxs_array.get_data();
//    auto c_vals = c_vals_array.get_data();
//
//    map<IndexType, ValueType> local_row_nzs(exec);
//    for (size_type a_row = 0; a_row < num_rows; ++a_row) {
//        local_row_nzs.clear();
//        spgemm_accumulate_row2(local_row_nzs, a, b, one<ValueType>(), a_row);
//        // store result
//        auto c_nz = c_row_ptrs[a_row];
//        for (auto pair : local_row_nzs) {
//            c_col_idxs[c_nz] = pair.first;
//            c_vals[c_nz] = pair.second;
//            ++c_nz;
//        }
//    }
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_FBCSR_SPGEMM_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spgemm(std::shared_ptr<const ReferenceExecutor> exec,
                     const matrix::Dense<ValueType> *alpha,
                     const matrix::Fbcsr<ValueType, IndexType> *a,
                     const matrix::Fbcsr<ValueType, IndexType> *b,
                     const matrix::Dense<ValueType> *beta,
                     const matrix::Fbcsr<ValueType, IndexType> *d,
                     matrix::Fbcsr<ValueType, IndexType> *c)
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    auto num_rows = a->get_size()[0];
//    auto valpha = alpha->at(0, 0);
//    auto vbeta = beta->at(0, 0);
//
//    // first sweep: count nnz for each row
//    auto c_row_ptrs = c->get_row_ptrs();
//
//    unordered_set<IndexType> local_col_idxs(exec);
//    for (size_type a_row = 0; a_row < num_rows; ++a_row) {
//        local_col_idxs.clear();
//        spgemm_insert_row(local_col_idxs, d, a_row);
//        spgemm_insert_row2(local_col_idxs, a, b, a_row);
//        c_row_ptrs[a_row] = local_col_idxs.size();
//    }
//
//    // build row pointers
//    components::prefix_sum(exec, c_row_ptrs, num_rows + 1);
//
//    // second sweep: accumulate non-zeros
//    auto new_nnz = c_row_ptrs[num_rows];
//    matrix::FbcsrBuilder<ValueType, IndexType> c_builder{c};
//    auto &c_col_idxs_array = c_builder.get_col_idx_array();
//    auto &c_vals_array = c_builder.get_value_array();
//    c_col_idxs_array.resize_and_reset(new_nnz);
//    c_vals_array.resize_and_reset(new_nnz);
//    auto c_col_idxs = c_col_idxs_array.get_data();
//    auto c_vals = c_vals_array.get_data();
//
//    map<IndexType, ValueType> local_row_nzs(exec);
//    for (size_type a_row = 0; a_row < num_rows; ++a_row) {
//        local_row_nzs.clear();
//        spgemm_accumulate_row(local_row_nzs, d, vbeta, a_row);
//        spgemm_accumulate_row2(local_row_nzs, a, b, valpha, a_row);
//        // store result
//        auto c_nz = c_row_ptrs[a_row];
//        for (auto pair : local_row_nzs) {
//            c_col_idxs[c_nz] = pair.first;
//            c_vals[c_nz] = pair.second;
//            ++c_nz;
//        }
//    }
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_ADVANCED_SPGEMM_KERNEL);


template <typename ValueType, typename IndexType>
void spgeam(std::shared_ptr<const ReferenceExecutor> exec,
            const matrix::Dense<ValueType> *alpha,
            const matrix::Fbcsr<ValueType, IndexType> *a,
            const matrix::Dense<ValueType> *beta,
            const matrix::Fbcsr<ValueType, IndexType> *b,
            matrix::Fbcsr<ValueType, IndexType> *c) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    auto num_rows = a->get_size()[0];
//    auto valpha = alpha->at(0, 0);
//    auto vbeta = beta->at(0, 0);
//
//    // first sweep: count nnz for each row
//    auto c_row_ptrs = c->get_row_ptrs();
//
//    abstract_spgeam(
//        a, b, [](IndexType) { return IndexType{}; },
//        [](IndexType, IndexType, ValueType, ValueType, IndexType &nnz) {
//            ++nnz;
//        },
//        [&](IndexType row, IndexType nnz) { c_row_ptrs[row] = nnz; });
//
//    // build row pointers
//    components::prefix_sum(exec, c_row_ptrs, num_rows + 1);
//
//    // second sweep: accumulate non-zeros
//    auto new_nnz = c_row_ptrs[num_rows];
//    matrix::FbcsrBuilder<ValueType, IndexType> c_builder{c};
//    auto &c_col_idxs_array = c_builder.get_col_idx_array();
//    auto &c_vals_array = c_builder.get_value_array();
//    c_col_idxs_array.resize_and_reset(new_nnz);
//    c_vals_array.resize_and_reset(new_nnz);
//    auto c_col_idxs = c_col_idxs_array.get_data();
//    auto c_vals = c_vals_array.get_data();
//
//    abstract_spgeam(
//        a, b, [&](IndexType row) { return c_row_ptrs[row]; },
//        [&](IndexType, IndexType col, ValueType a_val, ValueType b_val,
//            IndexType &nz) {
//            c_vals[nz] = valpha * a_val + vbeta * b_val;
//            c_col_idxs[nz] = col;
//            ++nz;
//        },
//        [](IndexType, IndexType) {});
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_FBCSR_SPGEAM_KERNEL);


// template <typename IndexType>
// void convert_row_ptrs_to_idxs(std::shared_ptr<const ReferenceExecutor> exec,
//                               const IndexType *ptrs, size_type num_rows,
//                               IndexType *idxs)
// {
//    convert_ptrs_to_idxs(ptrs, num_rows, idxs);
// }


template <typename ValueType, typename IndexType>
void convert_to_coo(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::Fbcsr<ValueType, IndexType> *source,
                    matrix::Coo<ValueType, IndexType> *result)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_CONVERT_TO_COO_KERNEL);

template <typename ValueType, typename IndexType>
void convert_to_dense(const std::shared_ptr<const ReferenceExecutor> exec,
                      const matrix::Fbcsr<ValueType, IndexType> *const source,
                      matrix::Dense<ValueType> *const result)
{
    const int bs = source->get_block_size();
    const size_type nbrows =
        gko::blockutils::getNumFixedBlocks(bs, source->get_size()[0]);
    const size_type nbcols =
        gko::blockutils::getNumFixedBlocks(bs, source->get_size()[1]);
    const IndexType *const row_ptrs = source->get_const_row_ptrs();
    const IndexType *const col_idxs = source->get_const_col_idxs();
    const ValueType *const vals = source->get_const_values();

    const gko::blockutils::DenseBlocksView<const ValueType, IndexType> values(
        vals, bs, bs);

    for (size_type brow = 0; brow < nbrows; ++brow) {
        for (size_type bcol = 0; bcol < nbcols; ++bcol) {
            for (int ib = 0; ib < bs; ib++)
                for (int jb = 0; jb < bs; jb++)
                    result->at(brow * bs + ib, bcol * bs + jb) =
                        zero<ValueType>();
        }
        for (IndexType ibnz = row_ptrs[brow]; ibnz < row_ptrs[brow + 1];
             ++ibnz) {
            for (int ib = 0; ib < bs; ib++) {
                const IndexType row = brow * bs + ib;
                for (int jb = 0; jb < bs; jb++)
                    result->at(row, col_idxs[ibnz] * bs + jb) =
                        values(ibnz, ib, jb);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(const std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::Fbcsr<ValueType, IndexType> *const source,
                    matrix::Csr<ValueType, IndexType> *const result)
{
    const int bs = source->get_block_size();
    const size_type nbrows =
        gko::blockutils::getNumFixedBlocks(bs, source->get_size()[0]);
    const size_type nbcols =
        gko::blockutils::getNumFixedBlocks(bs, source->get_size()[1]);
    const IndexType *const browptrs = source->get_const_row_ptrs();
    const IndexType *const bcolinds = source->get_const_col_idxs();
    const ValueType *const bvals = source->get_const_values();

    assert(nbrows * bs == result->get_size()[0]);
    assert(nbcols * bs == result->get_size()[1]);
    assert(source->get_num_stored_elements() ==
           result->get_num_stored_elements());

    IndexType *const row_ptrs = result->get_row_ptrs();
    IndexType *const col_idxs = result->get_col_idxs();
    ValueType *const vals = result->get_values();

    const gko::blockutils::DenseBlocksView<const ValueType, IndexType> bvalues(
        bvals, bs, bs);

    for (size_type brow = 0; brow < nbrows; ++brow) {
        const IndexType nz_browstart = browptrs[brow] * bs * bs;
        const IndexType numblocks_brow = browptrs[brow + 1] - browptrs[brow];
        for (int ib = 0; ib < bs; ib++)
            row_ptrs[brow * bs + ib] = nz_browstart + numblocks_brow * bs * ib;

        for (IndexType ibnz = browptrs[brow]; ibnz < browptrs[brow + 1];
             ++ibnz) {
            const IndexType bcol = bcolinds[ibnz];

            for (int ib = 0; ib < bs; ib++) {
                const IndexType row = brow * bs + ib;
                const IndexType inz_blockstart =
                    row_ptrs[row] + (ibnz - browptrs[brow]) * bs;

                for (int jb = 0; jb < bs; jb++) {
                    const IndexType inz = inz_blockstart + jb;
                    vals[inz] = bvalues(ibnz, ib, jb);
                    col_idxs[inz] = bcol * bs + jb;
                }
            }
        }
    }

    row_ptrs[source->get_size()[0]] = source->get_num_stored_elements();
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_sellp(std::shared_ptr<const ReferenceExecutor> exec,
                      const matrix::Fbcsr<ValueType, IndexType> *source,
                      matrix::Sellp<ValueType, IndexType> *result)
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    auto num_rows = result->get_size()[0];
//    auto num_cols = result->get_size()[1];
//    auto vals = result->get_values();
//    auto col_idxs = result->get_col_idxs();
//    auto slice_lengths = result->get_slice_lengths();
//    auto slice_sets = result->get_slice_sets();
//    auto slice_size = (result->get_slice_size() == 0)
//                          ? matrix::default_slice_size
//                          : result->get_slice_size();
//    auto stride_factor = (result->get_stride_factor() == 0)
//                             ? matrix::default_stride_factor
//                             : result->get_stride_factor();
//
//    const auto source_row_ptrs = source->get_const_row_ptrs();
//    const auto source_col_idxs = source->get_const_col_idxs();
//    const auto source_values = source->get_const_values();
//
//    int slice_num = ceildiv(num_rows, slice_size);
//    slice_sets[0] = 0;
//    for (size_type slice = 0; slice < slice_num; slice++) {
//        if (slice > 0) {
//            slice_sets[slice] =
//                slice_sets[slice - 1] + slice_lengths[slice - 1];
//        }
//        slice_lengths[slice] = 0;
//        for (size_type row = 0; row < slice_size; row++) {
//            size_type global_row = slice * slice_size + row;
//            if (global_row >= num_rows) {
//                break;
//            }
//            slice_lengths[slice] =
//                (slice_lengths[slice] >
//                 source_row_ptrs[global_row + 1] -
//                 source_row_ptrs[global_row])
//                    ? slice_lengths[slice]
//                    : source_row_ptrs[global_row + 1] -
//                          source_row_ptrs[global_row];
//        }
//        slice_lengths[slice] =
//            stride_factor * ceildiv(slice_lengths[slice], stride_factor);
//        for (size_type row = 0; row < slice_size; row++) {
//            size_type global_row = slice * slice_size + row;
//            if (global_row >= num_rows) {
//                break;
//            }
//            size_type sellp_ind = slice_sets[slice] * slice_size + row;
//            for (size_type fbcsr_ind = source_row_ptrs[global_row];
//                 fbcsr_ind < source_row_ptrs[global_row + 1]; fbcsr_ind++) {
//                vals[sellp_ind] = source_values[fbcsr_ind];
//                col_idxs[sellp_ind] = source_col_idxs[fbcsr_ind];
//                sellp_ind += slice_size;
//            }
//            for (size_type i = sellp_ind;
//                 i <
//                 (slice_sets[slice] + slice_lengths[slice]) * slice_size +
//                 row; i += slice_size) {
//                col_idxs[i] = 0;
//                vals[i] = zero<ValueType>();
//            }
//        }
//    }
//    if (slice_num > 0) {
//        slice_sets[slice_num] =
//            slice_sets[slice_num - 1] + slice_lengths[slice_num - 1];
//    }
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_CONVERT_TO_SELLP_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_total_cols(std::shared_ptr<const ReferenceExecutor> exec,
                          const matrix::Fbcsr<ValueType, IndexType> *source,
                          size_type *result, size_type stride_factor,
                          size_type slice_size) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    size_type total_cols = 0;
//    const auto num_rows = source->get_size()[0];
//    const auto slice_num = ceildiv(num_rows, slice_size);
//
//    const auto row_ptrs = source->get_const_row_ptrs();
//
//    for (size_type slice = 0; slice < slice_num; slice++) {
//        IndexType max_nnz_per_row_in_this_slice = 0;
//        for (size_type row = 0;
//             row < slice_size && row + slice * slice_size < num_rows; row++) {
//            size_type global_row = slice * slice_size + row;
//            max_nnz_per_row_in_this_slice =
//                max(row_ptrs[global_row + 1] - row_ptrs[global_row],
//                    max_nnz_per_row_in_this_slice);
//        }
//        total_cols += ceildiv(max_nnz_per_row_in_this_slice, stride_factor) *
//                      stride_factor;
//    }
//
//    *result = total_cols;
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_CALCULATE_TOTAL_COLS_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_ell(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::Fbcsr<ValueType, IndexType> *source,
                    matrix::Ell<ValueType, IndexType> *result)
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    const auto num_rows = source->get_size()[0];
//    const auto num_cols = source->get_size()[1];
//    const auto vals = source->get_const_values();
//    const auto col_idxs = source->get_const_col_idxs();
//    const auto row_ptrs = source->get_const_row_ptrs();
//
//    const auto num_stored_elements_per_row =
//        result->get_num_stored_elements_per_row();
//
//    for (size_type row = 0; row < num_rows; row++) {
//        for (size_type i = 0; i < num_stored_elements_per_row; i++) {
//            result->val_at(row, i) = zero<ValueType>();
//            result->col_at(row, i) = 0;
//        }
//        for (size_type col_idx = 0; col_idx < row_ptrs[row + 1] -
//        row_ptrs[row];
//             col_idx++) {
//            result->val_at(row, col_idx) = vals[row_ptrs[row] + col_idx];
//            result->col_at(row, col_idx) = col_idxs[row_ptrs[row] + col_idx];
//        }
//    }
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_CONVERT_TO_ELL_KERNEL);


template <typename ValueType, typename IndexType, typename UnaryOperator,
          bool transpose_blocks>
void convert_fbcsr_to_fbcsc(const size_type num_blk_rows, const int blksz,
                            const IndexType *const row_ptrs,
                            const IndexType *const col_idxs,
                            const ValueType *const fbcsr_vals,
                            IndexType *const row_idxs,
                            IndexType *const col_ptrs,
                            ValueType *const csc_vals, UnaryOperator op)
{
    const gko::blockutils::DenseBlocksView<const ValueType, IndexType> rvalues(
        fbcsr_vals, blksz, blksz);
    gko::blockutils::DenseBlocksView<ValueType, IndexType> cvalues(
        csc_vals, blksz, blksz);
    for (size_type brow = 0; brow < num_blk_rows; ++brow) {
        for (auto i = row_ptrs[brow]; i < row_ptrs[brow + 1]; ++i) {
            const auto dest_idx = col_ptrs[col_idxs[i]]++;
            row_idxs[dest_idx] = brow;
            for (int ib = 0; ib < blksz; ib++)
                for (int jb = 0; jb < blksz; jb++)
                    // csc_vals[dest_idx] = op(fbcsr_vals[i]);
                    cvalues(dest_idx, ib, jb) =
                        op(transpose_blocks ? rvalues(i, jb, ib)
                                            : rvalues(i, ib, jb));
        }
    }
}


template <typename ValueType, typename IndexType, typename UnaryOperator>
void transpose_and_transform(
    const std::shared_ptr<const ReferenceExecutor> exec,
    matrix::Fbcsr<ValueType, IndexType> *const trans,
    const matrix::Fbcsr<ValueType, IndexType> *const orig, UnaryOperator op)
{
    const int bs = orig->get_block_size();
    auto trans_row_ptrs = trans->get_row_ptrs();
    auto orig_row_ptrs = orig->get_const_row_ptrs();
    auto trans_col_idxs = trans->get_col_idxs();
    auto orig_col_idxs = orig->get_const_col_idxs();
    auto trans_vals = trans->get_values();
    auto orig_vals = orig->get_const_values();

    auto orig_num_cols = orig->get_size()[1];
    const size_type nbcols =
        gko::blockutils::getNumFixedBlocks(bs, orig_num_cols);
    auto orig_num_rows = orig->get_size()[0];
    const size_type nbrows =
        gko::blockutils::getNumFixedBlocks(bs, orig_num_rows);
    auto orig_nbnz = orig_row_ptrs[nbrows];

    trans_row_ptrs[0] = 0;
    convert_idxs_to_ptrs(orig_col_idxs, orig_nbnz, trans_row_ptrs + 1, nbcols);

    convert_fbcsr_to_fbcsc<ValueType, IndexType, UnaryOperator, true>(
        nbrows, bs, orig_row_ptrs, orig_col_idxs, orig_vals, trans_col_idxs,
        trans_row_ptrs + 1, trans_vals, op);
}


template <typename ValueType, typename IndexType>
void transpose(std::shared_ptr<const ReferenceExecutor> exec,
               const matrix::Fbcsr<ValueType, IndexType> *orig,
               matrix::Fbcsr<ValueType, IndexType> *trans)
{
    transpose_and_transform(exec, trans, orig,
                            [](const ValueType x) { return x; });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void conj_transpose(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::Fbcsr<ValueType, IndexType> *orig,
                    matrix::Fbcsr<ValueType, IndexType> *trans)
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    transpose_and_transform(exec, trans, orig,
//                            [](const ValueType x) { return conj(x); });
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_max_nnz_per_row(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Fbcsr<ValueType, IndexType> *source,
    size_type *result) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    const auto num_rows = source->get_size()[0];
//    const auto row_ptrs = source->get_const_row_ptrs();
//    IndexType max_nnz = 0;
//
//    for (size_type i = 0; i < num_rows; i++) {
//        max_nnz = std::max(row_ptrs[i + 1] - row_ptrs[i], max_nnz);
//    }
//
//    *result = max_nnz;
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_CALCULATE_MAX_NNZ_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_hybrid(std::shared_ptr<const ReferenceExecutor> exec,
                       const matrix::Fbcsr<ValueType, IndexType> *source,
                       matrix::Hybrid<ValueType, IndexType> *result)
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    auto num_rows = result->get_size()[0];
//    auto num_cols = result->get_size()[1];
//    auto strategy = result->get_strategy();
//    auto ell_lim = strategy->get_ell_num_stored_elements_per_row();
//    auto coo_lim = strategy->get_coo_nnz();
//    auto coo_val = result->get_coo_values();
//    auto coo_col = result->get_coo_col_idxs();
//    auto coo_row = result->get_coo_row_idxs();
//
//    // Initial Hybrid Matrix
//    for (size_type i = 0; i < result->get_ell_num_stored_elements_per_row();
//         i++) {
//        for (size_type j = 0; j < result->get_ell_stride(); j++) {
//            result->ell_val_at(j, i) = zero<ValueType>();
//            result->ell_col_at(j, i) = 0;
//        }
//    }
//    for (size_type i = 0; i < result->get_coo_num_stored_elements(); i++) {
//        coo_val[i] = zero<ValueType>();
//        coo_col[i] = 0;
//        coo_row[i] = 0;
//    }
//
//    const auto fbcsr_row_ptrs = source->get_const_row_ptrs();
//    const auto fbcsr_vals = source->get_const_values();
//    size_type fbcsr_idx = 0;
//    size_type coo_idx = 0;
//    for (IndexType row = 0; row < num_rows; row++) {
//        size_type ell_idx = 0;
//        while (fbcsr_idx < fbcsr_row_ptrs[row + 1]) {
//            const auto val = fbcsr_vals[fbcsr_idx];
//            if (ell_idx < ell_lim) {
//                result->ell_val_at(row, ell_idx) = val;
//                result->ell_col_at(row, ell_idx) =
//                    source->get_const_col_idxs()[fbcsr_idx];
//                ell_idx++;
//            } else {
//                coo_val[coo_idx] = val;
//                coo_col[coo_idx] = source->get_const_col_idxs()[fbcsr_idx];
//                coo_row[coo_idx] = row;
//                coo_idx++;
//            }
//            fbcsr_idx++;
//        }
//    }
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_CONVERT_TO_HYBRID_KERNEL);


template <typename ValueType, typename IndexType>
void row_permute_impl(std::shared_ptr<const ReferenceExecutor> exec,
                      const Array<IndexType> *permutation_indices,
                      const matrix::Fbcsr<ValueType, IndexType> *orig,
                      matrix::Fbcsr<ValueType, IndexType> *row_permuted)
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    auto perm = permutation_indices->get_const_data();
//    auto orig_row_ptrs = orig->get_const_row_ptrs();
//    auto orig_col_idxs = orig->get_const_col_idxs();
//    auto orig_vals = orig->get_const_values();
//    auto rp_row_ptrs = row_permuted->get_row_ptrs();
//    auto rp_col_idxs = row_permuted->get_col_idxs();
//    auto rp_vals = row_permuted->get_values();
//    size_type num_rows = orig->get_size()[0];
//    size_type num_nnz = orig->get_num_stored_elements();
//
//    size_type cur_ptr = 0;
//    rp_row_ptrs[0] = cur_ptr;
//    vector<size_type> orig_num_nnz_per_row(num_rows, 0, exec);
//    for (size_type row = 0; row < num_rows; ++row) {
//        orig_num_nnz_per_row[row] = orig_row_ptrs[row + 1] -
//        orig_row_ptrs[row];
//    }
//    for (size_type row = 0; row < num_rows; ++row) {
//        rp_row_ptrs[row + 1] =
//            rp_row_ptrs[row] + orig_num_nnz_per_row[perm[row]];
//    }
//    rp_row_ptrs[num_rows] = orig_row_ptrs[num_rows];
//    for (size_type row = 0; row < num_rows; ++row) {
//        auto new_row = perm[row];
//        auto new_k = orig_row_ptrs[new_row];
//        for (size_type k = rp_row_ptrs[row];
//             k < size_type(rp_row_ptrs[row + 1]); ++k) {
//            rp_col_idxs[k] = orig_col_idxs[new_k];
//            rp_vals[k] = orig_vals[new_k];
//            new_k++;
//        }
//    }
//}


template <typename ValueType, typename IndexType>
void row_permute(std::shared_ptr<const ReferenceExecutor> exec,
                 const Array<IndexType> *permutation_indices,
                 const matrix::Fbcsr<ValueType, IndexType> *orig,
                 matrix::Fbcsr<ValueType, IndexType> *row_permuted)
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    row_permute_impl(exec, permutation_indices, orig, row_permuted);
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_ROW_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inverse_row_permute(std::shared_ptr<const ReferenceExecutor> exec,
                         const Array<IndexType> *permutation_indices,
                         const matrix::Fbcsr<ValueType, IndexType> *orig,
                         matrix::Fbcsr<ValueType, IndexType> *row_permuted)
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    auto perm = permutation_indices->get_const_data();
//    Array<IndexType> inv_perm(*permutation_indices);
//    auto iperm = inv_perm.get_data();
//    for (size_type ind = 0; ind < inv_perm.get_num_elems(); ++ind) {
//        iperm[perm[ind]] = ind;
//    }
//
//    row_permute_impl(exec, &inv_perm, orig, row_permuted);
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_INVERSE_ROW_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void column_permute_impl(const Array<IndexType> *permutation_indices,
                         const matrix::Fbcsr<ValueType, IndexType> *orig,
                         matrix::Fbcsr<ValueType, IndexType> *column_permuted)
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    auto perm = permutation_indices->get_const_data();
//    auto orig_row_ptrs = orig->get_const_row_ptrs();
//    auto orig_col_idxs = orig->get_const_col_idxs();
//    auto orig_vals = orig->get_const_values();
//    auto cp_row_ptrs = column_permuted->get_row_ptrs();
//    auto cp_col_idxs = column_permuted->get_col_idxs();
//    auto cp_vals = column_permuted->get_values();
//    auto num_nnz = orig->get_num_stored_elements();
//    size_type num_rows = orig->get_size()[0];
//    size_type num_cols = orig->get_size()[1];
//
//    for (size_type row = 0; row < num_rows; ++row) {
//        cp_row_ptrs[row] = orig_row_ptrs[row];
//        for (size_type k = orig_row_ptrs[row];
//             k < size_type(orig_row_ptrs[row + 1]); ++k) {
//            cp_col_idxs[k] = perm[orig_col_idxs[k]];
//            cp_vals[k] = orig_vals[k];
//        }
//    }
//    cp_row_ptrs[num_rows] = orig_row_ptrs[num_rows];
//}


template <typename ValueType, typename IndexType>
void column_permute(std::shared_ptr<const ReferenceExecutor> exec,
                    const Array<IndexType> *permutation_indices,
                    const matrix::Fbcsr<ValueType, IndexType> *orig,
                    matrix::Fbcsr<ValueType, IndexType> *column_permuted)
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    auto perm = permutation_indices->get_const_data();
//    Array<IndexType> inv_perm(*permutation_indices);
//    auto iperm = inv_perm.get_data();
//    for (size_type ind = 0; ind < inv_perm.get_num_elems(); ++ind) {
//        iperm[perm[ind]] = ind;
//    }
//    column_permute_impl(&inv_perm, orig, column_permuted);
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_COLUMN_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inverse_column_permute(
    std::shared_ptr<const ReferenceExecutor> exec,
    const Array<IndexType> *permutation_indices,
    const matrix::Fbcsr<ValueType, IndexType> *orig,
    matrix::Fbcsr<ValueType, IndexType> *column_permuted) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    column_permute_impl(permutation_indices, orig, column_permuted);
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_INVERSE_COLUMN_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_nonzeros_per_row(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Fbcsr<ValueType, IndexType> *source, Array<size_type> *result)
{
    const auto row_ptrs = source->get_const_row_ptrs();
    auto row_nnz_val = result->get_data();
    const int bs = source->get_block_size();
    assert(result->get_num_elems() == source->get_size()[0]);

    for (size_type i = 0; i < result->get_num_elems(); i++) {
        const size_type ibrow = i / bs;
        row_nnz_val[i] = (row_ptrs[ibrow + 1] - row_ptrs[ibrow]) * bs;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_CALCULATE_NONZEROS_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void sort_by_column_index(std::shared_ptr<const ReferenceExecutor> exec,
                          matrix::Fbcsr<ValueType, IndexType> *to_sort)
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    auto values = to_sort->get_values();
//    auto row_ptrs = to_sort->get_row_ptrs();
//    auto col_idxs = to_sort->get_col_idxs();
//    const auto number_rows = to_sort->get_size()[0];
//    for (size_type i = 0; i < number_rows; ++i) {
//        auto start_row_idx = row_ptrs[i];
//        auto row_nnz = row_ptrs[i + 1] - start_row_idx;
//        auto helper = detail::IteratorFactory<IndexType, ValueType>(
//            col_idxs + start_row_idx, values + start_row_idx, row_nnz);
//        std::sort(helper.begin(), helper.end());
//    }
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_SORT_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void is_sorted_by_column_index(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Fbcsr<ValueType, IndexType> *const to_check,
    bool *const is_sorted)
{
    const auto row_ptrs = to_check->get_const_row_ptrs();
    const auto col_idxs = to_check->get_const_col_idxs();
    const auto size = to_check->get_size();
    const int bs = to_check->get_block_size();
    const size_type nbrows = gko::blockutils::getNumFixedBlocks(bs, size[0]);

    for (size_type i = 0; i < nbrows; ++i) {
        for (auto idx = row_ptrs[i] + 1; idx < row_ptrs[i + 1]; ++idx) {
            if (col_idxs[idx - 1] > col_idxs[idx]) {
                *is_sorted = false;
                return;
            }
        }
    }
    *is_sorted = true;
    return;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_IS_SORTED_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void extract_diagonal(std::shared_ptr<const ReferenceExecutor> exec,
                      const matrix::Fbcsr<ValueType, IndexType> *const orig,
                      matrix::Diagonal<ValueType> *const diag)
{
    const auto row_ptrs = orig->get_const_row_ptrs();
    const auto col_idxs = orig->get_const_col_idxs();
    const auto values = orig->get_const_values();
    const int bs = orig->get_block_size();
    const size_type diag_size = diag->get_size()[0];
    const size_type nbrows =
        gko::blockutils::getNumFixedBlocks(bs, orig->get_size()[0]);
    auto diag_values = diag->get_values();
    assert(diag_size == orig->get_size()[0]);

    const gko::blockutils::DenseBlocksView<const ValueType, IndexType> vblocks(
        values, bs, bs);

    for (size_type ibrow = 0; ibrow < nbrows; ++ibrow) {
        for (size_type idx = row_ptrs[ibrow]; idx < row_ptrs[ibrow + 1];
             ++idx) {
            if (col_idxs[idx] == ibrow) {
                for (int ib = 0; ib < bs; ib++)
                    diag_values[ibrow * bs + ib] = vblocks(idx, ib, ib);
                break;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_EXTRACT_DIAGONAL);


}  // namespace fbcsr
}  // namespace reference
}  // namespace kernels
}  // namespace gko
