/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include <ginkgo/core/matrix/csr.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/components/absolute_array.hpp"
#include "core/components/fill_array.hpp"
#include "core/components/prefix_sum.hpp"
#include "core/matrix/csr_kernels.hpp"


namespace gko {
namespace matrix {
namespace csr {
namespace {


GKO_REGISTER_OPERATION(spmv, csr::spmv);
GKO_REGISTER_OPERATION(advanced_spmv, csr::advanced_spmv);
GKO_REGISTER_OPERATION(spgemm, csr::spgemm);
GKO_REGISTER_OPERATION(advanced_spgemm, csr::advanced_spgemm);
GKO_REGISTER_OPERATION(spgeam, csr::spgeam);
GKO_REGISTER_OPERATION(convert_to_coo, csr::convert_to_coo);
GKO_REGISTER_OPERATION(convert_to_dense, csr::convert_to_dense);
GKO_REGISTER_OPERATION(convert_to_sellp, csr::convert_to_sellp);
GKO_REGISTER_OPERATION(calculate_nonzeros_per_row_in_span,
                       csr::calculate_nonzeros_per_row_in_span);
GKO_REGISTER_OPERATION(block_approx, csr::block_approx);
GKO_REGISTER_OPERATION(compute_sub_matrix, csr::compute_sub_matrix);
GKO_REGISTER_OPERATION(calculate_total_cols, csr::calculate_total_cols);
GKO_REGISTER_OPERATION(convert_to_ell, csr::convert_to_ell);
GKO_REGISTER_OPERATION(convert_to_hybrid, csr::convert_to_hybrid);
GKO_REGISTER_OPERATION(transpose, csr::transpose);
GKO_REGISTER_OPERATION(conj_transpose, csr::conj_transpose);
GKO_REGISTER_OPERATION(inv_symm_permute, csr::inv_symm_permute);
GKO_REGISTER_OPERATION(row_permute, csr::row_permute);
GKO_REGISTER_OPERATION(inverse_row_permute, csr::inverse_row_permute);
GKO_REGISTER_OPERATION(inverse_column_permute, csr::inverse_column_permute);
GKO_REGISTER_OPERATION(invert_permutation, csr::invert_permutation);
GKO_REGISTER_OPERATION(calculate_max_nnz_per_row,
                       csr::calculate_max_nnz_per_row);
GKO_REGISTER_OPERATION(calculate_nonzeros_per_row,
                       csr::calculate_nonzeros_per_row);
GKO_REGISTER_OPERATION(sort_by_column_index, csr::sort_by_column_index);
GKO_REGISTER_OPERATION(is_sorted_by_column_index,
                       csr::is_sorted_by_column_index);
GKO_REGISTER_OPERATION(extract_diagonal, csr::extract_diagonal);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(inplace_absolute_array,
                       components::inplace_absolute_array);
GKO_REGISTER_OPERATION(outplace_absolute_array,
                       components::outplace_absolute_array);


}  // anonymous namespace
}  // namespace csr


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::apply_impl(const LinOp *b, LinOp *x) const
{
    using ComplexDense = Dense<to_complex<ValueType>>;
    using TCsr = Csr<ValueType, IndexType>;
    if (auto b_csr = dynamic_cast<const TCsr *>(b)) {
        // if b is a CSR matrix, we compute a SpGeMM
        auto x_csr = as<TCsr>(x);
        this->get_executor()->run(csr::make_spgemm(this, b_csr, x_csr));
    } else {
        precision_dispatch_real_complex<ValueType>(
            [this](auto dense_b, auto dense_x) {
                this->get_executor()->run(
                    csr::make_spmv(this, dense_b, dense_x));
            },
            b, x);
    }
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                           const LinOp *beta, LinOp *x) const
{
    using ComplexDense = Dense<to_complex<ValueType>>;
    using RealDense = Dense<remove_complex<ValueType>>;
    using TCsr = Csr<ValueType, IndexType>;
    if (auto b_csr = dynamic_cast<const TCsr *>(b)) {
        // if b is a CSR matrix, we compute a SpGeMM
        auto x_csr = as<TCsr>(x);
        auto x_copy = x_csr->clone();
        this->get_executor()->run(csr::make_advanced_spgemm(
            as<Dense<ValueType>>(alpha), this, b_csr,
            as<Dense<ValueType>>(beta), x_copy.get(), x_csr));
    } else if (dynamic_cast<const Identity<ValueType> *>(b)) {
        // if b is an identity matrix, we compute an SpGEAM
        auto x_csr = as<TCsr>(x);
        auto x_copy = x_csr->clone();
        this->get_executor()->run(
            csr::make_spgeam(as<Dense<ValueType>>(alpha), this,
                             as<Dense<ValueType>>(beta), lend(x_copy), x_csr));
    } else {
        precision_dispatch_real_complex<ValueType>(
            [this](auto dense_alpha, auto dense_b, auto dense_beta,
                   auto dense_x) {
                this->get_executor()->run(csr::make_advanced_spmv(
                    dense_alpha, this, dense_b, dense_beta, dense_x));
            },
            alpha, b, beta, x);
    }
}


template <typename ValueType, typename IndexType>
std::vector<std::unique_ptr<SubMatrix<Csr<ValueType, IndexType>>>>
Csr<ValueType, IndexType>::get_block_approx(
    const Array<size_type> &block_sizes_in,
    const Overlap<size_type> &block_overlaps_in,
    const Array<size_type> &permutation) const
{
    auto exec = this->get_executor();
    auto bsizes_in = Array<size_type>{exec->get_master()};
    bsizes_in = block_sizes_in;
    auto num_blocks = block_sizes_in.get_num_elems();
    std::vector<std::unique_ptr<SubMatrix<Csr<ValueType, IndexType>>>>
        block_mtxs;
    size_type offset = 0;
    for (size_type b = 0; b < num_blocks; ++b) {
        auto bsize = bsizes_in.get_const_data()[b];
        auto mspan = gko::span{offset, offset + bsize};
        auto unidir = block_overlaps_in.get_unidirectional_array()[b];
        auto overlap = block_overlaps_in.get_overlaps()[b];
        auto st_overlap = block_overlaps_in.get_overlap_at_start_array()[b];
        if (block_overlaps_in.get_num_elems() > 0) {
            auto overlap_spans = calculate_overlap_row_and_col_spans(
                this->get_size(), mspan, mspan, unidir, overlap, st_overlap);
            block_mtxs.emplace_back(
                SubMatrix<Csr<ValueType, IndexType>>::create(
                    exec, this, mspan, mspan, std::get<0>(overlap_spans),
                    std::get<1>(overlap_spans)));
        } else {
            block_mtxs.emplace_back(
                SubMatrix<Csr<ValueType, IndexType>>::create(exec, this, mspan,
                                                             mspan));
        }
        offset += bsize;
    }
    return block_mtxs;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Csr<ValueType, IndexType>>
Csr<ValueType, IndexType>::get_submatrix(const gko::span &row_span,
                                         const gko::span &column_span) const
{
    using Mat = Csr<ValueType, IndexType>;
    auto exec = this->get_executor();
    auto sub_mat_size = gko::dim<2>(row_span.length(), column_span.length());
    Array<size_type> row_nnz(exec, row_span.length());
    row_nnz.fill(size_type(0));
    exec->run(csr::make_calculate_nonzeros_per_row_in_span(
        this, row_span, column_span, &row_nnz));
    auto sub_mat_nnz = row_nnz.reduce();
    auto sub_mat =
        Mat::create(exec, sub_mat_size, sub_mat_nnz, this->get_strategy());
    exec->run(csr::make_compute_sub_matrix(this, &row_nnz, row_span,
                                           column_span, sub_mat.get()));
    return sub_mat;
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::convert_to(
    Csr<next_precision<ValueType>, IndexType> *result) const
{
    result->values_ = this->values_;
    result->col_idxs_ = this->col_idxs_;
    result->row_ptrs_ = this->row_ptrs_;
    result->set_size(this->get_size());
    convert_strategy_helper(result);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::move_to(
    Csr<next_precision<ValueType>, IndexType> *result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::convert_to(
    Coo<ValueType, IndexType> *result) const
{
    auto exec = this->get_executor();
    auto tmp = Coo<ValueType, IndexType>::create(
        exec, this->get_size(), this->get_num_stored_elements());
    tmp->values_ = this->values_;
    tmp->col_idxs_ = this->col_idxs_;
    exec->run(csr::make_convert_to_coo(this, tmp.get()));
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::move_to(Coo<ValueType, IndexType> *result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::convert_to(Dense<ValueType> *result) const
{
    auto exec = this->get_executor();
    auto tmp = Dense<ValueType>::create(exec, this->get_size());
    exec->run(csr::make_convert_to_dense(this, tmp.get()));
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::move_to(Dense<ValueType> *result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::convert_to(
    Hybrid<ValueType, IndexType> *result) const
{
    auto exec = this->get_executor();
    Array<size_type> row_nnz(exec, this->get_size()[0]);
    exec->run(csr::make_calculate_nonzeros_per_row(this, &row_nnz));
    size_type ell_lim = zero<size_type>();
    size_type coo_lim = zero<size_type>();
    result->get_strategy()->compute_hybrid_config(row_nnz, &ell_lim, &coo_lim);
    const auto max_nnz_per_row =
        std::max(result->get_ell_num_stored_elements_per_row(), ell_lim);
    const auto stride = std::max(result->get_ell_stride(), this->get_size()[0]);
    const auto coo_nnz =
        std::max(result->get_coo_num_stored_elements(), coo_lim);
    auto tmp = Hybrid<ValueType, IndexType>::create(
        exec, this->get_size(), max_nnz_per_row, stride, coo_nnz,
        result->get_strategy());
    exec->run(csr::make_convert_to_hybrid(this, tmp.get()));
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::move_to(Hybrid<ValueType, IndexType> *result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::convert_to(
    Sellp<ValueType, IndexType> *result) const
{
    auto exec = this->get_executor();
    const auto stride_factor = (result->get_stride_factor() == 0)
                                   ? default_stride_factor
                                   : result->get_stride_factor();
    const auto slice_size = (result->get_slice_size() == 0)
                                ? default_slice_size
                                : result->get_slice_size();
    size_type total_cols = 0;
    exec->run(csr::make_calculate_total_cols(this, &total_cols, stride_factor,
                                             slice_size));
    auto tmp = Sellp<ValueType, IndexType>::create(
        exec, this->get_size(), slice_size, stride_factor, total_cols);
    exec->run(csr::make_convert_to_sellp(this, tmp.get()));
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::move_to(Sellp<ValueType, IndexType> *result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::convert_to(
    SparsityCsr<ValueType, IndexType> *result) const
{
    auto exec = this->get_executor();
    auto tmp = SparsityCsr<ValueType, IndexType>::create(
        exec, this->get_size(), this->get_num_stored_elements());
    tmp->col_idxs_ = this->col_idxs_;
    tmp->row_ptrs_ = this->row_ptrs_;
    if (result->value_.get_data()) {
        tmp->value_ = result->value_;
    } else {
        tmp->value_ = gko::Array<ValueType>(exec, {one<ValueType>()});
    }
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::move_to(
    SparsityCsr<ValueType, IndexType> *result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::convert_to(
    Ell<ValueType, IndexType> *result) const
{
    auto exec = this->get_executor();
    size_type max_nnz_per_row;
    exec->run(csr::make_calculate_max_nnz_per_row(this, &max_nnz_per_row));
    auto tmp = Ell<ValueType, IndexType>::create(exec, this->get_size(),
                                                 max_nnz_per_row);
    exec->run(csr::make_convert_to_ell(this, tmp.get()));
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::move_to(Ell<ValueType, IndexType> *result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::read(const mat_data &data)
{
    size_type nnz = 0;
    for (const auto &elem : data.nonzeros) {
        nnz += (elem.value != zero<ValueType>());
    }
    auto tmp = Csr::create(this->get_executor()->get_master(), data.size, nnz,
                           this->get_strategy());
    size_type ind = 0;
    size_type cur_ptr = 0;
    tmp->get_row_ptrs()[0] = cur_ptr;
    for (size_type row = 0; row < data.size[0]; ++row) {
        for (; ind < data.nonzeros.size(); ++ind) {
            if (data.nonzeros[ind].row > row) {
                break;
            }
            auto val = data.nonzeros[ind].value;
            if (val != zero<ValueType>()) {
                tmp->get_values()[cur_ptr] = val;
                tmp->get_col_idxs()[cur_ptr] = data.nonzeros[ind].column;
                ++cur_ptr;
            }
        }
        tmp->get_row_ptrs()[row + 1] = cur_ptr;
    }
    tmp->make_srow();
    tmp->move_to(this);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::write(mat_data &data) const
{
    std::unique_ptr<const LinOp> op{};
    const Csr *tmp{};
    if (this->get_executor()->get_master() != this->get_executor()) {
        op = this->clone(this->get_executor()->get_master());
        tmp = static_cast<const Csr *>(op.get());
    } else {
        tmp = this;
    }

    data = {tmp->get_size(), {}};

    for (size_type row = 0; row < tmp->get_size()[0]; ++row) {
        const auto start = tmp->row_ptrs_.get_const_data()[row];
        const auto end = tmp->row_ptrs_.get_const_data()[row + 1];
        for (auto i = start; i < end; ++i) {
            const auto col = tmp->col_idxs_.get_const_data()[i];
            const auto val = tmp->values_.get_const_data()[i];
            data.nonzeros.emplace_back(row, col, val);
        }
    }
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Csr<ValueType, IndexType>::transpose() const
{
    auto exec = this->get_executor();
    auto trans_cpy =
        Csr::create(exec, gko::transpose(this->get_size()),
                    this->get_num_stored_elements(), this->get_strategy());

    exec->run(csr::make_transpose(this, trans_cpy.get()));
    trans_cpy->make_srow();
    return std::move(trans_cpy);
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Csr<ValueType, IndexType>::conj_transpose() const
{
    auto exec = this->get_executor();
    auto trans_cpy =
        Csr::create(exec, gko::transpose(this->get_size()),
                    this->get_num_stored_elements(), this->get_strategy());

    exec->run(csr::make_conj_transpose(this, trans_cpy.get()));
    trans_cpy->make_srow();
    return std::move(trans_cpy);
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Csr<ValueType, IndexType>::permute(
    const Array<IndexType> *permutation_indices) const
{
    GKO_ASSERT_IS_SQUARE_MATRIX(this);
    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[0]);
    auto exec = this->get_executor();
    auto permute_cpy =
        Csr::create(exec, this->get_size(), this->get_num_stored_elements(),
                    this->get_strategy());
    Array<IndexType> inv_permutation(exec, this->get_size()[1]);

    exec->run(csr::make_invert_permutation(
        this->get_size()[1],
        make_temporary_clone(exec, permutation_indices)->get_const_data(),
        inv_permutation.get_data()));
    exec->run(csr::make_inv_symm_permute(inv_permutation.get_const_data(), this,
                                         permute_cpy.get()));
    permute_cpy->make_srow();
    return std::move(permute_cpy);
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Csr<ValueType, IndexType>::inverse_permute(
    const Array<IndexType> *permutation_indices) const
{
    GKO_ASSERT_IS_SQUARE_MATRIX(this);
    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[0]);
    auto exec = this->get_executor();
    auto permute_cpy =
        Csr::create(exec, this->get_size(), this->get_num_stored_elements(),
                    this->get_strategy());

    exec->run(csr::make_inv_symm_permute(
        make_temporary_clone(exec, permutation_indices)->get_const_data(), this,
        permute_cpy.get()));
    permute_cpy->make_srow();
    return std::move(permute_cpy);
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Csr<ValueType, IndexType>::row_permute(
    const Array<IndexType> *permutation_indices) const
{
    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[0]);
    auto exec = this->get_executor();
    auto permute_cpy =
        Csr::create(exec, this->get_size(), this->get_num_stored_elements(),
                    this->get_strategy());

    exec->run(csr::make_row_permute(
        make_temporary_clone(exec, permutation_indices)->get_const_data(), this,
        permute_cpy.get()));
    permute_cpy->make_srow();
    return std::move(permute_cpy);
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Csr<ValueType, IndexType>::column_permute(
    const Array<IndexType> *permutation_indices) const
{
    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[1]);
    auto exec = this->get_executor();
    auto permute_cpy =
        Csr::create(exec, this->get_size(), this->get_num_stored_elements(),
                    this->get_strategy());
    Array<IndexType> inv_permutation(exec, this->get_size()[1]);

    exec->run(csr::make_invert_permutation(
        this->get_size()[1],
        make_temporary_clone(exec, permutation_indices)->get_const_data(),
        inv_permutation.get_data()));
    exec->run(csr::make_inverse_column_permute(inv_permutation.get_const_data(),
                                               this, permute_cpy.get()));
    permute_cpy->make_srow();
    permute_cpy->sort_by_column_index();
    return std::move(permute_cpy);
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Csr<ValueType, IndexType>::inverse_row_permute(
    const Array<IndexType> *permutation_indices) const
{
    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[0]);
    auto exec = this->get_executor();
    auto inverse_permute_cpy =
        Csr::create(exec, this->get_size(), this->get_num_stored_elements(),
                    this->get_strategy());

    exec->run(csr::make_inverse_row_permute(
        make_temporary_clone(exec, permutation_indices)->get_const_data(), this,
        inverse_permute_cpy.get()));
    inverse_permute_cpy->make_srow();
    return std::move(inverse_permute_cpy);
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Csr<ValueType, IndexType>::inverse_column_permute(
    const Array<IndexType> *permutation_indices) const
{
    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[1]);
    auto exec = this->get_executor();
    auto inverse_permute_cpy =
        Csr::create(exec, this->get_size(), this->get_num_stored_elements(),
                    this->get_strategy());

    exec->run(csr::make_inverse_column_permute(
        make_temporary_clone(exec, permutation_indices)->get_const_data(), this,
        inverse_permute_cpy.get()));
    inverse_permute_cpy->make_srow();
    inverse_permute_cpy->sort_by_column_index();
    return std::move(inverse_permute_cpy);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::sort_by_column_index()
{
    auto exec = this->get_executor();
    exec->run(csr::make_sort_by_column_index(this));
}


template <typename ValueType, typename IndexType>
bool Csr<ValueType, IndexType>::is_sorted_by_column_index() const
{
    auto exec = this->get_executor();
    bool is_sorted;
    exec->run(csr::make_is_sorted_by_column_index(this, &is_sorted));
    return is_sorted;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Diagonal<ValueType>>
Csr<ValueType, IndexType>::extract_diagonal() const
{
    auto exec = this->get_executor();

    const auto diag_size = std::min(this->get_size()[0], this->get_size()[1]);
    auto diag = Diagonal<ValueType>::create(exec, diag_size);
    exec->run(csr::make_fill_array(diag->get_values(), diag->get_size()[0],
                                   zero<ValueType>()));
    exec->run(csr::make_extract_diagonal(this, lend(diag)));
    return diag;
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::compute_absolute_inplace()
{
    auto exec = this->get_executor();

    exec->run(csr::make_inplace_absolute_array(
        this->get_values(), this->get_num_stored_elements()));
}


template <typename ValueType, typename IndexType>
std::unique_ptr<typename Csr<ValueType, IndexType>::absolute_type>
Csr<ValueType, IndexType>::compute_absolute() const
{
    auto exec = this->get_executor();

    auto abs_csr = absolute_type::create(exec, this->get_size(),
                                         this->get_num_stored_elements());

    abs_csr->col_idxs_ = col_idxs_;
    abs_csr->row_ptrs_ = row_ptrs_;
    exec->run(csr::make_outplace_absolute_array(this->get_const_values(),
                                                this->get_num_stored_elements(),
                                                abs_csr->get_values()));

    convert_strategy_helper(abs_csr.get());
    return abs_csr;
}


#define GKO_DECLARE_CSR_MATRIX(ValueType, IndexType) \
    class Csr<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_MATRIX);


}  // namespace matrix
}  // namespace gko
