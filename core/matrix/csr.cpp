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

#include <ginkgo/core/matrix/csr.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/index_set.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/fbcsr.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/base/device_matrix_data_kernels.hpp"
#include "core/components/absolute_array_kernels.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/matrix/ell_kernels.hpp"
#include "core/matrix/hybrid_kernels.hpp"
#include "core/matrix/sellp_kernels.hpp"


namespace gko {
namespace matrix {
namespace csr {
namespace {


GKO_REGISTER_OPERATION(spmv, csr::spmv);
GKO_REGISTER_OPERATION(advanced_spmv, csr::advanced_spmv);
GKO_REGISTER_OPERATION(spgemm, csr::spgemm);
GKO_REGISTER_OPERATION(advanced_spgemm, csr::advanced_spgemm);
GKO_REGISTER_OPERATION(spgeam, csr::spgeam);
GKO_REGISTER_OPERATION(convert_idxs_to_ptrs, components::convert_idxs_to_ptrs);
GKO_REGISTER_OPERATION(convert_ptrs_to_idxs, components::convert_ptrs_to_idxs);
GKO_REGISTER_OPERATION(fill_in_dense, csr::fill_in_dense);
GKO_REGISTER_OPERATION(compute_slice_sets, sellp::compute_slice_sets);
GKO_REGISTER_OPERATION(convert_to_sellp, csr::convert_to_sellp);
GKO_REGISTER_OPERATION(compute_max_row_nnz, ell::compute_max_row_nnz);
GKO_REGISTER_OPERATION(convert_to_ell, csr::convert_to_ell);
GKO_REGISTER_OPERATION(convert_to_fbcsr, csr::convert_to_fbcsr);
GKO_REGISTER_OPERATION(compute_hybrid_coo_row_ptrs,
                       hybrid::compute_coo_row_ptrs);
GKO_REGISTER_OPERATION(convert_to_hybrid, csr::convert_to_hybrid);
GKO_REGISTER_OPERATION(calculate_nonzeros_per_row_in_span,
                       csr::calculate_nonzeros_per_row_in_span);
GKO_REGISTER_OPERATION(calculate_nonzeros_per_row_in_index_set,
                       csr::calculate_nonzeros_per_row_in_index_set);
GKO_REGISTER_OPERATION(compute_submatrix, csr::compute_submatrix);
GKO_REGISTER_OPERATION(compute_submatrix_from_index_set,
                       csr::compute_submatrix_from_index_set);
GKO_REGISTER_OPERATION(transpose, csr::transpose);
GKO_REGISTER_OPERATION(conj_transpose, csr::conj_transpose);
GKO_REGISTER_OPERATION(inv_symm_permute, csr::inv_symm_permute);
GKO_REGISTER_OPERATION(row_permute, csr::row_permute);
GKO_REGISTER_OPERATION(inverse_row_permute, csr::inverse_row_permute);
GKO_REGISTER_OPERATION(inverse_column_permute, csr::inverse_column_permute);
GKO_REGISTER_OPERATION(invert_permutation, csr::invert_permutation);
GKO_REGISTER_OPERATION(convert_ptrs_to_sizes,
                       components::convert_ptrs_to_sizes);
GKO_REGISTER_OPERATION(sort_by_column_index, csr::sort_by_column_index);
GKO_REGISTER_OPERATION(is_sorted_by_column_index,
                       csr::is_sorted_by_column_index);
GKO_REGISTER_OPERATION(extract_diagonal, csr::extract_diagonal);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(prefix_sum, components::prefix_sum);
GKO_REGISTER_OPERATION(inplace_absolute_array,
                       components::inplace_absolute_array);
GKO_REGISTER_OPERATION(outplace_absolute_array,
                       components::outplace_absolute_array);
GKO_REGISTER_OPERATION(scale, csr::scale);
GKO_REGISTER_OPERATION(inv_scale, csr::inv_scale);
GKO_REGISTER_OPERATION(add_scaled_identity, csr::add_scaled_identity);
GKO_REGISTER_OPERATION(check_diagonal_entries,
                       csr::check_diagonal_entries_exist);


}  // anonymous namespace
}  // namespace csr


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>::strategy_type::strategy_type(std::string name)
    : name_(name)
{}


template <typename ValueType, typename IndexType>
std::string Csr<ValueType, IndexType>::strategy_type::get_name()
{
    return name_;
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::strategy_type::set_name(std::string name)
{
    name_ = name;
}


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>::classical::classical()
    : strategy_type("classical"), max_length_per_row_(0)
{}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::classical::process(
    const array<index_type>& mtx_row_ptrs, array<index_type>* mtx_srow)
{
    auto host_mtx_exec = mtx_row_ptrs.get_executor()->get_master();
    array<index_type> row_ptrs_host(host_mtx_exec);
    const bool is_mtx_on_host{host_mtx_exec == mtx_row_ptrs.get_executor()};
    const index_type* row_ptrs{};
    if (is_mtx_on_host) {
        row_ptrs = mtx_row_ptrs.get_const_data();
    } else {
        row_ptrs_host = mtx_row_ptrs;
        row_ptrs = row_ptrs_host.get_const_data();
    }
    auto num_rows = mtx_row_ptrs.get_num_elems() - 1;
    max_length_per_row_ = 0;
    for (size_type i = 0; i < num_rows; i++) {
        max_length_per_row_ =
            std::max(max_length_per_row_, row_ptrs[i + 1] - row_ptrs[i]);
    }
}


template <typename ValueType, typename IndexType>
int64_t Csr<ValueType, IndexType>::classical::clac_size(const int64_t nnz)
{
    return 0;
}


template <typename ValueType, typename IndexType>
IndexType Csr<ValueType, IndexType>::classical::get_max_length_per_row() const
    noexcept
{
    return max_length_per_row_;
}


template <typename ValueType, typename IndexType>
std::shared_ptr<typename Csr<ValueType, IndexType>::strategy_type>
Csr<ValueType, IndexType>::classical::copy()
{
    return std::make_shared<classical>();
}


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>::merge_path::merge_path()
    : strategy_type("merge_path")
{}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::merge_path::process(
    const array<index_type>& mtx_row_ptrs, array<index_type>* mtx_srow)
{}


template <typename ValueType, typename IndexType>
int64_t Csr<ValueType, IndexType>::merge_path::clac_size(const int64_t nnz)
{
    return 0;
}


template <typename ValueType, typename IndexType>
std::shared_ptr<typename Csr<ValueType, IndexType>::strategy_type>
Csr<ValueType, IndexType>::merge_path::copy()
{
    return std::make_shared<merge_path>();
}


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>::cusparse::cusparse() : strategy_type("cusparse")
{}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::cusparse::process(
    const array<index_type>& mtx_row_ptrs, array<index_type>* mtx_srow)
{}


template <typename ValueType, typename IndexType>
int64_t Csr<ValueType, IndexType>::cusparse::clac_size(const int64_t nnz)
{
    return 0;
}


template <typename ValueType, typename IndexType>
std::shared_ptr<typename Csr<ValueType, IndexType>::strategy_type>
Csr<ValueType, IndexType>::cusparse::copy()
{
    return std::make_shared<cusparse>();
}


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>::sparselib::sparselib() : strategy_type("sparselib")
{}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::sparselib::process(
    const array<index_type>& mtx_row_ptrs, array<index_type>* mtx_srow)
{}


template <typename ValueType, typename IndexType>
int64_t Csr<ValueType, IndexType>::sparselib::clac_size(const int64_t nnz)
{
    return 0;
}


template <typename ValueType, typename IndexType>
std::shared_ptr<typename Csr<ValueType, IndexType>::strategy_type>
Csr<ValueType, IndexType>::sparselib::copy()
{
    return std::make_shared<sparselib>();
}


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>::load_balance::load_balance()
    : load_balance(
          std::move(gko::CudaExecutor::create(0, gko::OmpExecutor::create())))
{}


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>::load_balance::load_balance(
    std::shared_ptr<const CudaExecutor> exec)
    : load_balance(exec->get_num_warps(), exec->get_warp_size())
{}


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>::load_balance::load_balance(
    std::shared_ptr<const HipExecutor> exec)
    : load_balance(exec->get_num_warps(), exec->get_warp_size(), false)
{}


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>::load_balance::load_balance(
    std::shared_ptr<const DpcppExecutor> exec)
    : load_balance(exec->get_num_computing_units() * 7, 32, false, "intel")
{}


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>::load_balance::load_balance(int64_t nwarps,
                                                      int warp_size,
                                                      bool cuda_strategy,
                                                      std::string strategy_name)
    : strategy_type("load_balance"),
      nwarps_(nwarps),
      warp_size_(warp_size),
      cuda_strategy_(cuda_strategy),
      strategy_name_(strategy_name)
{}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::load_balance::process(
    const array<index_type>& mtx_row_ptrs, array<index_type>* mtx_srow)
{
    auto nwarps = mtx_srow->get_num_elems();

    if (nwarps > 0) {
        auto host_srow_exec = mtx_srow->get_executor()->get_master();
        auto host_mtx_exec = mtx_row_ptrs.get_executor()->get_master();
        const bool is_srow_on_host{host_srow_exec == mtx_srow->get_executor()};
        const bool is_mtx_on_host{host_mtx_exec == mtx_row_ptrs.get_executor()};
        array<index_type> row_ptrs_host(host_mtx_exec);
        array<index_type> srow_host(host_srow_exec);
        const index_type* row_ptrs{};
        index_type* srow{};
        if (is_srow_on_host) {
            srow = mtx_srow->get_data();
        } else {
            srow_host = *mtx_srow;
            srow = srow_host.get_data();
        }
        if (is_mtx_on_host) {
            row_ptrs = mtx_row_ptrs.get_const_data();
        } else {
            row_ptrs_host = mtx_row_ptrs;
            row_ptrs = row_ptrs_host.get_const_data();
        }
        for (size_type i = 0; i < nwarps; i++) {
            srow[i] = 0;
        }
        const auto num_rows = mtx_row_ptrs.get_num_elems() - 1;
        const auto num_elems = row_ptrs[num_rows];
        const auto bucket_divider =
            num_elems > 0 ? ceildiv(num_elems, warp_size_) : 1;
        for (size_type i = 0; i < num_rows; i++) {
            auto bucket =
                ceildiv((ceildiv(row_ptrs[i + 1], warp_size_) * nwarps),
                        bucket_divider);
            if (bucket < nwarps) {
                srow[bucket]++;
            }
        }
        // find starting row for thread i
        for (size_type i = 1; i < nwarps; i++) {
            srow[i] += srow[i - 1];
        }
        if (!is_srow_on_host) {
            *mtx_srow = srow_host;
        }
    }
}


template <typename ValueType, typename IndexType>
int64_t Csr<ValueType, IndexType>::load_balance::clac_size(const int64_t nnz)
{
    if (warp_size_ > 0) {
        int multiple = 8;
        if (nnz >= static_cast<int64_t>(2e8)) {
            multiple = 2048;
        } else if (nnz >= static_cast<int64_t>(2e7)) {
            multiple = 512;
        } else if (nnz >= static_cast<int64_t>(2e6)) {
            multiple = 128;
        } else if (nnz >= static_cast<int64_t>(2e5)) {
            multiple = 32;
        }
        if (strategy_name_ == "intel") {
            multiple = 8;
            if (nnz >= static_cast<int64_t>(2e8)) {
                multiple = 256;
            } else if (nnz >= static_cast<int64_t>(2e7)) {
                multiple = 32;
            }
        }
#if GINKGO_HIP_PLATFORM_HCC
        if (!cuda_strategy_) {
            multiple = 8;
            if (nnz >= static_cast<int64_t>(1e7)) {
                multiple = 64;
            } else if (nnz >= static_cast<int64_t>(1e6)) {
                multiple = 16;
            }
        }
#endif  // GINKGO_HIP_PLATFORM_HCC

        auto nwarps = nwarps_ * multiple;
        return min(ceildiv(nnz, warp_size_), nwarps);
    } else {
        return 0;
    }
}


template <typename ValueType, typename IndexType>
std::shared_ptr<typename Csr<ValueType, IndexType>::strategy_type>
Csr<ValueType, IndexType>::load_balance::copy()
{
    return std::make_shared<load_balance>(nwarps_, warp_size_, cuda_strategy_,
                                          strategy_name_);
}


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>::automatical::automatical()
    : automatical(
          std::move(gko::CudaExecutor::create(0, gko::OmpExecutor::create())))
{}


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>::automatical::automatical(
    std::shared_ptr<const CudaExecutor> exec)
    : automatical(exec->get_num_warps(), exec->get_warp_size())
{}


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>::automatical::automatical(
    std::shared_ptr<const HipExecutor> exec)
    : automatical(exec->get_num_warps(), exec->get_warp_size(), false)
{}


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>::automatical::automatical(
    std::shared_ptr<const DpcppExecutor> exec)
    : automatical(exec->get_num_computing_units() * 7, 32, false, "intel")
{}


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>::automatical::automatical(int64_t nwarps,
                                                    int warp_size,
                                                    bool cuda_strategy,
                                                    std::string strategy_name)
    : strategy_type("automatical"),
      nwarps_(nwarps),
      warp_size_(warp_size),
      cuda_strategy_(cuda_strategy),
      strategy_name_(strategy_name),
      max_length_per_row_(0)
{}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::automatical::process(
    const array<index_type>& mtx_row_ptrs, array<index_type>* mtx_srow)
{
    // if the number of stored elements is larger than <nnz_limit> or
    // the maximum number of stored elements per row is larger than
    // <row_len_limit>, use load_balance otherwise use classical
    index_type nnz_limit = nvidia_nnz_limit;
    index_type row_len_limit = nvidia_row_len_limit;
    if (strategy_name_ == "intel") {
        nnz_limit = intel_nnz_limit;
        row_len_limit = intel_row_len_limit;
    }
#if GINKGO_HIP_PLATFORM_HCC
    if (!cuda_strategy_) {
        nnz_limit = amd_nnz_limit;
        row_len_limit = amd_row_len_limit;
    }
#endif  // GINKGO_HIP_PLATFORM_HCC
    auto host_mtx_exec = mtx_row_ptrs.get_executor()->get_master();
    const bool is_mtx_on_host{host_mtx_exec == mtx_row_ptrs.get_executor()};
    array<index_type> row_ptrs_host(host_mtx_exec);
    const index_type* row_ptrs{};
    if (is_mtx_on_host) {
        row_ptrs = mtx_row_ptrs.get_const_data();
    } else {
        row_ptrs_host = mtx_row_ptrs;
        row_ptrs = row_ptrs_host.get_const_data();
    }
    const auto num_rows = mtx_row_ptrs.get_num_elems() - 1;
    if (row_ptrs[num_rows] > nnz_limit) {
        load_balance actual_strategy(nwarps_, warp_size_, cuda_strategy_,
                                     strategy_name_);
        if (is_mtx_on_host) {
            actual_strategy.process(mtx_row_ptrs, mtx_srow);
        } else {
            actual_strategy.process(row_ptrs_host, mtx_srow);
        }
        this->set_name(actual_strategy.get_name());
    } else {
        index_type maxnum = 0;
        for (size_type i = 0; i < num_rows; i++) {
            maxnum = std::max(maxnum, row_ptrs[i + 1] - row_ptrs[i]);
        }
        if (maxnum > row_len_limit) {
            load_balance actual_strategy(nwarps_, warp_size_, cuda_strategy_,
                                         strategy_name_);
            if (is_mtx_on_host) {
                actual_strategy.process(mtx_row_ptrs, mtx_srow);
            } else {
                actual_strategy.process(row_ptrs_host, mtx_srow);
            }
            this->set_name(actual_strategy.get_name());
        } else {
            classical actual_strategy;
            if (is_mtx_on_host) {
                actual_strategy.process(mtx_row_ptrs, mtx_srow);
                max_length_per_row_ = actual_strategy.get_max_length_per_row();
            } else {
                actual_strategy.process(row_ptrs_host, mtx_srow);
                max_length_per_row_ = actual_strategy.get_max_length_per_row();
            }
            this->set_name(actual_strategy.get_name());
        }
    }
}


template <typename ValueType, typename IndexType>
int64_t Csr<ValueType, IndexType>::automatical::clac_size(const int64_t nnz)
{
    return std::make_shared<load_balance>(nwarps_, warp_size_, cuda_strategy_,
                                          strategy_name_)
        ->clac_size(nnz);
}


template <typename ValueType, typename IndexType>
IndexType Csr<ValueType, IndexType>::automatical::get_max_length_per_row() const
    noexcept
{
    return max_length_per_row_;
}


template <typename ValueType, typename IndexType>
std::shared_ptr<typename Csr<ValueType, IndexType>::strategy_type>
Csr<ValueType, IndexType>::automatical::copy()
{
    return std::make_shared<automatical>(nwarps_, warp_size_, cuda_strategy_,
                                         strategy_name_);
}


template <typename ValueType, typename IndexType>
ValueType* Csr<ValueType, IndexType>::get_values() noexcept
{
    return values_.get_data();
}


template <typename ValueType, typename IndexType>
const ValueType* Csr<ValueType, IndexType>::get_const_values() const noexcept
{
    return values_.get_const_data();
}

template <typename ValueType, typename IndexType>
IndexType* Csr<ValueType, IndexType>::get_col_idxs() noexcept
{
    return col_idxs_.get_data();
}


template <typename ValueType, typename IndexType>
const IndexType* Csr<ValueType, IndexType>::get_const_col_idxs() const noexcept
{
    return col_idxs_.get_const_data();
}


template <typename ValueType, typename IndexType>
IndexType* Csr<ValueType, IndexType>::get_row_ptrs() noexcept
{
    return row_ptrs_.get_data();
}


template <typename ValueType, typename IndexType>
const IndexType* Csr<ValueType, IndexType>::get_const_row_ptrs() const noexcept
{
    return row_ptrs_.get_const_data();
}


template <typename ValueType, typename IndexType>
IndexType* Csr<ValueType, IndexType>::get_srow() noexcept
{
    return srow_.get_data();
}


template <typename ValueType, typename IndexType>
const IndexType* Csr<ValueType, IndexType>::get_const_srow() const noexcept
{
    return srow_.get_const_data();
}


template <typename ValueType, typename IndexType>
size_type Csr<ValueType, IndexType>::get_num_srow_elements() const noexcept
{
    return srow_.get_num_elems();
}


template <typename ValueType, typename IndexType>
size_type Csr<ValueType, IndexType>::get_num_stored_elements() const noexcept
{
    return values_.get_num_elems();
}


template <typename ValueType, typename IndexType>
std::shared_ptr<typename Csr<ValueType, IndexType>::strategy_type>
Csr<ValueType, IndexType>::get_strategy() const noexcept
{
    return strategy_;
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::set_strategy(
    std::shared_ptr<strategy_type> strategy)
{
    strategy_ = std::move(strategy->copy());
    this->make_srow();
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::scale(const LinOp* alpha)
{
    auto exec = this->get_executor();
    GKO_ASSERT_EQUAL_DIMENSIONS(alpha, dim<2>(1, 1));
    this->scale_impl(make_temporary_clone(exec, alpha).get());
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::inv_scale(const LinOp* alpha)
{
    auto exec = this->get_executor();
    GKO_ASSERT_EQUAL_DIMENSIONS(alpha, dim<2>(1, 1));
    this->inv_scale_impl(make_temporary_clone(exec, alpha).get());
}


template <typename ValueType, typename IndexType>
std::unique_ptr<const Csr<ValueType, IndexType>>
Csr<ValueType, IndexType>::create_const(
    std::shared_ptr<const Executor> exec, const dim<2>& size,
    gko::detail::const_array_view<ValueType>&& values,
    gko::detail::const_array_view<IndexType>&& col_idxs,
    gko::detail::const_array_view<IndexType>&& row_ptrs,
    std::shared_ptr<strategy_type> strategy)
{
    // cast const-ness away, but return a const object afterwards,
    // so we can ensure that no modifications take place.
    return std::unique_ptr<const Csr>(
        new Csr{exec, size, gko::detail::array_const_cast(std::move(values)),
                gko::detail::array_const_cast(std::move(col_idxs)),
                gko::detail::array_const_cast(std::move(row_ptrs)), strategy});
}


template <typename ValueType, typename IndexType>
std::unique_ptr<const Csr<ValueType, IndexType>>
Csr<ValueType, IndexType>::create_const(
    std::shared_ptr<const Executor> exec, const dim<2>& size,
    gko::detail::const_array_view<ValueType>&& values,
    gko::detail::const_array_view<IndexType>&& col_idxs,
    gko::detail::const_array_view<IndexType>&& row_ptrs)
{
    return Csr::create_const(exec, size, std::move(values), std::move(col_idxs),
                             std::move(row_ptrs),
                             Csr::make_default_strategy(exec));
}


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>::Csr(std::shared_ptr<const Executor> exec,
                               std::shared_ptr<strategy_type> strategy)
    : Csr(std::move(exec), dim<2>{}, {}, std::move(strategy))
{}


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>::Csr(std::shared_ptr<const Executor> exec,
                               const dim<2>& size, size_type num_nonzeros,
                               std::shared_ptr<strategy_type> strategy)
    : EnableLinOp<Csr>(exec, size),
      values_(exec, num_nonzeros),
      col_idxs_(exec, num_nonzeros),
      row_ptrs_(exec, size[0] + 1),
      srow_(exec, strategy->clac_size(num_nonzeros)),
      strategy_(strategy->copy())
{
    row_ptrs_.fill(0);
    this->make_srow();
}


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>::Csr(std::shared_ptr<const Executor> exec,
                               const dim<2>& size, size_type num_nonzeros)
    : Csr{exec, size, num_nonzeros, Csr::make_default_strategy(exec)}
{}


template <typename ValueType, typename IndexType>
// TODO: This provides some more sane settings. Please fix this!
std::shared_ptr<typename Csr<ValueType, IndexType>::strategy_type>
Csr<ValueType, IndexType>::make_default_strategy(
    std::shared_ptr<const Executor> exec)
{
    auto cuda_exec = std::dynamic_pointer_cast<const CudaExecutor>(exec);
    auto hip_exec = std::dynamic_pointer_cast<const HipExecutor>(exec);
    auto dpcpp_exec = std::dynamic_pointer_cast<const DpcppExecutor>(exec);
    std::shared_ptr<strategy_type> new_strategy;
    if (cuda_exec) {
        new_strategy = std::make_shared<automatical>(cuda_exec);
    } else if (hip_exec) {
        new_strategy = std::make_shared<automatical>(hip_exec);
    } else if (dpcpp_exec) {
        new_strategy = std::make_shared<automatical>(dpcpp_exec);
    } else {
        new_strategy = std::make_shared<classical>();
    }
    return new_strategy;
}


// TODO clean this up as soon as we improve strategy_type
template <typename ValueType, typename IndexType>
template <typename CsrType>
void Csr<ValueType, IndexType>::convert_strategy_helper(CsrType* result) const
{
    auto strat = this->get_strategy().get();
    std::shared_ptr<typename CsrType::strategy_type> new_strat;
    if (dynamic_cast<classical*>(strat)) {
        new_strat = std::make_shared<typename CsrType::classical>();
    } else if (dynamic_cast<merge_path*>(strat)) {
        new_strat = std::make_shared<typename CsrType::merge_path>();
    } else if (dynamic_cast<cusparse*>(strat)) {
        new_strat = std::make_shared<typename CsrType::cusparse>();
    } else if (dynamic_cast<sparselib*>(strat)) {
        new_strat = std::make_shared<typename CsrType::sparselib>();
    } else {
        auto rexec = result->get_executor();
        auto cuda_exec = std::dynamic_pointer_cast<const CudaExecutor>(rexec);
        auto hip_exec = std::dynamic_pointer_cast<const HipExecutor>(rexec);
        auto dpcpp_exec = std::dynamic_pointer_cast<const DpcppExecutor>(rexec);
        auto lb = dynamic_cast<load_balance*>(strat);
        if (cuda_exec) {
            if (lb) {
                new_strat =
                    std::make_shared<typename CsrType::load_balance>(cuda_exec);
            } else {
                new_strat =
                    std::make_shared<typename CsrType::automatical>(cuda_exec);
            }
        } else if (hip_exec) {
            if (lb) {
                new_strat =
                    std::make_shared<typename CsrType::load_balance>(hip_exec);
            } else {
                new_strat =
                    std::make_shared<typename CsrType::automatical>(hip_exec);
            }
        } else if (dpcpp_exec) {
            if (lb) {
                new_strat = std::make_shared<typename CsrType::load_balance>(
                    dpcpp_exec);
            } else {
                new_strat =
                    std::make_shared<typename CsrType::automatical>(dpcpp_exec);
            }
        } else {
            // Try to preserve this executor's configuration
            auto this_cuda_exec = std::dynamic_pointer_cast<const CudaExecutor>(
                this->get_executor());
            auto this_hip_exec = std::dynamic_pointer_cast<const HipExecutor>(
                this->get_executor());
            auto this_dpcpp_exec =
                std::dynamic_pointer_cast<const DpcppExecutor>(
                    this->get_executor());
            if (this_cuda_exec) {
                if (lb) {
                    new_strat =
                        std::make_shared<typename CsrType::load_balance>(
                            this_cuda_exec);
                } else {
                    new_strat = std::make_shared<typename CsrType::automatical>(
                        this_cuda_exec);
                }
            } else if (this_hip_exec) {
                if (lb) {
                    new_strat =
                        std::make_shared<typename CsrType::load_balance>(
                            this_hip_exec);
                } else {
                    new_strat = std::make_shared<typename CsrType::automatical>(
                        this_hip_exec);
                }
            } else if (this_dpcpp_exec) {
                if (lb) {
                    new_strat =
                        std::make_shared<typename CsrType::load_balance>(
                            this_dpcpp_exec);
                } else {
                    new_strat = std::make_shared<typename CsrType::automatical>(
                        this_dpcpp_exec);
                }
            } else {
                // FIXME: this changes strategies.
                // We had a load balance or automatical strategy from a non
                // HIP or Cuda executor and are moving to a non HIP or Cuda
                // executor.
                new_strat = std::make_shared<typename CsrType::classical>();
            }
        }
    }
    result->set_strategy(new_strat);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::make_srow()
{
    srow_.resize_and_reset(strategy_->clac_size(values_.get_num_elems()));
    strategy_->process(row_ptrs_, &srow_);
}


namespace detail {


/**
 * When strategy is load_balance or automatical, rebuild the strategy
 * according to executor's property.
 *
 * @param result  the csr matrix.
 */
template <typename ValueType, typename IndexType>
void strategy_rebuild_helper(Csr<ValueType, IndexType>* result)
{
    using load_balance = typename Csr<ValueType, IndexType>::load_balance;
    using automatical = typename Csr<ValueType, IndexType>::automatical;
    auto strategy = result->get_strategy();
    auto executor = result->get_executor();
    if (std::dynamic_pointer_cast<load_balance>(strategy)) {
        if (auto exec =
                std::dynamic_pointer_cast<const HipExecutor>(executor)) {
            result->set_strategy(std::make_shared<load_balance>(exec));
        } else if (auto exec = std::dynamic_pointer_cast<const CudaExecutor>(
                       executor)) {
            result->set_strategy(std::make_shared<load_balance>(exec));
        }
    } else if (std::dynamic_pointer_cast<automatical>(strategy)) {
        if (auto exec =
                std::dynamic_pointer_cast<const HipExecutor>(executor)) {
            result->set_strategy(std::make_shared<automatical>(exec));
        } else if (auto exec = std::dynamic_pointer_cast<const CudaExecutor>(
                       executor)) {
            result->set_strategy(std::make_shared<automatical>(exec));
        }
    }
}


}  // namespace detail


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>& Csr<ValueType, IndexType>::operator=(
    const Csr<ValueType, IndexType>& other)
{
    if (&other != this) {
        EnableLinOp<Csr>::operator=(other);
        // NOTE: as soon as strategies are improved, this can be reverted
        values_ = other.values_;
        col_idxs_ = other.col_idxs_;
        row_ptrs_ = other.row_ptrs_;
        srow_ = other.srow_;
        if (this->get_executor() != other.get_executor()) {
            other.convert_strategy_helper(this);
        } else {
            this->set_strategy(other.get_strategy()->copy());
        }
        // END NOTE
    }
    return *this;
}


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>& Csr<ValueType, IndexType>::operator=(
    Csr<ValueType, IndexType>&& other)
{
    if (&other != this) {
        EnableLinOp<Csr>::operator=(std::move(other));
        values_ = std::move(other.values_);
        col_idxs_ = std::move(other.col_idxs_);
        row_ptrs_ = std::move(other.row_ptrs_);
        srow_ = std::move(other.srow_);
        strategy_ = other.strategy_;
        if (this->get_executor() != other.get_executor()) {
            detail::strategy_rebuild_helper(this);
        }
        // restore other invariant
        other.row_ptrs_.resize_and_reset(1);
        other.row_ptrs_.fill(0);
        other.make_srow();
    }
    return *this;
}


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>::Csr(const Csr<ValueType, IndexType>& other)
    : Csr{other.get_executor()}
{
    *this = other;
}


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>::Csr(Csr<ValueType, IndexType>&& other)
    : Csr{other.get_executor()}
{
    *this = std::move(other);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::apply_impl(const LinOp* b, LinOp* x) const
{
    using ComplexDense = Dense<to_complex<ValueType>>;
    using TCsr = Csr<ValueType, IndexType>;
    if (auto b_csr = dynamic_cast<const TCsr*>(b)) {
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
void Csr<ValueType, IndexType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                           const LinOp* beta, LinOp* x) const
{
    using ComplexDense = Dense<to_complex<ValueType>>;
    using RealDense = Dense<remove_complex<ValueType>>;
    using TCsr = Csr<ValueType, IndexType>;
    if (auto b_csr = dynamic_cast<const TCsr*>(b)) {
        // if b is a CSR matrix, we compute a SpGeMM
        auto x_csr = as<TCsr>(x);
        auto x_copy = x_csr->clone();
        this->get_executor()->run(csr::make_advanced_spgemm(
            as<Dense<ValueType>>(alpha), this, b_csr,
            as<Dense<ValueType>>(beta), x_copy.get(), x_csr));
    } else if (dynamic_cast<const Identity<ValueType>*>(b)) {
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
void Csr<ValueType, IndexType>::convert_to(
    Csr<next_precision<ValueType>, IndexType>* result) const
{
    result->values_ = this->values_;
    result->col_idxs_ = this->col_idxs_;
    result->row_ptrs_ = this->row_ptrs_;
    result->set_size(this->get_size());
    convert_strategy_helper(result);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::move_to(
    Csr<next_precision<ValueType>, IndexType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::convert_to(
    Coo<ValueType, IndexType>* result) const
{
    auto exec = this->get_executor();
    auto tmp = make_temporary_clone(exec, result);
    tmp->values_ = this->values_;
    tmp->col_idxs_ = this->col_idxs_;
    tmp->row_idxs_.resize_and_reset(this->get_num_stored_elements());
    tmp->set_size(this->get_size());
    exec->run(csr::make_convert_ptrs_to_idxs(
        this->get_const_row_ptrs(), this->get_size()[0], tmp->get_row_idxs()));
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::move_to(Coo<ValueType, IndexType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::convert_to(Dense<ValueType>* result) const
{
    auto exec = this->get_executor();
    auto tmp_result = make_temporary_output_clone(exec, result);
    tmp_result->resize(this->get_size());
    tmp_result->fill(zero<ValueType>());
    exec->run(csr::make_fill_in_dense(this, tmp_result.get()));
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::move_to(Dense<ValueType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::convert_to(
    Hybrid<ValueType, IndexType>* result) const
{
    auto exec = this->get_executor();
    const auto num_rows = this->get_size()[0];
    const auto num_cols = this->get_size()[1];
    array<size_type> row_nnz{exec, num_rows};
    array<int64> coo_row_ptrs{exec, num_rows + 1};
    exec->run(csr::make_convert_ptrs_to_sizes(this->get_const_row_ptrs(),
                                              num_rows, row_nnz.get_data()));
    size_type ell_lim{};
    size_type coo_nnz{};
    result->get_strategy()->compute_hybrid_config(row_nnz, &ell_lim, &coo_nnz);
    if (ell_lim > num_cols) {
        // TODO remove temporary fix after ELL gains true structural zeros
        ell_lim = num_cols;
    }
    exec->run(csr::make_compute_hybrid_coo_row_ptrs(row_nnz, ell_lim,
                                                    coo_row_ptrs.get_data()));
    coo_nnz = exec->copy_val_to_host(coo_row_ptrs.get_const_data() + num_rows);
    auto tmp = make_temporary_clone(exec, result);
    tmp->resize(this->get_size(), ell_lim, coo_nnz);
    exec->run(csr::make_convert_to_hybrid(this, coo_row_ptrs.get_const_data(),
                                          tmp.get()));
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::move_to(Hybrid<ValueType, IndexType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::convert_to(
    Sellp<ValueType, IndexType>* result) const
{
    auto exec = this->get_executor();
    const auto stride_factor = result->get_stride_factor();
    const auto slice_size = result->get_slice_size();
    const auto num_rows = this->get_size()[0];
    const auto num_slices = ceildiv(num_rows, slice_size);
    auto tmp = make_temporary_clone(exec, result);
    tmp->slice_sets_.resize_and_reset(num_slices + 1);
    tmp->slice_lengths_.resize_and_reset(num_slices);
    tmp->stride_factor_ = stride_factor;
    tmp->slice_size_ = slice_size;
    exec->run(csr::make_compute_slice_sets(this->row_ptrs_, slice_size,
                                           stride_factor, tmp->get_slice_sets(),
                                           tmp->get_slice_lengths()));
    auto total_cols =
        exec->copy_val_to_host(tmp->get_slice_sets() + num_slices);
    tmp->col_idxs_.resize_and_reset(total_cols * slice_size);
    tmp->values_.resize_and_reset(total_cols * slice_size);
    tmp->set_size(this->get_size());
    exec->run(csr::make_convert_to_sellp(this, tmp.get()));
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::move_to(Sellp<ValueType, IndexType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::convert_to(
    SparsityCsr<ValueType, IndexType>* result) const
{
    result->col_idxs_ = this->col_idxs_;
    result->row_ptrs_ = this->row_ptrs_;
    if (!result->value_.get_data()) {
        result->value_ =
            array<ValueType>(result->get_executor(), {one<ValueType>()});
    }
    result->set_size(this->get_size());
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::move_to(
    SparsityCsr<ValueType, IndexType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::convert_to(
    Ell<ValueType, IndexType>* result) const
{
    auto exec = this->get_executor();
    size_type max_nnz_per_row{};
    exec->run(csr::make_compute_max_row_nnz(this->row_ptrs_, max_nnz_per_row));
    auto tmp = make_temporary_clone(exec, result);
    if (tmp->get_size() != this->get_size() ||
        tmp->num_stored_elements_per_row_ != max_nnz_per_row) {
        tmp->num_stored_elements_per_row_ = max_nnz_per_row;
        tmp->stride_ = this->get_size()[0];
        const auto storage = tmp->num_stored_elements_per_row_ * tmp->stride_;
        tmp->col_idxs_.resize_and_reset(storage);
        tmp->values_.resize_and_reset(storage);
        tmp->set_size(this->get_size());
    }
    exec->run(csr::make_convert_to_ell(this, tmp.get()));
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::move_to(Ell<ValueType, IndexType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::convert_to(
    Fbcsr<ValueType, IndexType>* result) const
{
    auto exec = this->get_executor();
    const auto bs = result->get_block_size();
    const auto row_blocks = detail::get_num_blocks(bs, this->get_size()[0]);
    const auto col_blocks = detail::get_num_blocks(bs, this->get_size()[1]);
    auto tmp = make_temporary_clone(exec, result);
    tmp->row_ptrs_.resize_and_reset(row_blocks + 1);
    tmp->set_size(this->get_size());
    exec->run(csr::make_convert_to_fbcsr(this, bs, tmp->row_ptrs_,
                                         tmp->col_idxs_, tmp->values_));
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::move_to(Fbcsr<ValueType, IndexType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::read(const mat_data& data)
{
    this->read(device_mat_data::create_from_host(this->get_executor(), data));
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::read(const device_mat_data& data)
{
    // make a copy, read the data in
    this->read(device_mat_data{this->get_executor(), data});
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::read(device_mat_data&& data)
{
    auto size = data.get_size();
    auto exec = this->get_executor();
    auto arrays = data.empty_out();
    this->row_ptrs_.resize_and_reset(size[0] + 1);
    this->set_size(size);
    this->values_ = std::move(arrays.values);
    this->col_idxs_ = std::move(arrays.col_idxs);
    const auto row_idxs = std::move(arrays.row_idxs);
    auto local_row_idxs = make_temporary_clone(exec, &row_idxs);
    exec->run(csr::make_convert_idxs_to_ptrs(local_row_idxs->get_const_data(),
                                             local_row_idxs->get_num_elems(),
                                             size[0], this->get_row_ptrs()));
    this->make_srow();
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::write(mat_data& data) const
{
    auto tmp = make_temporary_clone(this->get_executor()->get_master(), this);

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
    const array<IndexType>* permutation_indices) const
{
    GKO_ASSERT_IS_SQUARE_MATRIX(this);
    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[0]);
    auto exec = this->get_executor();
    auto permute_cpy =
        Csr::create(exec, this->get_size(), this->get_num_stored_elements(),
                    this->get_strategy());
    array<IndexType> inv_permutation(exec, this->get_size()[1]);

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
    const array<IndexType>* permutation_indices) const
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
    const array<IndexType>* permutation_indices) const
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
    const array<IndexType>* permutation_indices) const
{
    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[1]);
    auto exec = this->get_executor();
    auto permute_cpy =
        Csr::create(exec, this->get_size(), this->get_num_stored_elements(),
                    this->get_strategy());
    array<IndexType> inv_permutation(exec, this->get_size()[1]);

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
    const array<IndexType>* permutation_indices) const
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
    const array<IndexType>* permutation_indices) const
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
std::unique_ptr<Csr<ValueType, IndexType>>
Csr<ValueType, IndexType>::create_submatrix(const gko::span& row_span,
                                            const gko::span& column_span) const
{
    using Mat = Csr<ValueType, IndexType>;
    auto exec = this->get_executor();
    auto sub_mat_size = gko::dim<2>(row_span.length(), column_span.length());
    array<IndexType> row_ptrs(exec, row_span.length() + 1);
    exec->run(csr::make_calculate_nonzeros_per_row_in_span(
        this, row_span, column_span, &row_ptrs));
    exec->run(csr::make_prefix_sum(row_ptrs.get_data(), row_span.length() + 1));
    auto num_nnz =
        exec->copy_val_to_host(row_ptrs.get_data() + sub_mat_size[0]);
    auto sub_mat = Mat::create(exec, sub_mat_size,
                               std::move(array<ValueType>(exec, num_nnz)),
                               std::move(array<IndexType>(exec, num_nnz)),
                               std::move(row_ptrs), this->get_strategy());
    exec->run(csr::make_compute_submatrix(this, row_span, column_span,
                                          sub_mat.get()));
    sub_mat->make_srow();
    return sub_mat;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Csr<ValueType, IndexType>>
Csr<ValueType, IndexType>::create_submatrix(
    const index_set<IndexType>& row_index_set,
    const index_set<IndexType>& col_index_set) const
{
    using Mat = Csr<ValueType, IndexType>;
    auto exec = this->get_executor();
    if (!row_index_set.get_num_elems() || !col_index_set.get_num_elems()) {
        return Mat::create(exec);
    }
    if (row_index_set.is_contiguous() && col_index_set.is_contiguous()) {
        auto row_st = row_index_set.get_executor()->copy_val_to_host(
            row_index_set.get_subsets_begin());
        auto row_end = row_index_set.get_executor()->copy_val_to_host(
            row_index_set.get_subsets_end());
        auto col_st = col_index_set.get_executor()->copy_val_to_host(
            col_index_set.get_subsets_begin());
        auto col_end = col_index_set.get_executor()->copy_val_to_host(
            col_index_set.get_subsets_end());

        return this->create_submatrix(span(row_st, row_end),
                                      span(col_st, col_end));
    } else {
        auto submat_num_rows = row_index_set.get_num_elems();
        auto submat_num_cols = col_index_set.get_num_elems();
        auto sub_mat_size = gko::dim<2>(submat_num_rows, submat_num_cols);
        array<IndexType> row_ptrs(exec, submat_num_rows + 1);
        exec->run(csr::make_calculate_nonzeros_per_row_in_index_set(
            this, row_index_set, col_index_set, row_ptrs.get_data()));
        exec->run(
            csr::make_prefix_sum(row_ptrs.get_data(), submat_num_rows + 1));
        auto num_nnz =
            exec->copy_val_to_host(row_ptrs.get_data() + sub_mat_size[0]);
        auto sub_mat = Mat::create(exec, sub_mat_size,
                                   std::move(array<ValueType>(exec, num_nnz)),
                                   std::move(array<IndexType>(exec, num_nnz)),
                                   std::move(row_ptrs), this->get_strategy());
        exec->run(csr::make_compute_submatrix_from_index_set(
            this, row_index_set, col_index_set, sub_mat.get()));
        sub_mat->make_srow();
        return sub_mat;
    }
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


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::scale_impl(const LinOp* alpha)
{
    auto exec = this->get_executor();
    exec->run(csr::make_scale(make_temporary_conversion<ValueType>(alpha).get(),
                              this));
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::inv_scale_impl(const LinOp* alpha)
{
    auto exec = this->get_executor();
    exec->run(csr::make_inv_scale(
        make_temporary_conversion<ValueType>(alpha).get(), this));
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::add_scaled_identity_impl(const LinOp* const a,
                                                         const LinOp* const b)
{
    bool has_diags{false};
    this->get_executor()->run(
        csr::make_check_diagonal_entries(this, has_diags));
    if (!has_diags) {
        GKO_UNSUPPORTED_MATRIX_PROPERTY(
            "The matrix has one or more structurally zero diagonal entries!");
    }
    this->get_executor()->run(csr::make_add_scaled_identity(
        make_temporary_conversion<ValueType>(a).get(),
        make_temporary_conversion<ValueType>(b).get(), this));
}


#define GKO_DECLARE_CSR_MATRIX(ValueType, IndexType) \
    class Csr<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_MATRIX);


}  // namespace matrix
}  // namespace gko
