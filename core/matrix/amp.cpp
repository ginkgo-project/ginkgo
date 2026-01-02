// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/temporary_clone.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/amp.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/ell.hpp>

#include "core/base/allocator.hpp"
#include "core/base/array_access.hpp"
#include "core/base/device_matrix_data_kernels.hpp"
#include "core/components/absolute_array_kernels.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/amp_kernels.hpp"


namespace gko {
namespace matrix {
namespace amp {


template <typename ValueType, typename IndexType>
using bin_mtx_type = gko::matrix::Ell<ValueType, IndexType>;


namespace {


// GKO_REGISTER_OPERATION(spmv, amp::spmv);
// GKO_REGISTER_OPERATION(advanced_spmv, amp::advanced_spmv);
GKO_REGISTER_OPERATION(convert_idxs_to_ptrs, components::convert_idxs_to_ptrs);
GKO_REGISTER_OPERATION(fill_in_dense, amp::fill_in_dense);
GKO_REGISTER_OPERATION(generate_ell_rownorms_storage,
                       amp::generate_ell_rownorms_storage);
GKO_REGISTER_OPERATION(generate_ell_scatter_bins,
                       amp::generate_ell_scatter_bins);
GKO_REGISTER_OPERATION(extract_diagonal, amp::extract_diagonal);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(prefix_sum_nonnegative,
                       components::prefix_sum_nonnegative);


}  // anonymous namespace
}  // namespace amp


template <typename ValueType, typename IndexType>
AMP<ValueType, IndexType>& AMP<ValueType, IndexType>::operator=(
    const AMP& other)
{
    if (&other != this) {
        // const auto old_size = this->get_size();
        EnableLinOp<AMP>::operator=(other);
        this->n_bins_ = other.n_bins_;
        for (int i = 0; i < this->n_bins_; i++) {
            auto tmtx = amp::bin_mtx_type<ValueType, IndexType>::create(
                this->get_executor());
            tmtx->copy_from(other.mat_bins_[i].get());
            this->mat_bins_[i] = std::move(tmtx);
        }
    }
    return *this;
}


template <typename ValueType, typename IndexType>
AMP<ValueType, IndexType>& AMP<ValueType, IndexType>::operator=(AMP&& other)
{
    if (&other != this) {
        EnableLinOp<AMP>::operator=(std::move(other));
        mat_bins_ = std::move(other.mat_bins_);
        n_bins_ = std::exchange(other.n_bins_, 0);
    }
    return *this;
}


template <typename ValueType, typename IndexType>
void AMP<ValueType, IndexType>::apply_impl(const LinOp* b, LinOp* x) const
{
    GKO_NOT_IMPLEMENTED;
    auto exec = this->get_executor();
    auto one = Dense<ValueType>::create(exec, dim<2>{1, 1},
                                        array<ValueType>(exec, {1.0}), 1);
    constexpr int first_idx =
        gko::amp::precision_index<remove_complex<ValueType>>::index;

    mat_bins_[first_idx]->apply(b, x);

    for (int ip = first_idx + 1; ip < num_precisions; ip++) {
        mat_bins_[ip]->apply(one.get(), b, one.get(), x);
    }
}


template <typename ValueType, typename IndexType>
void AMP<ValueType, IndexType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                           const LinOp* beta, LinOp* x) const
{
    GKO_NOT_IMPLEMENTED;
    // constexpr int first_idx =
    //     gko::amp::precision_index<remove_complex<ValueType>>::index;

    // mat_bins_[first_idx]->apply(alpha, b, beta x);

    // for(int ip = first_idx+1; ip < num_precisions; ip++) {
    //     mat_bins_[ip]->apply(one.get(), b, one.get(), x);
    // }
}


template <typename ValueType, typename IndexType>
void AMP<ValueType, IndexType>::generate_amp(const LinOp* const mtx)
{
    this->set_size(mtx->get_size());
}


template <typename ValueType, typename IndexType>
void AMP<ValueType, IndexType>::convert_to(Dense<ValueType>* result) const
{
    auto exec = this->get_executor();
    auto tmp_result = make_temporary_output_clone(exec, result);
    tmp_result->resize(this->get_size());
    tmp_result->fill(zero<ValueType>());
    exec->run(amp::make_fill_in_dense(this, tmp_result.get()));
}


template <typename ValueType, typename IndexType>
void AMP<ValueType, IndexType>::move_to(Dense<ValueType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Diagonal<ValueType>>
AMP<ValueType, IndexType>::extract_diagonal() const
{
    auto exec = this->get_executor();

    const auto diag_size = std::min(this->get_size()[0], this->get_size()[1]);
    auto diag = Diagonal<ValueType>::create(exec, diag_size);
    exec->run(amp::make_fill_array(diag->get_values(), diag->get_size()[0],
                                   zero<ValueType>()));
    exec->run(amp::make_extract_diagonal(this, diag.get()));
    return diag;
}


template <typename ValueType, typename IndexType, typename Fn, typename... Args>
void dispatch_to_concrete_matrix_type(LinOp* mat, Fn fn, Args... args)
{
    auto a = dynamic_cast<matrix::Ell<ValueType, IndexType>*>(mat);
    if (a) {
        fn(a, args...);
    } else {
        auto b = dynamic_cast<matrix::Csr<ValueType, IndexType>*>(mat);
        if (b) {
            fn(b, args...);
        } else {
            GKO_NOT_SUPPORTED(mat);
        }
    }
}


template <typename ValueType, typename IndexType>
AMP<ValueType, IndexType>::AMP(std::shared_ptr<const Executor> exec)
    : EnableLinOp<AMP<ValueType, IndexType>>(std::move(exec))
{}


template <typename ValueType, typename IndexType>
AMP<ValueType, IndexType>::AMP(const AMP& other) : AMP(other.get_executor())
{
    *this = other;
}


template <typename ValueType, typename IndexType>
AMP<ValueType, IndexType>::AMP(AMP&& other) : AMP(other.get_executor())
{
    *this = std::move(other);
}


template <typename ValueType, typename IndexType>
AMP<ValueType, IndexType>::AMP(
    const int num_bins,
    std::array<std::unique_ptr<const LinOp>, num_precisions>&& matrix_bins)
    : EnableLinOp<AMP<ValueType, IndexType>>(matrix_bins[0]->get_executor(),
                                             matrix_bins[0]->get_size()),
      n_bins_{num_bins},
      mat_bins_{std::move(matrix_bins)}
{
    GKO_ENSURE_ALLOCATED(mat_bins_[0].get(),
                         this->get_executor()->get_description(), 1);
    for (int i = 0; i < num_bins - 1; i++) {
        GKO_ENSURE_ALLOCATED(mat_bins_[i + 1].get(),
                             this->get_executor()->get_description(), 1);
        GKO_ASSERT_EQUAL_DIMENSIONS(mat_bins_[i], mat_bins_[i + 1]);
    }
}


#define GKO_DECLARE_AMP_MATRIX(ValueType, IndexType) \
    class AMP<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE_BASE(GKO_DECLARE_AMP_MATRIX);


}  // namespace matrix
}  // namespace gko
