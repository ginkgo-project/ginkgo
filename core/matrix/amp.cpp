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


GKO_REGISTER_OPERATION(spmv, amp::spmv);
GKO_REGISTER_OPERATION(advanced_spmv, amp::advanced_spmv);
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
        EnableLinOp<AMP>::operator=(other);
        for (int i = 0; i < num_precisions; i++) {
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
    }
    return *this;
}


template <typename ValueType, typename IndexType>
void AMP<ValueType, IndexType>::apply_impl(const LinOp* b, LinOp* x) const
{
    mixed_precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->get_executor()->run(amp::make_spmv(this, dense_b, dense_x));
        },
        b, x);
}


template <typename ValueType, typename IndexType>
void AMP<ValueType, IndexType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                           const LinOp* beta, LinOp* x) const
{
    mixed_precision_dispatch_real_complex<ValueType>(
        [this, alpha, beta](auto dense_b, auto dense_x) {
            auto d_alpha = make_temporary_conversion<ValueType>(alpha);
            auto d_beta = make_temporary_conversion<
                typename std::decay_t<decltype(*dense_x)>::value_type>(beta);
            this->get_executor()->run(amp::make_advanced_spmv(
                d_alpha.get(), this, dense_b, d_beta.get(), dense_x));
        },
        b, x);
}


template <typename ValueType, typename IndexType, typename Fn, typename... Args>
auto dispatch_to_concrete_matrix_type(const LinOp* mat, Fn fn, Args... args)
{
    auto a = dynamic_cast<const matrix::Ell<ValueType, IndexType>*>(mat);
    if (a) {
        return fn(a, args...);
    } else {
        auto b = dynamic_cast<const matrix::Csr<ValueType, IndexType>*>(mat);
        if (b) {
            return fn(b, args...);
        } else {
            GKO_NOT_SUPPORTED(mat);
            // will never reach the following line but required for auto return
            // return decltype(fn(a, args...)){};
        }
    }
}


template <typename ValueType, typename IndexType>
auto generate_amp_impl(const matrix::Ell<ValueType, IndexType>* const mtx,
                       std::shared_ptr<const Executor> exec, const float tol)
{
    gko::amp::array_prec<int, ValueType> max_nnz;
    gko::array<remove_complex<ValueType>> rownorms(exec, mtx->get_size()[0]);
    exec->run(
        amp::make_generate_ell_rownorms_storage(mtx, tol, max_nnz, rownorms));

    auto abins = gko::amp::allocate_bins<ValueType, IndexType>(
        exec, mtx->get_size(), max_nnz);
    constexpr auto num_bins = std::tuple_size<decltype(abins)>::value;
    static_assert(num_bins == AMP<ValueType, IndexType>::num_precisions,
                  "Wrong number of bins!");
    gko::amp::array_prec<gko::LinOp*, ValueType> amat;
    gko::constexpr_for<0, num_bins, 1>(
        [&](auto k) { amat[k] = abins[k].get(); });

    exec->run(amp::make_generate_ell_scatter_bins(mtx, tol, amat));

    gko::amp::array_prec<std::unique_ptr<const LinOp>, ValueType> cabins;
    for (int i = 0; i < matrix::AMP<ValueType, IndexType>::num_precisions;
         i++) {
        cabins[i] = std::move(abins[i]);
    }
    return cabins;
}


template <typename ValueType, typename IndexType>
std::array<std::unique_ptr<const LinOp>,
           AMP<ValueType, IndexType>::num_precisions>
AMP<ValueType, IndexType>::generate_amp(const LinOp* const mtx) const
{
    const auto tol = parameters_.tolerance;
    // typedef std::array<std::unique_ptr<const LinOp>, num_precisions>
    // ret_type;
    auto a = dynamic_cast<const matrix::Ell<ValueType, IndexType>*>(mtx);
    if (a) {
        return generate_amp_impl<ValueType, IndexType>(a, this->get_executor(),
                                                       tol);
    } else {
        GKO_NOT_SUPPORTED(mtx);
        // return decltype(mat_bins_){};
    }
}


template <typename ValueType, typename IndexType>
void AMP<ValueType, IndexType>::convert_to(Dense<ValueType>* const result) const
{
    auto exec = this->get_executor();
    auto tmp_result = make_temporary_output_clone(exec, result);
    tmp_result->resize(this->get_size());
    tmp_result->fill(zero<ValueType>());
    exec->run(amp::make_fill_in_dense(this, tmp_result.get()));
}


template <typename ValueType, typename IndexType>
void AMP<ValueType, IndexType>::move_to(Dense<ValueType>* const result)
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
    std::array<std::unique_ptr<const LinOp>, num_precisions>&& matrix_bins)
    : EnableLinOp<AMP<ValueType, IndexType>>(matrix_bins[0]->get_executor(),
                                             matrix_bins[0]->get_size()),
      mat_bins_{std::move(matrix_bins)}
{
    GKO_ENSURE_ALLOCATED(mat_bins_[0].get(),
                         this->get_executor()->get_description(), 1);
    for (int i = 0; i < num_precisions - 1; i++) {
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
